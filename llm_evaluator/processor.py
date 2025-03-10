import asyncio
import random
from datetime import datetime
from typing import Any, Dict, List, Optional

from openai import OpenAI

from .config import DatasetConfig, LLMConfig
from .metrics_config import MetricsConfig


class ItemProcessor:
    def __init__(
            self,
            solver_config: LLMConfig,
            validator_config: LLMConfig,
            dataset_config: DatasetConfig,
            metrics_config: MetricsConfig,
    ):
        self.metrics_config = metrics_config
        self.solver_config = solver_config
        self.validator_config = validator_config
        self.dataset_config = dataset_config

        # Initialize clients
        self.solver_client = OpenAI(
            api_key=solver_config.api_key,
            base_url=solver_config.base_url
        )
        self.validator_client = OpenAI(
            api_key=validator_config.api_key,
            base_url=validator_config.base_url
        )

    def format_prompt(self, question: str, choices: Optional[List[str]] = None) -> str:
        """Format prompt based on whether it's multiple choice or not."""
        if self.dataset_config.multiple_choice and choices:
            # Ensure choices are randomly ordered to avoid position bias
            shuffled_choices = choices.copy()
            random.shuffle(shuffled_choices)
            choices_text = "\n".join([f"{i+1}. {choice}" for i, choice in enumerate(shuffled_choices)])
            return f"Question: {question}\n\nChoices:\n{choices_text}\n\nAnswer:"
        return f"Question: {question}\n\nProvide your answer:"

    def format_validation_prompt(self, question: str, llm_answer: str, correct_answer: str) -> str:
        """Format prompt for validation."""
        return f"""Question: {question}

LLM's Answer: {llm_answer}

Correct Answer: {correct_answer}

Is the LLM's answer correct? Please respond with only 'yes' or 'no' and briefly explain why."""

    async def get_llm_response(self, client: OpenAI, prompt: str, config: LLMConfig) -> str:
        """Get single response from LLM."""
        try:
            if not prompt or not isinstance(prompt, str):
                print("Invalid prompt format")
                return ""

            response = await asyncio.to_thread(
                client.chat.completions.create,
                model=config.model,
                messages=[
                    {"role": "system", "content":config.system_prompt},
                          {"role": "user", "content": prompt}
                ],
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                top_p=0.95,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error getting LLM response: {e}")
            return ""

    async def get_llm_responses(self, client: OpenAI, prompt: str, config: LLMConfig, num_samples: int = 1) -> List[str]:
        """Get multiple independent responses from LLM."""
        tasks = []
        for _ in range(num_samples):
            tasks.append(self.get_llm_response(client, prompt, config))
        return await asyncio.gather(*tasks)

    async def process_single_item(self, item: Dict[str, Any]) -> Optional[Dict]:
        """Process a single dataset item."""
        try:
            # Extract required fields
            question = item['question']
            answer = item['answer']
            choices = item.get('choices') if self.dataset_config.multiple_choice else None

            # Format solver prompt
            solver_prompt = self.format_prompt(question, choices)

            # Get solver answers
            solver_answers = await self.get_llm_responses(
                self.solver_client,
                solver_prompt,
                self.solver_config,
                max(self.metrics_config.k_values)
            )

            if not any(solver_answers):
                print("No valid responses from solver")
                return None

            # Get validator results
            validations = []
            validation_explanations = []

            for solver_answer in solver_answers:
                if not solver_answer:
                    continue

                validation_prompt = self.format_validation_prompt(
                    question=question,
                    llm_answer=solver_answer,
                    correct_answer=answer
                )

                explanation = await self.get_llm_response(
                    self.validator_client,
                    validation_prompt,
                    self.validator_config
                )

                if explanation:
                    validation_explanations.append(explanation)
                    validations.append(explanation.lower().startswith('yes'))

            if not validations:
                print("No valid validations received")
                return None

            # Format result
            result = {
                "question": solver_prompt,
                "correct_answer": answer,
                "predictions": [
                    {
                        "answer": ans,
                        "is_correct": val,
                        "validator_explanation": exp
                    }
                    for ans, val, exp in zip(solver_answers, validations, validation_explanations)
                    if ans and exp
                ],
                "pass_k_results": {
                    f"pass@{k}": any(validations[:k])
                    for k in self.metrics_config.k_values
                },
                "timestamp": datetime.now().isoformat()
            }

            # Add optional fields if they exist
            for field in ['subject', 'level']:
                if field in item:
                    result[field] = item[field]

            return result

        except Exception as e:
            print(f"Error processing item: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return None
