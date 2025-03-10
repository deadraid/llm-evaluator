import logging
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMConfig:
    """Configuration for LLM client"""
    api_key: str
    base_url: str
    model: str
    system_prompt: str
    temperature: float = float(os.getenv("MODEL_TEMPERATURE", "0.7"))  # Default temperature of 0.7
    max_tokens: int = int(os.getenv("MODEL_MAX_TOKENS", "4096"))      # Default max tokens of 4096

@dataclass
class DatasetConfig:
    """Configuration for dataset processing"""
    dataset_name: str
    question_field: str
    answer_field: str
    split: str = "test"
    multiple_choice: bool = False
    choices_field: Optional[str] = None
    choice_fields: Optional[list[str]] = None
    config_name: Optional[str] = None
    limit: Optional[int] = None
    subject_field: Optional[str] = None
    level_field: Optional[str] = None

def parse_test_limit() -> Optional[int]:
    """Helper function to parse the test limit from environment variable"""
    limit = os.getenv("TEST_LIMIT")
    if limit is not None:
        try:
            return int(limit)
        except ValueError:
            logging.warning(f"Invalid TEST_LIMIT value: {limit}. Using no limit.")
    return None

# Dataset configurations
DATASET_CONFIGS = {
    "gsm8k": DatasetConfig(
        dataset_name="openai/gsm8k",
        config_name="main",
        question_field="question",
        answer_field="answer",
        split="test",
        limit=parse_test_limit()
    ),
    "mathematics": DatasetConfig(
        dataset_name="hendrycks/mathematics",
        question_field="problem",
        answer_field="solution",
        split="test",
        multiple_choice=True,
        choices_field="choices",
        subject_field="subject",  # Mathematics dataset includes subject categorization
        limit=parse_test_limit()
    ),
    "math-500": DatasetConfig(
        dataset_name="HuggingFaceH4/MATH-500",
        question_field="problem",
        answer_field="solution",
        split="test",
        subject_field="topic",  # MATH-500 includes topic categorization
        level_field="difficulty", # MATH-500 includes difficulty levels
        limit=parse_test_limit()
    ),
    "mmlu": DatasetConfig(
        dataset_name="cais/mmlu",
        question_field="question",
        answer_field="answer",
        split="test",
        multiple_choice=True,
        choices_field="choices",
        subject_field="subject",  # MMLU includes subject categorization
        limit=parse_test_limit()
    ),
    "hellaswag": DatasetConfig(
        dataset_name="hellaswag",
        question_field="ctx",
        answer_field="label",
        split="validation",
        multiple_choice=True,
        choices_field="endings",
        limit=parse_test_limit()
    ),
    "truthful_qa": DatasetConfig(
        dataset_name="truthful_qa",
        question_field="question",
        answer_field="correct_answers",
        split="validation",
        limit=parse_test_limit()
    ),
    "gpqa": DatasetConfig(
        dataset_name="Idavidrein/gpqa",
        config_name="gpqa_diamond",
        question_field="Question",
        answer_field="Correct Answer",
        split="train",
        multiple_choice=True,
        choice_fields=["Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3"],
        subject_field="High-level domain",
        level_field="Writer's Difficulty Estimate",
        limit=parse_test_limit()
    ),
    "aime": DatasetConfig(
        dataset_name="AI-MO/aimo-validation-aime",
        question_field="problem",
        answer_field="answer",
        split="train",
        limit=parse_test_limit()
    ),
    "aime-2025": DatasetConfig(
        dataset_name="rawsh/aime_2025",
        question_field="problem",
        answer_field="answer",
        split="train",
        limit=parse_test_limit()
    ),
    "aime-2024": DatasetConfig(
        dataset_name="Maxwell-Jia/AIME_2024",
        question_field="Problem",
        answer_field="Answer",
        split="train",
        limit=parse_test_limit()
    )
}

def get_model_config(role: str) -> LLMConfig:
    """Get configuration for either solver or validator model with environment variable overrides"""
    api_key = os.getenv(f"{role}_API_KEY")
    base_url = os.getenv(f"{role}_API_BASE_URL", "https://api.openai.com/v1")
    model = os.getenv(f"{role}_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
    system_prompt = os.getenv(f"{role}_SYSTEM_PROMPT", "You're a helpful and harmless AI assistant. You solve problems step by step.")

    temperature = float(os.getenv(f"{role}_TEMPERATURE",
                                  os.getenv("MODEL_TEMPERATURE", "0.7")))

    max_tokens = int(os.getenv(f"{role}_MAX_TOKENS",
                               os.getenv("MODEL_MAX_TOKENS", "4096")))

    return LLMConfig(
        api_key=api_key,
        base_url=base_url,
        model=model,
        system_prompt = system_prompt,
        temperature=temperature,
        max_tokens=max_tokens
    )
