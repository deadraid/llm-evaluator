import asyncio
import time
from typing import Dict, Optional

from datasets import Dataset, load_dataset
from huggingface_hub import login
from tqdm import tqdm

from .config import DatasetConfig, LLMConfig
from .metrics import MetricsCalculator
from .metrics_config import MetricsConfig
from .processor import ItemProcessor
from .result_logger import ResultLogger


class LLMEvaluator:
    def __init__(
            self,
            solver_config: LLMConfig,
            validator_config: LLMConfig,
            dataset_config: DatasetConfig,
            metrics_config: MetricsConfig,
            auth_token: str,
            concurrent: bool = False,
            max_concurrent_tasks: int = 3
    ):
        self.solver_config = solver_config
        self.validator_config = validator_config
        self.dataset_config = dataset_config
        self.metrics_config = metrics_config
        self.auth_token = auth_token
        self.concurrent = concurrent
        self.max_concurrent_tasks = max_concurrent_tasks

        # Initialize components
        self.processor = ItemProcessor(solver_config, validator_config, dataset_config, metrics_config)
        self.metrics_calculator = MetricsCalculator(metrics_config)
        self.result_logger = ResultLogger(dataset_config, solver_config, validator_config, metrics_config)

        # Initialize results tracking
        self.results = {
            "total": 0,
            "timestamps": [],
            "detailed_results": [],
            "metrics": {f"pass@{k}": 0 for k in metrics_config.k_values}
        }

        # Store dataset
        self._dataset: Optional[Dataset] = None

    def _update_results(self, result: Dict, idx: int, start_time: float, pbar: tqdm):
        """Update results with new item and progress bar."""
        if not isinstance(result, dict) or 'pass_k_results' not in result:
            self.result_logger.log_warning(f"Invalid result format at index {idx}")
            return False

        self.results['total'] += 1
        self.results['detailed_results'].append(result)
        self.results['timestamps'].append(time.time())

        # Update metrics
        for metric, value in result['pass_k_results'].items():
            if value:
                self.results['metrics'][metric] += 1

        # Update progress bar
        if len(self.results['timestamps']) > 1:
            avg_time = (self.results['timestamps'][-1] - start_time) / len(self.results['detailed_results'])
            pass_1_pct = (self.results['metrics'].get('pass@1', 0) / self.results['total']) * 100

            pbar.set_postfix({
                'Pass@1': f"{pass_1_pct:.1f}%",
                'avg_time': f"{avg_time:.1f}s/q"
            })

        return True

    def load_dataset(self) -> Dataset:
        """Load and cache dataset from Hugging Face."""
        if self._dataset is not None:
            return self._dataset

        try:
            if self.auth_token:
                login(token=self.auth_token)

            # Load the dataset with caching enabled by default
            dataset = load_dataset(
                self.dataset_config.dataset_name,
                self.dataset_config.config_name,
                split=self.dataset_config.split,
                cache_dir=".cache/huggingface/datasets"  # Explicit cache directory
            )

            if not dataset:
                raise ValueError(f"Failed to load dataset: {self.dataset_config.dataset_name}")

            # Log dataset info
            self.result_logger.log_info(f"Loaded dataset: {self.dataset_config.dataset_name}")
            self.result_logger.log_info(f"Dataset size: {len(dataset)} items")

            # Apply dataset limit if specified
            if self.dataset_config.limit:
                dataset = dataset.select(range(min(self.dataset_config.limit, len(dataset))))
                self.result_logger.log_info(f"Applied limit: {self.dataset_config.limit} items")

            # Cache the dataset
            self._dataset = dataset
            return dataset

        except Exception as e:
            self.result_logger.log_error(f"Error loading dataset: {str(e)}", include_exc_info=True)
            raise

    def get_item_fields(self, batch_item, index: int = 0) -> Dict:
        """Extract relevant fields from dataset batch item."""
        # Handle both single items and batched items
        if isinstance(batch_item, dict):
            # Basic fields
            fields = {
                'question': batch_item[self.dataset_config.question_field][index] if isinstance(batch_item[self.dataset_config.question_field], list)
                else batch_item[self.dataset_config.question_field],
                'answer': batch_item[self.dataset_config.answer_field][index] if isinstance(batch_item[self.dataset_config.answer_field], list)
                else batch_item[self.dataset_config.answer_field],
            }

            # Handle multiple choice questions
            if self.dataset_config.multiple_choice:
                if self.dataset_config.choices_field:
                    # Handle single choices field
                    choices = batch_item.get(self.dataset_config.choices_field)
                    if choices is not None:
                        fields['choices'] = choices[index] if isinstance(choices, list) else choices
                elif self.dataset_config.choice_fields:
                    # Handle multiple choice fields (like in GPQA)
                    choices = []
                    for field in self.dataset_config.choice_fields:
                        value = batch_item.get(field)
                        if value is not None:
                            choice_value = value[index] if isinstance(value, list) else value
                            choices.append(choice_value)
                    if choices:
                        fields['choices'] = choices

            # Add optional fields if they exist
            for field_name, field_key in [
                (self.dataset_config.subject_field, 'subject'),
                (self.dataset_config.level_field, 'level')
            ]:
                if field_name and field_name in batch_item:
                    value = batch_item[field_name]
                    fields[field_key] = value[index] if isinstance(value, list) else value

            return fields
        else:
            # If the item is not a dictionary (e.g., if it's a dataset row), handle it directly
            return batch_item

    async def process_batch(self, items, pbar: tqdm, start_time: float):
        """Process a batch of items concurrently."""
        if not items:
            return

        try:
            # Process batch items
            processed_items = []
            for i in range(len(items[self.dataset_config.question_field])):
                processed_item = self.get_item_fields(items, i)
                processed_items.append(processed_item)

            if not processed_items:
                self.result_logger.log_warning("No items could be processed in this batch")
                return

            # Process items concurrently
            tasks = [self.processor.process_single_item(item) for item in processed_items]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            success_count = 0
            for idx, result in enumerate(results):
                try:
                    if isinstance(result, Exception):
                        self.result_logger.log_error(f"Error processing item {idx}: {str(result)}")
                        continue

                    if result and self._update_results(result, idx, start_time, pbar):
                        success_count += 1

                except Exception as e:
                    self.result_logger.log_error(f"Error handling result {idx}: {str(e)}")
                finally:
                    pbar.update(1)

            if success_count == 0:
                self.result_logger.log_warning("No items were processed successfully in this batch")

        except Exception as e:
            self.result_logger.log_error(f"Error in batch processing: {str(e)}", include_exc_info=True)

    async def evaluate_dataset_async(self):
        """Evaluate entire dataset asynchronously."""
        dataset = self.load_dataset()
        total_items = len(dataset)

        start_time = time.time()
        pbar = tqdm(total=total_items, desc="Evaluating", unit="questions")

        try:
            if self.concurrent:
                # Process in batches
                batch_size = self.max_concurrent_tasks
                for i in range(0, total_items, batch_size):
                    batch_end = min(i + batch_size, total_items)
                    batch_items = dataset[i:batch_end]
                    await self.process_batch(batch_items, pbar, start_time)

                    if i % 10 == 0:
                        self.result_logger.log_progress(self.results)
            else:
                # Sequential processing
                for idx, item in enumerate(dataset):
                    processed_item = self.get_item_fields(item)
                    result = await self.processor.process_single_item(processed_item)
                    if result:
                        self._update_results(result, idx, start_time, pbar)
                    pbar.update(1)

                    if (idx + 1) % 10 == 0:
                        self.result_logger.log_progress(self.results)

        except Exception as e:
            self.result_logger.log_error(f"Error in evaluation: {str(e)}")
            raise
        finally:
            pbar.close()
            if self.results['total'] > 0:
                self.result_logger.log_final_results(self.results)

    def evaluate_dataset(self):
        """Synchronous wrapper for evaluate_dataset_async."""
        try:
            asyncio.run(self.evaluate_dataset_async())
        except KeyboardInterrupt:
            self.result_logger.log_info("\nEvaluation interrupted by user")
            if self.results['total'] > 0:
                self.result_logger.log_final_results(self.results)
        except Exception as e:
            self.result_logger.log_error(f"Error in evaluation: {str(e)}")
            raise
