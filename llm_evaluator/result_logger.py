import json
import logging
import os
from datetime import datetime
from typing import Dict

import numpy as np

from .metrics import MetricsCalculator


class ResultLogger:
    def __init__(self, dataset_config, solver_config, validator_config, metrics_config):
        self.dataset_config = dataset_config
        self.solver_config = solver_config
        self.validator_config = validator_config
        self.metrics_config = metrics_config
        self.metrics_calculator = MetricsCalculator(metrics_config)

        # Set up logging
        self.logger = self._setup_logger()

        # Create results directory
        self.results_dir = "evaluation_results"
        os.makedirs(self.results_dir, exist_ok=True)

    def _setup_logger(self):
        """Set up logging configuration"""
        logger = logging.getLogger('llm_evaluator')
        logger.setLevel(logging.INFO)

        # Create handlers if they don't exist
        if not logger.handlers:
            c_handler = logging.StreamHandler()
            f_handler = logging.FileHandler('llm_evaluation.log')

            # Create formatters
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            c_handler.setFormatter(formatter)
            f_handler.setFormatter(formatter)

            # Add handlers
            logger.addHandler(c_handler)
            logger.addHandler(f_handler)

        return logger

    def log_info(self, message: str):
        """Log info message"""
        self.logger.info(message)

    def log_error(self, message: str, include_exc_info: bool = False):
        """Log error message"""
        self.logger.error(message, exc_info=include_exc_info)

    def log_warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)

    def log_progress(self, results: Dict):
        """Log current progress"""
        try:
            if not results or results.get('total', 0) == 0:
                self.log_info("No results to log yet")
                return

            metrics = self.metrics_calculator.calculate_metrics(results)

            # Calculate timing metrics
            timestamps = results.get('timestamps', [])
            if len(timestamps) > 1:
                avg_time = np.mean(np.diff(timestamps))
            else:
                avg_time = 0.0

            # Create progress message with safe formatting
            progress_msg = [
                f"Progress: {results['total']} questions processed",
                f"Pass@1: {metrics.get('pass@1', 0):.2f}%",
                f"Avg Pass@K: {metrics.get('average_pass@k', 0):.2f}%",
                f"Avg time per question: {avg_time:.2f}s"
            ]

            self.log_info(", ".join(progress_msg))

        except Exception as e:
            self.log_error(f"Error in log_progress: {str(e)}", include_exc_info=True)

    def _update_history(self, summary_results: Dict):
        """Update the evaluation history file"""
        try:
            history_file = os.path.join(self.results_dir, "evaluation_history.json")

            # Load existing history or create new if doesn't exist
            history = []
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    history = json.load(f)

            # Add new results to history
            history.append(summary_results)

            # Save updated history
            with open(history_file, 'w') as f:
                json.dump(history, f, indent=2)

            self.log_info(f"Updated evaluation history in: {history_file}")

        except Exception as e:
            self.log_error(f"Error updating evaluation history: {str(e)}", include_exc_info=True)

    def log_final_results(self, results: Dict):
        """Log final evaluation results with all metrics"""
        try:
            if not results or results.get('total', 0) == 0:
                self.log_info("No results to log")
                return

            metrics = self.metrics_calculator.calculate_metrics(results)
            timestamps = results.get('timestamps', [])

            avg_time = 0.0
            if len(timestamps) > 1:
                avg_time = np.mean(np.diff(timestamps))

            # Prepare summary results
            summary_results = {
                "total_questions": results['total'],
                "average_time_per_question": f"{avg_time:.2f}s",
                "average_time_raw": float(avg_time),
                "dataset": self.dataset_config.dataset_name,
                "solver_model": self.solver_config.model,
                "solver_temperature": self.solver_config.temperature,
                "solver_max_tokens": self.solver_config.max_tokens,
                "validator_model": self.validator_config.model,
                "validator_temperature": self.validator_config.temperature,
                "validator_max_tokens": self.validator_config.max_tokens,
                "metrics": metrics,
                "metrics_config": {
                    "k_values": self.metrics_config.k_values,
                    "weighted_k": self.metrics_config.weighted_k,
                    "weights": self.metrics_config.weights
                },
                "timestamp": datetime.now().isoformat()
            }

            # Create timestamped filenames
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            dataset_name = self.dataset_config.dataset_name.replace('/', '_')
            summary_filename = os.path.join(
                self.results_dir,
                f"summary_results_{dataset_name}_{timestamp_str}.json"
            )
            detailed_filename = os.path.join(
                self.results_dir,
                f"detailed_results_{dataset_name}_{timestamp_str}.json"
            )

            # Save results
            with open(summary_filename, 'w') as f:
                json.dump(summary_results, f, indent=2)

            with open(detailed_filename, 'w') as f:
                json.dump(results['detailed_results'], f, indent=2)

            # Update history and log summary
            self._update_history(summary_results)
            self._log_final_summary(summary_results, summary_filename, detailed_filename)

        except Exception as e:
            self.log_error(f"Error in log_final_results: {str(e)}", include_exc_info=True)

    def _log_final_summary(self, summary_results: Dict, summary_filename: str, detailed_filename: str):
        """Log final summary of results"""
        self.log_info("\nEvaluation completed!")
        self.log_info("Results saved to:")
        self.log_info(f"  - Summary: {summary_filename}")
        self.log_info(f"  - Detailed results: {detailed_filename}")

        # Log metrics
        self.log_info("\nMetrics:")
        for metric, value in summary_results['metrics'].items():
            if isinstance(value, (int, float)):
                self.log_info(f"{metric}: {value:.2f}%")
            else:
                self.log_info(f"{metric}: {value}")

        # Log configuration
        self.log_info("\nConfiguration:")
        for key, value in summary_results.items():
            if key not in ['metrics', 'metrics_config']:
                self.log_info(f"{key}: {value}")
