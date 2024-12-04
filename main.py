import os
import sys
from typing import List, Dict
from dotenv import load_dotenv, dotenv_values
from llm_evaluator.config import LLMConfig, DATASET_CONFIGS
from llm_evaluator.metrics_config import get_metrics_config
from llm_evaluator import LLMEvaluator
from logger import setup_logger

logger = setup_logger()

def load_environment_variables() -> Dict[str, str]:
    """
    Load environment variables from .env file and return them
    Returns dict of loaded variables or empty dict if loading fails
    """
    try:
        # Check for .env file
        env_file = os.path.join(os.path.dirname(__file__), '.env')
        if not os.path.exists(env_file):
            logger.warning("No .env file found in current directory")
            return {}

        # Load only .env file variables (without existing environment variables)
        env_vars = dotenv_values(env_file)

        # Load the variables into environment
        load_dotenv(env_file)

        if env_vars:
            # Log only variables from .env file
            logger.info("\nEnvironment Variables loaded from .env:")
            logger.info("-" * 40)

            # Filter out sensitive information when logging
            sensitive_keys = {'API_KEY', 'TOKEN', 'PASSWORD', 'SECRET'}
            for key, value in env_vars.items():
                if any(sensitive in key.upper() for sensitive in sensitive_keys):
                    logger.info(f"{key}: ****[HIDDEN]****")
                else:
                    logger.info(f"{key}: {value}")

        return env_vars

    except Exception as e:
        logger.error(f"Error loading environment variables: {str(e)}")
        return {}

def check_environment_variables():
    """Check if all required environment variables are set"""
    required_vars = [
        "SOLVER_API_KEY",
        "VALIDATOR_API_KEY",
        "HUGGINGFACE_TOKEN"
    ]

    # First load from .env file
    loaded_vars = load_environment_variables()

    # Then check for missing variables
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        logger.error("Please set all required environment variables in .env file or environment.")
        sys.exit(1)

def parse_dataset_names() -> List[str]:
    """Parse dataset names from environment variable"""
    dataset_names = os.getenv("DATASET_NAMES", "gsm8k").strip()
    return [name.strip() for name in dataset_names.split(",")]

def validate_dataset_names(dataset_names: List[str]) -> List[str]:
    """Validate dataset names and return valid ones"""
    valid_datasets = []
    invalid_datasets = []
    available_datasets = list(DATASET_CONFIGS.keys())

    for name in dataset_names:
        if name in DATASET_CONFIGS:
            valid_datasets.append(name)
        else:
            invalid_datasets.append(name)

    if invalid_datasets:
        logger.warning(f"Invalid dataset names found: {', '.join(invalid_datasets)}")
        logger.info(f"Available datasets: {', '.join(available_datasets)}")

    if not valid_datasets:
        logger.error("No valid dataset names provided. Exiting.")
        sys.exit(1)

    return valid_datasets

def get_model_config(role: str) -> LLMConfig:
    """Get configuration for either solver or validator model with environment variable overrides"""
    api_key = os.getenv(f"{role}_API_KEY")
    base_url = os.getenv(f"{role}_API_BASE_URL", "https://api.openai.com/v1")
    model = os.getenv(f"{role}_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")

    temperature = float(os.getenv(f"{role}_TEMPERATURE",
                                  os.getenv("MODEL_TEMPERATURE", "0.7")))

    max_tokens = int(os.getenv(f"{role}_MAX_TOKENS",
                               os.getenv("MODEL_MAX_TOKENS", "4096")))

    return LLMConfig(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens
    )

def parse_concurrency_settings():
    """Parse concurrency settings from environment variables"""
    concurrent = os.getenv("CONCURRENT_EXECUTION", "false").lower() == "true"
    max_tasks = int(os.getenv("MAX_CONCURRENT_TASKS", "3"))
    return concurrent, max_tasks

def run_evaluation(dataset_name: str, solver_config: LLMConfig, validator_config: LLMConfig,
                   concurrent: bool, max_concurrent_tasks: int):
    """Run evaluation for a single dataset"""
    try:
        dataset_config = DATASET_CONFIGS[dataset_name]
        metrics_config = get_metrics_config()

        logger.info(f"\nStarting evaluation for dataset: {dataset_name}")
        logger.info("=" * 50)
        logger.info(f"Using K values: {metrics_config.k_values}")
        if metrics_config.weighted_k:
            logger.info(f"Using weighted Pass@K with weights: {metrics_config.weights}")

        if concurrent:
            logger.info(f"Running in concurrent mode with max {max_concurrent_tasks} concurrent tasks")
        else:
            logger.info("Running in sequential mode")

        evaluator = LLMEvaluator(
            solver_config=solver_config,
            validator_config=validator_config,
            dataset_config=dataset_config,
            metrics_config=metrics_config,
            auth_token=os.getenv("HUGGINGFACE_TOKEN"),
            concurrent=concurrent,
            max_concurrent_tasks=max_concurrent_tasks
        )

        evaluator.evaluate_dataset()
        logger.info(f"Completed evaluation for {dataset_name}")
        logger.info("=" * 50 + "\n")

    except Exception as e:
        logger.error(f"Error evaluating dataset {dataset_name}: {str(e)}")
        logger.info("Continuing with next dataset if available...")

def main():
    try:
        # Check environment variables (now includes .env loading)
        check_environment_variables()

        # Get and validate dataset names
        dataset_names = parse_dataset_names()
        valid_datasets = validate_dataset_names(dataset_names)

        # Get model configurations
        solver_config = get_model_config("SOLVER")
        validator_config = get_model_config("VALIDATOR")

        # Get concurrency settings
        concurrent, max_concurrent_tasks = parse_concurrency_settings()

        # Print evaluation plan
        logger.info(f"Planning to evaluate {len(valid_datasets)} datasets:")
        for idx, name in enumerate(valid_datasets, 1):
            logger.info(f"{idx}. {name}")

        # Run evaluations sequentially
        for dataset_name in valid_datasets:
            run_evaluation(
                dataset_name,
                solver_config,
                validator_config,
                concurrent,
                max_concurrent_tasks
            )

        logger.info("All evaluations completed successfully!")

    except KeyboardInterrupt:
        logger.info("\nEvaluation interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"An error occurred during evaluation: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    # Print banner
    print("""
╔══════════════════════════════════════════╗
║         LLM Evaluation Framework         ║
╚══════════════════════════════════════════╝
    """)

    main()
