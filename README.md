# LLM Evaluator

A comprehensive framework for evaluating Large Language Model (LLM) performance across various datasets using a solver-validator approach. This framework supports concurrent evaluation, multiple choice and open-ended questions, and provides detailed metrics and analysis.

## Features

- Support for multiple popular benchmark datasets (GSM8K, MMLU, Mathematics, etc.)
- Configurable solver and validator models
- Concurrent evaluation capabilities
- Customizable Pass@K metrics
- Detailed logging and result analysis
- Support for multiple choice and open-ended questions
- Subject and difficulty level tracking

## Installation

This project uses Poetry for dependency management. To get started:

1. Install Poetry if you haven't already:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Clone the repository:
```bash
git clone <repository-url>
cd llm-evaluator
```

3. Install dependencies:
```bash
poetry install
```

## Configuration

Create a `.env` file in the project root with the following required variables:

```env
# API Keys
SOLVER_API_KEY=your-solver-api-key
VALIDATOR_API_KEY=your-validator-api-key
HUGGINGFACE_TOKEN=your-huggingface-token

# Model Configuration
SOLVER_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct
VALIDATOR_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct
MODEL_TEMPERATURE=0.7
MODEL_MAX_TOKENS=4096

# Dataset Configuration
DATASET_NAMES=gsm8k,mathematics,mmlu
TEST_LIMIT=100  # Optional: limit number of test samples

# Metrics Configuration
PASS_K_VALUES=1,2,3
USE_WEIGHTED_PASS_K=false
PASS_K_WEIGHTS=1:1.0,2:0.8,3:0.6  # Only if USE_WEIGHTED_PASS_K=true

# Execution Configuration
CONCURRENT_EXECUTION=false
MAX_CONCURRENT_TASKS=3
```

## Supported Datasets

The framework currently supports the following datasets:

- GSM8K (Grade School Math 8K)
- Mathematics (Hendrycks)
- MATH-500
- MMLU (Massive Multitask Language Understanding)
- HellaSwag
- TruthfulQA
- GPQA
- AIME (American Invitational Mathematics Examination)

## Usage

1. Activate the Poetry environment:
```bash
poetry shell
```

2. Run the evaluator:
```bash
python main.py
```

The evaluator will:
1. Load the specified datasets
2. Process questions using the solver model
3. Validate answers using the validator model
4. Calculate metrics and generate reports

## Output

The framework generates several output files:

- `evaluation_results/`: Directory containing all evaluation results
    - `summary_results_[dataset]_[timestamp].json`: Summary metrics for each evaluation
    - `detailed_results_[dataset]_[timestamp].json`: Detailed results including all predictions
    - `evaluation_history.json`: Historical record of all evaluations
- `llm_evaluation.log`: Detailed logging information

## Metrics

The framework calculates several metrics:

- Pass@K: Success rate for top K predictions
- Average Pass@K: Average of all Pass@K metrics
- Weighted Pass@K: Weighted average of Pass@K metrics (if enabled)
- Speed metrics: Average time per question and standard deviation

## Development

This project uses Ruff for linting. To run the linter:

```bash
poetry run ruff check .
```

## License

[Your chosen license]

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Run linting checks (`poetry run ruff check .`)
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## Contact

Your Name - [your.email@example.com](mailto:your.email@example.com)

Project Link: [https://github.com/username/llm-evaluator](https://github.com/username/llm-evaluator)
