import logging

def setup_logger():
    """Configure logging settings"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('llm_evaluation.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)
