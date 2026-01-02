import argparse
import logging
import sys
from pathlib import Path
from src.main import run_pipeline
from src.config import MODELS, FEATURE_ENGINEERING

# Setup logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def parse_args():
    """Parse command line arguments for the ML pipeline."""
    parser = argparse.ArgumentParser(
        description="Run Machine Learning Pipeline for Malicious Traffic Detection"
    )
    parser.add_argument(
        "--algorithm",
        choices=MODELS.keys(),
        required=True,
        help="Choose the machine learning algorithm to use."
    )
    parser.add_argument(
        "--feature-engineering",
        choices=FEATURE_ENGINEERING.keys(),
        default='None',
        help="Choose the feature engineering technique to apply."
    )
    parser.add_argument(
        "--tune-hyperparameters",
        action="store_true",
        help="Enable hyperparameter tuning using RandomizedSearchCV."
    )
    parser.add_argument(
        "--cross-validation",
        action="store_true",
        help="Enable cross-validation for model evaluation."
    )
    return parser.parse_args()

def main_cli():
    """Main entry point for the command-line interface."""
    try:
        args = parse_args()
        logging.info("Starting ML pipeline execution...")
        logging.info(f"Algorithm: {args.algorithm}")
        logging.info(f"Feature Engineering: {args.feature_engineering}")
        logging.info(f"Hyperparameter Tuning: {args.tune_hyperparameters}")
        logging.info(f"Cross-Validation: {args.cross_validation}")
        
        run_pipeline(
            args.algorithm,
            args.feature_engineering,
            args.tune_hyperparameters,
            args.cross_validation
        )
        logging.info("Pipeline execution completed successfully.")
    except Exception as e:
        logging.error(f"Error during pipeline execution: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main_cli()
