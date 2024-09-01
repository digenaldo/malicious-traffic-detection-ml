import argparse
import logging
from src.main import run_pipeline
from src.config import MODELS, FEATURE_ENGINEERING

def parse_args():
    parser = argparse.ArgumentParser(description="Run Machine Learning Pipeline")
    parser.add_argument("--algorithm", choices=MODELS.keys(), required=True, help="Choose the algorithm to run.")
    parser.add_argument("--feature-engineering", choices=FEATURE_ENGINEERING.keys(), default='None', help="Choose the feature engineering technique.")
    parser.add_argument("--tune-hyperparameters", action="store_true", help="Enable hyperparameter tuning.")
    parser.add_argument("--cross-validation", action="store_true", help="Enable cross-validation.")
    return parser.parse_args()

def main_cli():
    args = parse_args()
    run_pipeline(args.algorithm, args.feature_engineering, args.tune_hyperparameters, args.cross_validation)

if __name__ == "__main__":
    main_cli()
