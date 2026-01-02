#!/bin/bash
# Example usage script for the Malicious Traffic Detection ML pipeline
# This script demonstrates how to run the pipeline with different configurations

echo "=== Malicious Traffic Detection ML - Example Usage ==="
echo ""

# Example 1: Basic usage with Random Forest and PCA
echo "Example 1: Random Forest with PCA feature engineering"
echo "Command: python -m src.cli --algorithm \"Random Forest\" --feature-engineering \"PCA\""
echo ""

# Example 2: With hyperparameter tuning
echo "Example 2: Logistic Regression with Standard Scaling and hyperparameter tuning"
echo "Command: python -m src.cli --algorithm \"Logistic Regression\" --feature-engineering \"Standard Scaling\" --tune-hyperparameters"
echo ""

# Example 3: Full pipeline with all options
echo "Example 3: Decision Tree with Min-Max Scaling, hyperparameter tuning, and cross-validation"
echo "Command: python -m src.cli --algorithm \"Decision Tree\" --feature-engineering \"Min-Max Scaling\" --tune-hyperparameters --cross-validation"
echo ""

# Example 4: Using Docker/Podman
echo "Example 4: Running with Podman Compose"
echo "Command: podman-compose run ml-pipeline python -m src.cli --algorithm \"Random Forest\" --feature-engineering \"PCA\" --tune-hyperparameters --cross-validation"
echo ""

echo "Note: Make sure you have the data files in the data/ directory before running!"
echo "Required files: train_mosaic.csv.zip and test_mosaic.csv.zip"

