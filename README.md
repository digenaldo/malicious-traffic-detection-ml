
# Malicious Traffic Detection ML

A machine learning-based system for detecting malicious network traffic, developed as part of a master's research project.

- [Paper](https://sol.sbc.org.br/index.php/wgrs/article/view/35631/35418)

## Project Overview

This project implements a modularized pipeline for detecting malicious network traffic using various machine learning algorithms. It provides flexibility in experimentation with different feature engineering techniques, hyperparameter tuning, and cross-validation. The project is designed to be easily extendable and adaptable to different research and practical needs.

## Features

- Supports multiple machine learning models:
  - Naive Bayes
  - Decision Tree
  - Logistic Regression
  - Random Forest
- Offers several feature engineering techniques:
  - Standard Scaling
  - Min-Max Scaling
  - PCA
  - Polynomial Features
  - SelectKBest
- Includes hyperparameter tuning capabilities and cross-validation for improving model performance.
- Generates detailed evaluation reports and model comparison summaries.

## Project Structure

The project is organized into the following directory structure:

```bash
project/
│
├── data/
│   ├── train_mosaic.csv.zip         # Training dataset
│   └── test_mosaic.csv.zip          # Testing dataset
│
├── results/                         # Directory for storing model outputs and evaluation reports
│
├── src/
│   ├── __init__.py                  # Indicates that src is a module
│   ├── main.py                      # Orchestrates the entire pipeline
│   ├── config.py                    # Configuration file containing paths and model settings
│   ├── cli.py                       # Command-line interface for running the pipeline
│   └── utils/                       # Directory for utility scripts
│       ├── __init__.py              # Indicates that utils is a module
│       ├── data_preprocessing.py    # Functions for loading and preprocessing data
│       ├── feature_engineering.py   # Functions for applying feature engineering techniques
│       ├── model_training.py        # Functions for model training and hyperparameter tuning
│       └── evaluation.py            # Functions for evaluating models and generating reports
│
├── Dockerfile                       # Docker image definition
├── docker-compose.yml               # Docker Compose configuration for Podman/Docker
├── .dockerignore                    # Files to exclude from Docker build
├── requirements.txt                 # Python dependencies
├── generate_synthetic_data.py       # Script to generate synthetic test data
└── README.md                        # Project overview and instructions
```

### Explanation of Key Components

- **`data/`**: This directory contains the datasets used for training and testing the models. The files are expected to be in CSV format, compressed into ZIP files.
  
- **`results/`**: All model outputs, including trained models and evaluation reports, are stored in this directory. The results are organized based on the feature engineering techniques and algorithms used.

- **`src/config.py`**: This configuration file stores global settings, including paths to data and results directories, and the definitions of available models and feature engineering techniques.

- **`src/utils/data_preprocessing.py`**: Contains functions for loading the datasets and preprocessing them, including handling missing or invalid values.

- **`src/utils/feature_engineering.py`**: Defines the feature engineering techniques that can be applied to the data before training the models.

- **`src/utils/model_training.py`**: Includes functions for setting up the machine learning pipelines, performing hyperparameter tuning, and training the models.

- **`src/utils/evaluation.py`**: Handles the evaluation of trained models, including calculating accuracy, generating classification reports, and saving these results to the results directory.

- **`src/main.py`**: This is the core script that ties together all components, running the entire machine learning pipeline based on the specified parameters.

- **`src/cli.py`**: Provides a command-line interface to interact with the pipeline, allowing users to specify which algorithm, feature engineering technique, and other options to use during execution.

## Installation and Setup

### Option 1: Using Docker/Podman (Recommended)

This project includes Docker and Podman support for easy deployment and consistent environments.

#### Prerequisites
- Docker or Podman installed on your system
- For Podman, you may need `podman-compose` (install with: `pip install podman-compose`)

#### Building the Image

Using Podman:
```bash
podman build -t malicious-traffic-detection-ml:latest .
```

Using Docker:
```bash
docker build -t malicious-traffic-detection-ml:latest .
```

#### Running with Podman Compose

**Note:** For Podman, you may need to install `podman-compose`:
```bash
pip install podman-compose
```

Then build and run:
```bash
# Build and run
podman-compose build
podman-compose run ml-pipeline python -m src.cli --algorithm "Random Forest" --feature-engineering "PCA" --tune-hyperparameters --cross-validation
```

Alternatively, you can use `podman compose` (without the hyphen) if you have Podman 4.0+:
```bash
podman compose build
podman compose run ml-pipeline python -m src.cli --algorithm "Random Forest" --feature-engineering "PCA"
```

#### Running with Docker Compose

```bash
# Build and run
docker-compose build
docker-compose run ml-pipeline python -m src.cli --algorithm "Random Forest" --feature-engineering "PCA" --tune-hyperparameters --cross-validation
```

#### Direct Container Execution

Using Podman:
```bash
podman run --rm -v $(pwd)/data:/app/data:ro -v $(pwd)/results:/app/results malicious-traffic-detection-ml:latest python -m src.cli --algorithm "Random Forest" --feature-engineering "PCA"
```

Using Docker:
```bash
docker run --rm -v $(pwd)/data:/app/data:ro -v $(pwd)/results:/app/results malicious-traffic-detection-ml:latest python -m src.cli --algorithm "Random Forest" --feature-engineering "PCA"
```

### Option 2: Local Python Installation

To get started, clone the repository and navigate into the project directory:

```bash
git clone https://github.com/yourusername/malicious-traffic-detection-ml.git
cd malicious-traffic-detection-ml
```

Make sure you have Python 3.11 or higher installed. Install the required Python packages:

```bash
pip install -r requirements.txt
```

**Note:** Ensure you have the data files (`train_mosaic.csv.zip` and `test_mosaic.csv.zip`) in the `data/` directory before running the pipeline.

### Generating Synthetic Data for Testing

If you don't have real data files, you can generate synthetic network traffic data for testing:

```bash
python generate_synthetic_data.py
```

This script will create:
- `data/train_mosaic.csv.zip` - Training dataset (5000 samples)
- `data/test_mosaic.csv.zip` - Testing dataset (1000 samples)

The synthetic data includes features derived from network flow characteristics (packet counts, timing, flags, etc.) and labels (0 for benign, 1 for malicious traffic). The data is designed to test the ML pipeline functionality.

## Running the Pipeline

To run the pipeline using the command-line interface, use the following command:

```bash
python -m src.cli --algorithm "Random Forest" --feature-engineering "PCA" --tune-hyperparameters --cross-validation
```

### CLI Options

- `--algorithm`: Specifies the machine learning algorithm to use. Options include:
  - "Naive Bayes"
  - "Decision Tree"
  - "Logistic Regression"
  - "Random Forest"
  
- `--feature-engineering`: Specifies the feature engineering technique to apply. Options include:
  - "None"
  - "Standard Scaling"
  - "Min-Max Scaling"
  - "PCA"
  - "Polynomial Features"
  - "SelectKBest"
  
- `--tune-hyperparameters`: Enables hyperparameter tuning using RandomizedSearchCV.
  
- `--cross-validation`: Enables cross-validation to evaluate the model's performance.

## Example Command

Here's an example command to run a pipeline with Random Forest, PCA for feature engineering, hyperparameter tuning enabled, and cross-validation:

```bash
python -m src.cli --algorithm "Random Forest" --feature-engineering "PCA" --tune-hyperparameters --cross-validation
```

This command will execute the entire pipeline, including data preprocessing, feature engineering, model training, hyperparameter tuning, and evaluation. Results, including trained models and evaluation reports, will be saved in the \`results/\` directory.

## Contributing

If you'd like to contribute to this project, please fork the repository and submit a pull request with your changes. Contributions are always welcome!
