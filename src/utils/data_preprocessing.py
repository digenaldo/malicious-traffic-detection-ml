import pandas as pd
import numpy as np
import logging
from pathlib import Path

def load_data(data_dir):
    """
    Load training and testing data from CSV files.
    
    Args:
        data_dir: Path to the directory containing data files
    
    Returns:
        tuple: (train_data, test_data) DataFrames
    """
    data_dir = Path(data_dir)
    train_path = data_dir / "train_mosaic.csv.zip"
    test_path = data_dir / "test_mosaic.csv.zip"
    
    if not train_path.exists():
        raise FileNotFoundError(f"Training data file not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Testing data file not found: {test_path}")
    
    logging.info("Loading training and testing data...")
    try:
        train_data = pd.read_csv(train_path).sample(n=3000, random_state=42)
        test_data = pd.read_csv(test_path).sample(n=600, random_state=42)
        logging.info(f"Data loaded: Train shape {train_data.shape}, Test shape {test_data.shape}")
        return train_data, test_data
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        raise

def check_and_replace_invalid_values(df, df_name):
    """
    Check and replace invalid values (inf, NaN) in the dataframe.
    
    Args:
        df: DataFrame to check
        df_name: Name of the dataframe for logging purposes
    """
    logging.info(f"Checking for invalid values in {df_name}...")
    total_inf = 0
    total_nan = 0
    
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            inf_count = np.isinf(df[col].values).sum()
            nan_count = np.isnan(df[col].values).sum()
            
            if inf_count > 0:
                logging.warning(
                    f"{df_name} column '{col}' contains {inf_count} infinite values. "
                    "Replacing with large finite numbers."
                )
                df[col].replace([np.inf, -np.inf], np.finfo(np.float64).max, inplace=True)
                total_inf += inf_count
                
            if nan_count > 0:
                logging.warning(
                    f"{df_name} column '{col}' contains {nan_count} NaN values. "
                    "Replacing with zero."
                )
                df[col].fillna(0, inplace=True)
                total_nan += nan_count
    
    if total_inf == 0 and total_nan == 0:
        logging.info(f"No invalid values found in {df_name}.")
    else:
        logging.info(
            f"Completed checking {df_name}. "
            f"Replaced {total_inf} infinite values and {total_nan} NaN values."
        )
