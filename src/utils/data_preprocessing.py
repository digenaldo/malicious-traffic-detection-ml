import pandas as pd
import numpy as np
import logging

def load_data(data_dir):
    logging.info("Loading training and testing data...")
    train_data = pd.read_csv(data_dir / "train_mosaic.csv.zip").sample(n=3000, random_state=42)
    test_data = pd.read_csv(data_dir / "test_mosaic.csv.zip").sample(n=600, random_state=42)
    logging.info(f"Data loaded: Train shape {train_data.shape}, Test shape {test_data.shape}")
    return train_data, test_data

def check_and_replace_invalid_values(df, df_name):
    logging.info(f"Checking for invalid values in {df_name}...")
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            inf_count = np.isinf(df[col].values).sum()
            nan_count = np.isnan(df[col].values).sum()
            if inf_count > 0:
                logging.warning(f"{df_name} column '{col}' contains infinite values. Replacing with large finite numbers.")
                df[col].replace([np.inf, -np.inf], np.finfo(np.float64).max, inplace=True)
            if nan_count > 0:
                logging.warning(f"{df_name} column '{col}' contains NaN values. Replacing with zero.")
                df[col].fillna(0, inplace=True)
    logging.info(f"Completed checking {df_name}.")
