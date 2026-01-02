"""
Script to generate synthetic network traffic data for testing the ML pipeline.

This script creates CSV files with features derived from network flow characteristics
and saves them as compressed ZIP files compatible with the pipeline.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import zipfile
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_flow_features(n_samples, malicious_ratio=0.3, random_state=42):
    """
    Generate synthetic network flow features based on the flow structure.
    
    Args:
        n_samples: Number of samples to generate
        malicious_ratio: Ratio of malicious samples (0.0 to 1.0)
        random_state: Random seed for reproducibility
    
    Returns:
        DataFrame with features and labels
    """
    np.random.seed(random_state)
    
    n_malicious = int(n_samples * malicious_ratio)
    n_benign = n_samples - n_malicious
    
    data = []
    
    # Generate benign traffic (normal patterns)
    logging.info(f"Generating {n_benign} benign samples...")
    for i in range(n_benign):
        features = generate_benign_flow_features()
        features['label'] = 0  # Benign
        data.append(features)
    
    # Generate malicious traffic (anomalous patterns)
    logging.info(f"Generating {n_malicious} malicious samples...")
    for i in range(n_malicious):
        features = generate_malicious_flow_features()
        features['label'] = 1  # Malicious
        data.append(features)
    
    # Convert to DataFrame and shuffle
    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    return df

def generate_benign_flow_features():
    """Generate features for benign (normal) network traffic."""
    # Normal traffic patterns
    total_fwd_packets = np.random.randint(5, 100)
    total_bwd_packets = np.random.randint(5, 100)
    
    return {
        'start_time': np.random.uniform(0, 1000),
        'end_time': np.random.uniform(1000, 2000),
        'total_fwd_packets': total_fwd_packets,
        'total_bwd_packets': total_bwd_packets,
        'total_length_of_fwd_packets': np.random.randint(100, 50000),
        'total_length_of_bwd_packets': np.random.randint(100, 50000),
        'fwd_packet_length_mean': np.random.uniform(40, 1500),
        'bwd_packet_length_mean': np.random.uniform(40, 1500),
        'fwd_iat_mean': np.random.uniform(0.001, 1.0),
        'bwd_iat_mean': np.random.uniform(0.001, 1.0),
        'syn_flag_count': np.random.randint(1, 5),
        'fin_flag_count': np.random.randint(0, 3),
        'rst_flag_count': np.random.randint(0, 2),
        'psh_flag_count': np.random.randint(0, 10),
        'ack_flag_count': np.random.randint(5, 50),
        'urg_flag_count': 0,
        'ece_flag_count': np.random.randint(0, 2),
        'cwe_flag_count': 0,
        'fwd_psh_flags': np.random.randint(0, 5),
        'bwd_psh_flags': np.random.randint(0, 5),
        'fwd_urg_flags': 0,
        'bwd_urg_flags': 0,
        'fwd_header_length': np.random.randint(20, 200),
        'bwd_header_length': np.random.randint(20, 200),
        'fwd_packets_sec': np.random.uniform(1.0, 100.0),
        'bwd_packets_sec': np.random.uniform(1.0, 100.0),
        'subflow_fwd_bytes': np.random.randint(100, 10000),
        'subflow_bwd_bytes': np.random.randint(100, 10000),
        'init_win_bytes_forward': np.random.randint(1000, 65535),
        'init_win_bytes_backward': np.random.randint(1000, 65535),
        'act_data_pkt_fwd': np.random.randint(5, 50),
        'act_data_pkt_bwd': np.random.randint(5, 50),
        'min_seg_size_forward': np.random.randint(20, 1460),
        'down_up_ratio': np.random.uniform(0.1, 10.0),
        'average_packet_size': np.random.uniform(50, 1500),
        'flow_duration': np.random.uniform(0.1, 100.0),
        'flow_bytes_per_sec': np.random.uniform(100, 100000),
        'flow_packets_per_sec': np.random.uniform(1, 1000),
    }

def generate_malicious_flow_features():
    """Generate features for malicious (anomalous) network traffic."""
    # Malicious traffic patterns (often have unusual characteristics)
    total_fwd_packets = np.random.randint(1, 20)  # Often fewer packets
    total_bwd_packets = np.random.randint(0, 5)   # Often asymmetric
    
    return {
        'start_time': np.random.uniform(0, 1000),
        'end_time': np.random.uniform(1000, 2000),
        'total_fwd_packets': total_fwd_packets,
        'total_bwd_packets': total_bwd_packets,
        'total_length_of_fwd_packets': np.random.randint(50, 10000),  # Often smaller
        'total_length_of_bwd_packets': np.random.randint(0, 1000),     # Often minimal
        'fwd_packet_length_mean': np.random.uniform(20, 200),          # Often smaller packets
        'bwd_packet_length_mean': np.random.uniform(0, 100),
        'fwd_iat_mean': np.random.uniform(0.0001, 0.1),               # Often faster
        'bwd_iat_mean': np.random.uniform(0.001, 10.0),                # Often irregular
        'syn_flag_count': np.random.randint(0, 20),                    # Often many SYN (scans)
        'fin_flag_count': np.random.randint(0, 2),
        'rst_flag_count': np.random.randint(0, 10),                    # Often many RST
        'psh_flag_count': np.random.randint(0, 5),
        'ack_flag_count': np.random.randint(0, 20),                    # Often fewer ACK
        'urg_flag_count': np.random.randint(0, 5),                      # Sometimes URG flags
        'ece_flag_count': np.random.randint(0, 3),
        'cwe_flag_count': np.random.randint(0, 2),
        'fwd_psh_flags': np.random.randint(0, 3),
        'bwd_psh_flags': np.random.randint(0, 2),
        'fwd_urg_flags': np.random.randint(0, 3),                      # Sometimes URG
        'bwd_urg_flags': np.random.randint(0, 2),
        'fwd_header_length': np.random.randint(20, 100),              # Often smaller headers
        'bwd_header_length': np.random.randint(0, 50),
        'fwd_packets_sec': np.random.uniform(0.1, 1000.0),             # Often very fast or very slow
        'bwd_packets_sec': np.random.uniform(0.0, 10.0),                # Often very slow
        'subflow_fwd_bytes': np.random.randint(50, 5000),
        'subflow_bwd_bytes': np.random.randint(0, 500),
        'init_win_bytes_forward': np.random.randint(100, 10000),       # Often smaller windows
        'init_win_bytes_backward': np.random.randint(0, 5000),
        'act_data_pkt_fwd': np.random.randint(0, 20),                  # Often fewer data packets
        'act_data_pkt_bwd': np.random.randint(0, 5),
        'min_seg_size_forward': np.random.randint(0, 100),
        'down_up_ratio': np.random.uniform(0.01, 100.0),                # Often very asymmetric
        'average_packet_size': np.random.uniform(20, 500),              # Often smaller
        'flow_duration': np.random.uniform(0.01, 10.0),                # Often shorter
        'flow_bytes_per_sec': np.random.uniform(10, 50000),            # Often irregular
        'flow_packets_per_sec': np.random.uniform(0.1, 500),           # Often irregular
    }

def save_compressed_csv(df, filepath):
    """
    Save DataFrame as compressed CSV file.
    
    Args:
        df: DataFrame to save
        filepath: Path to save the file (should end with .zip)
    """
    filepath = Path(filepath)
    # Remove .zip extension to get base name, then add .csv
    base_name = filepath.stem if filepath.suffix == '.zip' else filepath.name
    csv_path = filepath.parent / f"{base_name}.csv"
    
    # Save as CSV first
    df.to_csv(csv_path, index=False)
    logging.info(f"Saved CSV to {csv_path} ({len(df)} rows, {len(df.columns)} columns)")
    
    # Compress to ZIP
    with zipfile.ZipFile(filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(csv_path, csv_path.name)
    
    # Remove temporary CSV
    csv_path.unlink()
    logging.info(f"Compressed to {filepath}")

def main():
    """Generate synthetic training and testing datasets."""
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    # Generate training data (more samples)
    logging.info("=" * 60)
    logging.info("Generating training dataset...")
    logging.info("=" * 60)
    train_df = generate_flow_features(n_samples=5000, malicious_ratio=0.3, random_state=42)
    train_path = data_dir / 'train_mosaic.csv.zip'
    save_compressed_csv(train_df, train_path)
    
    # Generate testing data (fewer samples)
    logging.info("=" * 60)
    logging.info("Generating testing dataset...")
    logging.info("=" * 60)
    test_df = generate_flow_features(n_samples=1000, malicious_ratio=0.3, random_state=123)
    test_path = data_dir / 'test_mosaic.csv.zip'
    save_compressed_csv(test_df, test_path)
    
    # Print summary
    logging.info("=" * 60)
    logging.info("Data generation complete!")
    logging.info("=" * 60)
    logging.info(f"Training set: {len(train_df)} samples")
    logging.info(f"  - Benign: {len(train_df[train_df['label'] == 0])} samples")
    logging.info(f"  - Malicious: {len(train_df[train_df['label'] == 1])} samples")
    logging.info(f"Testing set: {len(test_df)} samples")
    logging.info(f"  - Benign: {len(test_df[test_df['label'] == 0])} samples")
    logging.info(f"  - Malicious: {len(test_df[test_df['label'] == 1])} samples")
    logging.info(f"\nFiles created:")
    logging.info(f"  - {train_path}")
    logging.info(f"  - {test_path}")

if __name__ == '__main__':
    main()

