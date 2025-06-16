import os
import pandas as pd
import numpy as np
from pathlib import Path
import re

def extract_hyperparams_from_path(path):
    """Extract hyperparameters from the folder path"""
    # Extract lookback, batch size, and epochs from path like 'lookback_12_bs_32_epochs_50'
    pattern = r'lookback_(\d+)_bs_(\d+)_epochs_(\d+)'
    match = re.search(pattern, path)
    if match:
        return {
            'lookback': int(match.group(1)),
            'batch_size': int(match.group(2)),
            'epochs': int(match.group(3))
        }
    return None

def load_summary_metrics(file_path):
    """Load summary metrics from CSV file"""
    try:
        df = pd.read_csv(file_path, index_col=0)
        # Extract mean and std values
        metrics = {}
        for col in df.columns:
            if col in ['RMSE', 'MAE', 'MAPE']:  # Only keep the main metrics
                metrics[f'{col}_mean'] = df.loc['mean', col]
                metrics[f'{col}_std'] = df.loc['std', col]
        return metrics
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def consolidate_single_model(model_name, results_dir='results'):
    """Consolidate hyperparameter results for a single model and save in model folder"""
    model_path = Path(results_dir) / model_name / 'fixed_seed_variability'
    
    if not model_path.exists():
        print(f"Path {model_path} does not exist")
        return
    
    train_results = []
    test_results = []
    
    print(f"Processing {model_name}...")
    
    # Iterate through all hyperparameter combination folders
    for config_folder in model_path.iterdir():
        if config_folder.is_dir():
            hyperparams = extract_hyperparams_from_path(config_folder.name)
            if hyperparams is None:
                continue
            
            # Load train metrics
            train_file = config_folder / 'summary_metrics_train.csv'
            if train_file.exists():
                train_metrics = load_summary_metrics(train_file)
                if train_metrics:
                    result_row = {**hyperparams, **train_metrics}
                    train_results.append(result_row)
            
            # Load test metrics
            test_file = config_folder / 'summary_metrics_test.csv'
            if test_file.exists():
                test_metrics = load_summary_metrics(test_file)
                if test_metrics:
                    result_row = {**hyperparams, **test_metrics}
                    test_results.append(result_row)
    
    # Convert to DataFrames and sort by RMSE_mean (best first)
    if train_results:
        train_df = pd.DataFrame(train_results)
        train_df = train_df.sort_values('RMSE_mean')
        
        # Save in model folder
        train_output_path = Path(results_dir) / model_name / 'hyperparameter_results_train.csv'
        train_df.to_csv(train_output_path, index=False)
        print(f"✓ Saved train results: {train_output_path}")
        print(f"  Best train config: lookback={train_df.iloc[0]['lookback']}, "
              f"bs={train_df.iloc[0]['batch_size']}, epochs={train_df.iloc[0]['epochs']}, "
              f"RMSE={train_df.iloc[0]['RMSE_mean']:.2f}")
    
    if test_results:
        test_df = pd.DataFrame(test_results)
        test_df = test_df.sort_values('RMSE_mean')
        
        # Save in model folder
        test_output_path = Path(results_dir) / model_name / 'hyperparameter_results_test.csv'
        test_df.to_csv(test_output_path, index=False)
        print(f"✓ Saved test results: {test_output_path}")
        print(f"  Best test config: lookback={test_df.iloc[0]['lookback']}, "
              f"bs={test_df.iloc[0]['batch_size']}, epochs={test_df.iloc[0]['epochs']}, "
              f"RMSE={test_df.iloc[0]['RMSE_mean']:.2f}")
    
    print()

def process_all_models(model_names, results_dir='results'):
    """Process all specified models"""
    print("="*60)
    print("CONSOLIDATING HYPERPARAMETER RESULTS")
    print("="*60)
    
    for model_name in model_names:
        consolidate_single_model(model_name, results_dir)
    
    print("="*60)
    print("CONSOLIDATION COMPLETE!")
    print("="*60)
    print("Files created in each model folder:")
    print("- hyperparameter_results_train.csv")
    print("- hyperparameter_results_test.csv")
    print("\nTo find optimal hyperparameters:")
    print("1. Open the CSV file for your model")
    print("2. Sort by RMSE_mean (ascending) for best performance")
    print("3. Or sort by MAE_mean or MAPE_mean as needed")

# Main execution
if __name__ == "__main__":
    
    # MODEL_NAMES = ['lstm', 'sarima', 'tcn', 'seq2seq', 'tcn_updated']
    
    # Process all models
    # process_all_models(MODEL_NAMES)

    # Process single model
    consolidate_single_model('lstm')