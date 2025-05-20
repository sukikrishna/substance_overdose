"""Module for aggregating metrics across multiple trials."""

import os
import numpy as np
import pandas as pd
from utils import calculate_metrics, compute_overlap_with_ground_truth

def aggregate_all_metrics(output_dir):
    """
    Aggregate metrics across all models, parameter combinations, and trials.
    
    Parameters
    ----------
    output_dir : str
        Directory containing model outputs
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing aggregated metrics
    """
    rows = []
    
    # Get all model directories (excluding comparisons directory)
    model_dirs = [d for d in os.listdir(output_dir) 
                 if os.path.isdir(os.path.join(output_dir, d)) and d != 'comparisons']
    
    for model in model_dirs:
        model_dir = os.path.join(output_dir, model)
        
        # Get all parameter combinations
        combo_dirs = [d for d in os.listdir(model_dir) 
                     if os.path.isdir(os.path.join(model_dir, d))]
        
        for combo in combo_dirs:
            combo_dir = os.path.join(model_dir, combo)
            
            # Extract parameters from combo name
            params = combo.split('_')
            lookback = int(params[0].replace('lookback', ''))
            batch = int(params[1].replace('batch', ''))
            seed = int(params[2].replace('seed', ''))
            
            # Metrics lists to hold results across trials
            rmses, mapes, maes = [], [], []
            pi_widths, pi_overlaps = [], []
            
            # Get all trial directories
            trial_dirs = [d for d in os.listdir(combo_dir) 
                         if os.path.isdir(os.path.join(combo_dir, d))]
            
            for trial in trial_dirs:
                trial_dir = os.path.join(combo_dir, trial)
                
                # Load predictions
                pred_file = os.path.join(trial_dir, "predictions.csv")
                if not os.path.exists(pred_file):
                    continue
                    
                df = pd.read_csv(pred_file)
                
                # Calculate metrics
                actual = df['Deaths'].values
                preds = df['Predictions'].values
                lower_pi = df['Lower PI'].values
                upper_pi = df['Upper PI'].values
                
                # Calculate performance metrics
                metrics = calculate_metrics(actual, preds)
                rmses.append(metrics['RMSE'])
                mapes.append(metrics['MAPE'])
                maes.append(metrics['MAE'])
                
                # Calculate PI width and overlap
                pi_width = np.mean(upper_pi - lower_pi)
                pi_widths.append(pi_width)
                
                overlap = compute_overlap_with_ground_truth(actual, lower_pi, upper_pi)
                pi_overlaps.append(overlap)
            
            # If trials were processed, add results to rows
            if rmses:
                rows.append({
                    'Model': model,
                    'Lookback': lookback,
                    'Batch': batch,
                    'Seed': seed,
                    'RMSE Mean': np.mean(rmses),
                    'RMSE Std': np.std(rmses),
                    'MAPE Mean': np.mean(mapes),
                    'MAPE Std': np.std(mapes),
                    'MAE Mean': np.mean(maes),
                    'MAE Std': np.std(maes),
                    'PI Width Mean': np.mean(pi_widths),
                    'PI Width Std': np.std(pi_widths),
                    'PI GT Overlap % Mean': np.mean(pi_overlaps),
                    'PI GT Overlap % Std': np.std(pi_overlaps)
                })
    
    # Create DataFrame and save to CSV
    results_df = pd.DataFrame(rows)
    output_file = os.path.join(output_dir, 'aggregated_metrics.csv')
    results_df.to_csv(output_file, index=False)
    
    print(f"Aggregated metrics saved to {output_file}")
    
    return results_df