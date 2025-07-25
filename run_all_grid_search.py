#!/usr/bin/env python3
"""
Comprehensive grid search runner for all models with updated data.

This script runs hyperparameter grid search for all models:
- LSTM
- SARIMA
- TCN
- Seq2Seq (without attention)
- Seq2Seq with Attention
- Transformer

Usage:
    python run_all_grid_search.py

The script will create a separate grid search script for each model and run them sequentially.
"""

import os
import sys
import subprocess
import time
from datetime import datetime
import shutil

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80)

def create_model_specific_script(model_type, base_script_path):
    """Create a model-specific grid search script"""
    
    # Read the base script
    with open(base_script_path, 'r') as f:
        script_content = f.read()
    
    # Replace the MODEL_TYPE configuration
    script_content = script_content.replace(
        "MODEL_TYPE = 'lstm'  # Options: 'lstm', 'sarima', 'tcn', 'seq2seq', 'seq2seq_attn', 'transformer'",
        f"MODEL_TYPE = '{model_type}'"
    )
    
    # Create model-specific script
    model_script_path = f'grid_search_{model_type}.py'
    with open(model_script_path, 'w') as f:
        f.write(script_content)
    
    return model_script_path

def run_model_grid_search(model_type, script_path):
    """Run grid search for a specific model"""
    print_header(f"RUNNING GRID SEARCH FOR {model_type.upper()}")
    print(f"Script: {script_path}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    
    try:
        # Run the grid search script
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, text=True, check=True)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"✓ {model_type.upper()} grid search completed successfully!")
        print(f"  Duration: {duration:.2f} seconds ({duration/60:.1f} minutes)")
        
        if result.stdout:
            # Print last few lines of output
            output_lines = result.stdout.strip().split('\n')
            print("Last few lines of output:")
            for line in output_lines[-10:]:
                print(f"  {line}")
        
        return True, duration
        
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"✗ {model_type.upper()} grid search failed!")
        print(f"  Duration: {duration:.2f} seconds")
        print(f"  Error code: {e.returncode}")
        
        if e.stdout:
            print("Standard Output:")
            print(e.stdout)
            
        if e.stderr:
            print("Error Output:")
            print(e.stderr)
        
        return False, duration
    
    except Exception as e:
        print(f"✗ Unexpected error running {model_type}: {str(e)}")
        return False, 0

def analyze_results():
    """Analyze results across all models"""
    print_header("ANALYZING RESULTS ACROSS ALL MODELS")
    
    results_dir = 'results_updated'
    if not os.path.exists(results_dir):
        print("No results directory found!")
        return
    
    model_summary = {}
    
    for model_name in os.listdir(results_dir):
        model_path = os.path.join(results_dir, model_name)
        if not os.path.isdir(model_path):
            continue
        
        print(f"\nAnalyzing {model_name.upper()}...")
        
        # Count configurations
        config_count = 0
        best_rmse = float('inf')
        best_config = None
        
        # Look through all configurations
        for root, dirs, files in os.walk(model_path):
            if 'summary_metrics_test.csv' in files:
                config_count += 1
                try:
                    import pandas as pd
                    metrics_df = pd.read_csv(os.path.join(root, 'summary_metrics_test.csv'), index_col=0)
                    rmse_mean = metrics_df.loc['mean', 'RMSE']
                    
                    if rmse_mean < best_rmse:
                        best_rmse = rmse_mean
                        best_config = os.path.basename(root)
                except:
                    continue
        
        model_summary[model_name] = {
            'configurations': config_count,
            'best_rmse': best_rmse,
            'best_config': best_config
        }
        
        print(f"  Configurations tested: {config_count}")
        print(f"  Best RMSE: {best_rmse:.4f}")
        print(f"  Best configuration: {best_config}")
    
    # Create summary table
    import pandas as pd
    summary_df = pd.DataFrame(model_summary).T
    summary_df = summary_df.round(4)
    summary_df.to_csv(os.path.join(results_dir, 'all_models_summary.csv'))
    
    print("\nOVERALL SUMMARY:")
    print(summary_df.to_string())
    
    return summary_df

def main():
    """Main execution function"""
    print_header("COMPREHENSIVE GRID SEARCH FOR ALL MODELS")
    print("This script will run hyperparameter grid search for all models:")
    print("- LSTM")
    print("- SARIMA") 
    print("- TCN")
    print("- Seq2Seq")
    print("- Seq2Seq with Attention")
    print("- Transformer")
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if data file exists
    data_path = 'data_updated/state_month_overdose_2015_2023.xlsx'
    if not os.path.exists(data_path):
        print(f"\n✗ Data file not found: {data_path}")
        print("Please ensure your data file is in the correct location.")
        return False
    
    # Check if base script exists
    base_script = 'updated_grid_search.py'
    if not os.path.exists(base_script):
        print(f"\n✗ Base grid search script not found: {base_script}")
        print("Please create the base script first.")
        return False
    
    total_start_time = time.time()
    
    # Models to run
    models = ['lstm', 'sarima', 'tcn', 'seq2seq', 'seq2seq_attn', 'transformer']
    
    results = {}
    
    # Run grid search for each model
    for model in models:
        try:
            # Create model-specific script
            model_script = create_model_specific_script(model, base_script)
            
            # Run grid search
            success, duration = run_model_grid_search(model, model_script)
            
            results[model] = {
                'success': success,
                'duration': duration
            }
            
            # Clean up model-specific script
            if os.path.exists(model_script):
                os.remove(model_script)
                
        except Exception as e:
            print(f"Error processing {model}: {e}")
            results[model] = {
                'success': False,
                'duration': 0
            }
            continue
    
    # Total execution time
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    # Print summary
    print_header("GRID SEARCH EXECUTION SUMMARY")
    print(f"Total execution time: {total_duration:.2f} seconds ({total_duration/3600:.1f} hours)")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\nModel Execution Results:")
    for model, result in results.items():
        status = "✓ SUCCESS" if result['success'] else "✗ FAILED"
        duration_str = f"{result['duration']:.1f}s ({result['duration']/60:.1f}m)"
        print(f"  {model.upper():15} {status:10} Duration: {duration_str}")
    
    # Analyze results if any succeeded
    successful_models = [model for model, result in results.items() if result['success']]
    
    if successful_models:
        print(f"\nSuccessful models: {len(successful_models)}/{len(models)}")
        
        try:
            summary_df = analyze_results()
            
            print_header("NEXT STEPS")
            print("1. Review the results in 'results_updated/' directory")
            print("2. Check 'results_updated/all_models_summary.csv' for best configurations")
            print("3. Update your final evaluation script with the optimal hyperparameters")
            print("4. Run final evaluation with the best hyperparameters")
            
        except Exception as e:
            print(f"Error analyzing results: {e}")
    else:
        print("\n✗ No models completed successfully. Please check the errors above.")
    
    return len(successful_models) > 0

if __name__ == "__main__":
    success = main()
    if success:
        sys.exit(0)
    else:
        sys.exit(1)
