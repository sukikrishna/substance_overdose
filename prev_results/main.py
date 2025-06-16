# ### main.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # disable GPU entirely

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# from pipeline.run_trials import run_trials
# from pipeline.aggregate_metrics import aggregate_all_metrics
# from pipeline.plot_results import plot_model_vs_sarima
# from models.lstm_model import LSTMModel
# from models.sarima_model import SARIMAModel

# # Configuration
# DATA_PATH = "data/state_month_overdose.xlsx"
# OUTPUT_DIR = "results"
# LOOKBACKS = [3, 5, 7]
# BATCH_SIZES = [8]
# EPOCHS = 100
# SEEDS = [42, 123, 777]

# # Run trials for each model
# # run_trials("lstm", LSTMModel, DATA_PATH, OUTPUT_DIR, LOOKBACKS, BATCH_SIZES, EPOCHS, SEEDS)
# run_trials("sarima", SARIMAModel, DATA_PATH, OUTPUT_DIR, LOOKBACKS, BATCH_SIZES, EPOCHS, SEEDS)

# # Aggregate metrics
# aggregate_all_metrics(OUTPUT_DIR)

# # Plot results
# plot_model_vs_sarima("lstm", OUTPUT_DIR)

# ### main.py
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # disable GPU entirely

# import warnings
# warnings.filterwarnings("ignore", category=UserWarning)

# from pipeline.run_trials import run_trials
# from pipeline.aggregate_metrics import aggregate_all_metrics
# from pipeline.plot_results import plot_model_vs_sarima
# from models.lstm_model import LSTMModel
# from models.sarima_model import SARIMAModel

# # Configuration
# DATA_PATH = "data/state_month_overdose.xlsx"
# OUTPUT_DIR = "results"
# LOOKBACKS = [3, 5, 7]
# BATCH_SIZES = [8]
# EPOCHS = 100
# SEEDS = [42, 123, 777]

# # Run trials for each model
# # run_trials("lstm", LSTMModel, DATA_PATH, OUTPUT_DIR, LOOKBACKS, BATCH_SIZES, EPOCHS, SEEDS)
# run_trials("sarima", SARIMAModel, DATA_PATH, OUTPUT_DIR, LOOKBACKS, BATCH_SIZES, EPOCHS, SEEDS)

# # Aggregate metrics
# aggregate_all_metrics(OUTPUT_DIR)

# # Plot results
# plot_model_vs_sarima("lstm", OUTPUT_DIR)

"""Main script to run the time series forecasting pipeline."""

import os
import argparse
import numpy as np
import warnings

# Disable TensorFlow GPU warnings and info messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# Optionally disable GPU if needed
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Suppress common warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from models.lstm_model import LSTMModel
from models.sarima_model import SARIMAModel
from pipeline.run_trials import run_trials
from pipeline.aggregate_metrics import aggregate_all_metrics
from pipeline.plot_results import plot_model_vs_sarima, plot_metric_comparison, plot_parameter_effects

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run time series forecasting pipeline')
    
    parser.add_argument('--data_path', type=str, default='data/state_month_overdose.xlsx',
                       help='Path to overdose data Excel file')
    
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Directory to save results')
    
    # [3, 5, 7, 9, 12]
    parser.add_argument('--lookbacks', type=int, nargs='+', default=[3],
                       help='Lookback periods to test')
    # [1, 8]
    parser.add_argument('--batch_sizes', type=int, nargs='+', default=[8],
                       help='Batch sizes to test')
    
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    # [42, 123, 777]
    parser.add_argument('--seeds', type=int, nargs='+', default=[42],
                       help='Random seeds for reproducibility')
    
    parser.add_argument('--train_end', type=str, default='2020-01-01',
                       help='End date for training data (format: YYYY-MM-DD)')
    
    parser.add_argument('--run_lstm', action='store_true',
                       help='Run LSTM model')
    
    parser.add_argument('--run_sarima', action='store_true',
                       help='Run SARIMA model')
    
    parser.add_argument('--aggregate_only', action='store_true',
                       help='Only aggregate metrics without running trials')
    
    parser.add_argument('--plot_only', action='store_true',
                       help='Only generate plots without running trials')
    
    return parser.parse_args()

def main():
    """Run the main pipeline."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if not args.aggregate_only and not args.plot_only:
        print("Starting model trials...")
        
        # Run LSTM model if requested
        if args.run_lstm:
            print("\n=== Running LSTM model trials ===")
            run_trials(
                "lstm", 
                LSTMModel, 
                args.data_path, 
                args.output_dir, 
                args.lookbacks, 
                args.batch_sizes, 
                args.epochs, 
                args.seeds,
                args.train_end
            )
            
        # Run SARIMA model if requested
        if args.run_sarima:
            print("\n=== Running SARIMA model trials ===")
            run_trials(
                "sarima", 
                SARIMAModel, 
                args.data_path, 
                args.output_dir, 
                args.lookbacks, 
                args.batch_sizes, 
                args.epochs, 
                args.seeds,
                args.train_end
            )
    
    # Aggregate metrics
    print("\n=== Aggregating metrics ===")
    metrics_df = aggregate_all_metrics(args.output_dir)
    
    # Generate plots
    print("\n=== Generating plots ===")
    
    # Compare LSTM with SARIMA if both were run
    lstm_dir = os.path.join(args.output_dir, "lstm")
    sarima_dir = os.path.join(args.output_dir, "sarima")
    
    if os.path.exists(lstm_dir) and os.path.exists(sarima_dir):
        plot_model_vs_sarima("lstm", args.output_dir, args.train_end)
    
    # Generate metric comparison plots
    plot_metric_comparison(args.output_dir)
    
    # Generate parameter effect plots
    plot_parameter_effects(args.output_dir)
    
    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main()