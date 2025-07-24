#!/usr/bin/env python3
"""
Main execution script for comprehensive model evaluation.

This script runs the complete experimental pipeline with proper logging and error handling.
It's designed to be run from the command line and provides detailed progress updates.

Usage:
    python run_comprehensive_evaluation.py [--quick-test]

Arguments:
    --quick-test: Run with reduced parameters for quick testing (fewer seeds/trials)
"""

import sys
import os
import argparse
import time
import logging
from datetime import datetime
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from comprehensive_evaluation_pipeline import ComprehensiveEvaluationPipeline, RESULTS_DIR, DATA_PATH, OPTIMAL_PARAMS
except ImportError as e:
    print(f"Error importing pipeline: {e}")
    print("Make sure comprehensive_evaluation_pipeline.py is in the same directory")
    sys.exit(1)

def setup_logging(results_dir: str):
    """Setup logging configuration"""
    log_file = os.path.join(results_dir, f'evaluation_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

def check_requirements():
    """Check if all required packages are available"""
    required_packages = [
        'numpy', 'pandas', 'matplotlib', 'seaborn', 'sklearn', 
        'tensorflow', 'statsmodels', 'openpyxl'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nInstall missing packages with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_data_file():
    """Check if data file exists and is readable"""
    if not os.path.exists(DATA_PATH):
        print(f"Data file not found: {DATA_PATH}")
        print("Please ensure the Excel file is in the correct location")
        return False
    
    try:
        import pandas as pd
        df = pd.read_excel(DATA_PATH)
        print(f"Data file loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        return True
    except Exception as e:
        print(f"Error reading data file: {e}")
        return False

def print_system_info():
    """Print system information for debugging"""
    import platform
    import tensorflow as tf
    
    print("\n" + "="*80)
    print("SYSTEM INFORMATION")
    print("="*80)
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Data path: {DATA_PATH}")
    print(f"Results directory: {RESULTS_DIR}")

def run_quick_test():
    """Run a quick test with reduced parameters"""
    print("\n" + "="*80)
    print("RUNNING QUICK TEST")
    print("="*80)
    
    # Quick test parameters
    models = ['sarima', 'lstm']  # Only test 2 models
    seeds = [42, 123]  # Only 2 seeds
    trials_per_seed = 5  # Reduced trials
    
    pipeline = ComprehensiveEvaluationPipeline(DATA_PATH, RESULTS_DIR + "_quick_test")
    
    # Load data
    df = pipeline.load_and_preprocess_data()
    data_splits = pipeline.create_train_test_splits(df)
    
    print(f"Quick test setup:")
    print(f"  Models: {models}")
    print(f"  Seeds: {seeds}")
    print(f"  Trials per seed: {trials_per_seed}")
    
    # Run only Experiment 1 for quick test
    start_time = time.time()
    exp1_results = pipeline.experiment_1_excess_mortality(
        data_splits, models, seeds, trials_per_seed)
    
    end_time = time.time()
    
    print(f"\nQuick test completed in {end_time - start_time:.2f} seconds")
    print("Full evaluation can now be run with confidence")

def run_full_evaluation(logger):
    """Run the complete evaluation pipeline"""
    logger.info("Starting comprehensive evaluation pipeline")
    
    start_time = time.time()
    
    # Initialize pipeline
    pipeline = ComprehensiveEvaluationPipeline(DATA_PATH, RESULTS_DIR)
    
    # Load and preprocess data
    logger.info("Loading and preprocessing data...")
    df = pipeline.load_and_preprocess_data()
    
    # Create data splits
    logger.info("Creating train/test splits...")
    data_splits = pipeline.create_train_test_splits(df)
    
    # Define experimental parameters
    models = ['sarima', 'lstm', 'tcn', 'seq2seq_attn', 'transformer']
    seeds = [42, 123, 456, 789, 1000]
    trials_per_seed = 30
    
    logger.info(f"Experimental setup: {len(models)} models, {len(seeds)} seeds, {trials_per_seed} trials/seed")
    
    # Experiment 1: Excess mortality estimation
    logger.info("Starting Experiment 1: Excess mortality estimation...")
    exp1_start = time.time()
    
    try:
        exp1_results = pipeline.experiment_1_excess_mortality(
            data_splits, models, seeds, trials_per_seed)
        exp1_time = time.time() - exp1_start
        logger.info(f"Experiment 1 completed in {exp1_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Experiment 1 failed: {e}")
        raise
    
    # Experiment 2: Variance reduction analysis
    logger.info("Starting Experiment 2: Variance reduction analysis...")
    exp2_start = time.time()
    
    try:
        exp2_results = pipeline.experiment_2_variance_analysis(
            data_splits, models, seeds, trials_per_seed)
        exp2_time = time.time() - exp2_start
        logger.info(f"Experiment 2 completed in {exp2_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Experiment 2 failed: {e}")
        raise
    
    # Experiment 3: Sensitivity analysis
    logger.info("Starting Experiment 3: Sensitivity analysis...")
    exp3_start = time.time()
    
    try:
        exp3_results = pipeline.experiment_3_sensitivity_analysis(data_splits, models)
        exp3_time = time.time() - exp3_start
        logger.info(f"Experiment 3 completed in {exp3_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Experiment 3 failed: {e}")
        raise
    
    total_time = time.time() - start_time
    
    logger.info("="*80)
    logger.info("EVALUATION COMPLETED SUCCESSFULLY!")
    logger.info("="*80)
    logger.info(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
    logger.info(f"Experiment 1 time: {exp1_time:.2f} seconds")
    logger.info(f"Experiment 2 time: {exp2_time:.2f} seconds")
    logger.info(f"Experiment 3 time: {exp3_time:.2f} seconds")
    logger.info(f"Results saved to: {RESULTS_DIR}")
    
    return exp1_results, exp2_results, exp3_results

def generate_final_report(logger):
    """Generate a final summary report"""
    logger.info("Generating final summary report...")
    
    try:
        import pickle
        
        # Load results
        exp1_path = os.path.join(RESULTS_DIR, 'experiment_1_excess_mortality', 'results.pkl')
        exp2_path = os.path.join(RESULTS_DIR, 'experiment_2_variance_analysis', 'results.pkl')
        exp3_path = os.path.join(RESULTS_DIR, 'experiment_3_sensitivity', 'results.pkl')
        
        report_content = []
        report_content.append("COMPREHENSIVE EVALUATION SUMMARY REPORT")
        report_content.append("=" * 60)
        report_content.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_content.append("")
        
        # Check which experiments completed
        experiments_completed = []
        
        if os.path.exists(exp1_path):
            experiments_completed.append("Experiment 1: Excess Mortality Estimation")
            
        if os.path.exists(exp2_path):
            experiments_completed.append("Experiment 2: Variance Reduction Analysis")
            
        if os.path.exists(exp3_path):
            experiments_completed.append("Experiment 3: Sensitivity Analysis")
        
        report_content.append("COMPLETED EXPERIMENTS:")
        for exp in experiments_completed:
            report_content.append(f"  ✓ {exp}")
        report_content.append("")
        
        # Add file inventory
        report_content.append("GENERATED FILES:")
        
        for root, dirs, files in os.walk(RESULTS_DIR):
            level = root.replace(RESULTS_DIR, '').count(os.sep)
            indent = '  ' * level
            report_content.append(f"{indent}{os.path.basename(root)}/")
            
            subindent = '  ' * (level + 1)
            for file in files:
                if not file.startswith('.'):  # Skip hidden files
                    report_content.append(f"{subindent}{file}")
        
        report_content.append("")
        report_content.append("OPTIMAL HYPERPARAMETERS USED:")
        for model, params in OPTIMAL_PARAMS.items():
            report_content.append(f"  {model.upper()}: {params}")
        
        # Save report
        report_path = os.path.join(RESULTS_DIR, 'evaluation_summary_report.txt')
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_content))
        
        logger.info(f"Summary report saved to: {report_path}")
        
        # Print report to console
        print("\n" + "\n".join(report_content))
        
    except Exception as e:
        logger.error(f"Failed to generate summary report: {e}")

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Comprehensive Evaluation Pipeline for Substance Overdose Mortality Forecasting"
    )
    parser.add_argument(
        '--quick-test', 
        action='store_true', 
        help='Run quick test with reduced parameters'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("COMPREHENSIVE EVALUATION PIPELINE")
    print("Advanced Machine Learning for Substance Overdose Mortality Prediction")
    print("=" * 80)
    
    # Print system information
    print_system_info()
    
    # Check requirements
    print("\nChecking requirements...")
    if not check_requirements():
        sys.exit(1)
    print("✓ All required packages available")
    
    # Check data file
    print("\nChecking data file...")
    if not check_data_file():
        sys.exit(1)
    print("✓ Data file accessible")
    
    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    if args.quick_test:
        # Run quick test
        try:
            run_quick_test()
        except Exception as e:
            print(f"Quick test failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        # Setup logging
        logger = setup_logging(RESULTS_DIR)
        
        try:
            # Run full evaluation
            exp1_results, exp2_results, exp3_results = run_full_evaluation(logger)
            
            # Generate final report
            generate_final_report(logger)
            
            print("\n🎉 EVALUATION COMPLETED SUCCESSFULLY! 🎉")
            print(f"\nCheck results in: {RESULTS_DIR}")
            print("\nNext steps:")
            print("1. Review generated plots in the 'figures' folder")
            print("2. Examine CSV files for detailed metrics")
            print("3. Use trained models from 'trained_models' folder for dashboard")
            print("4. Read 'evaluation_summary_report.txt' for overview")
            
        except KeyboardInterrupt:
            logger.warning("Evaluation interrupted by user")
            print("\nEvaluation interrupted. Partial results may be available.")
            sys.exit(1)
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

if __name__ == "__main__":
    main()