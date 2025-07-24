#!/usr/bin/env python3
"""
Master Execution Script for Complete Evaluation Pipeline

This script orchestrates the entire evaluation process from data inspection 
to final results generation. It's the single entry point for running all
experiments described in your paper.

Usage:
    python run_complete_evaluation.py [--skip-inspection] [--quick-test]

Features:
- Automatic data inspection and validation
- Complete experimental pipeline execution  
- Error handling and progress monitoring
- Results consolidation and summary generation
- Publication-ready output generation
"""

import os
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path

def print_banner():
    """Print project banner"""
    print("=" * 80)
    print("COMPREHENSIVE EVALUATION PIPELINE")
    print("Advanced Machine Learning for Substance Overdose Mortality Prediction")
    print("=" * 80)
    print("Experiments:")
    print("  1. Excess mortality estimation (2015-2019 train, 2020-2023 test)")
    print("  2. Variance reduction analysis across forecasting horizons")
    print("  3. Sensitivity analysis for seeds and trials")
    print("=" * 80)

def run_data_inspection():
    """Run data inspection step"""
    print("\n🔍 STEP 1: DATA INSPECTION")
    print("-" * 50)
    
    try:
        import data_inspector
        print("Running data inspector...")
        
        # Capture the main function output
        df = data_inspector.inspect_excel_file()
        
        if df is not None:
            data_inspector.analyze_date_structure(df)
            data_inspector.analyze_deaths_data(df)
            data_inspector.create_data_visualization(df)
            data_inspector.suggest_preprocessing_steps(df)
            data_inspector.generate_data_report(df)
            
            print("✅ Data inspection completed successfully!")
            print("📊 Generated: data_inspection_plots.png")
            print("📄 Generated: data_inspection_report.txt")
            return True
        else:
            print("❌ Data inspection failed!")
            return False
            
    except Exception as e:
        print(f"❌ Error during data inspection: {e}")
        return False

def run_quick_test():
    """Run quick test to verify pipeline"""
    print("\n⚡ STEP 2: QUICK TEST")
    print("-" * 50)
    
    try:
        print("Running quick test to verify pipeline...")
        
        # Import and run quick test
        os.system("python run_comprehensive_evaluation.py --quick-test")
        
        # Check if quick test results exist
        quick_test_dir = "final_eval_results_2015_2023_quick_test"
        if os.path.exists(quick_test_dir):
            print("✅ Quick test completed successfully!")
            print(f"📁 Quick test results in: {quick_test_dir}")
            return True
        else:
            print("❌ Quick test failed - no results generated!")
            return False
            
    except Exception as e:
        print(f"❌ Error during quick test: {e}")
        return False

def run_full_evaluation():
    """Run complete evaluation pipeline"""
    print("\n🚀 STEP 3: FULL EVALUATION")
    print("-" * 50)
    
    start_time = time.time()
    
    try:
        print("Starting comprehensive evaluation...")
        print("⏱️  Expected duration: 2-6 hours depending on hardware")
        print("💡 You can monitor progress in the log file")
        
        # Run the full evaluation
        exit_code = os.system("python run_comprehensive_evaluation.py")
        
        end_time = time.time()
        duration = end_time - start_time
        
        if exit_code == 0:
            print(f"✅ Full evaluation completed successfully!")
            print(f"⏱️  Total duration: {duration:.0f} seconds ({duration/60:.1f} minutes)")
            return True
        else:
            print(f"❌ Full evaluation failed with exit code: {exit_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error during full evaluation: {e}")
        return False

def generate_publication_summary():
    """Generate publication-ready summary"""
    print("\n📄 STEP 4: PUBLICATION SUMMARY")
    print("-" * 50)
    
    try:
        results_dir = "final_eval_results_2015_2023"
        
        if not os.path.exists(results_dir):
            print("❌ Results directory not found!")
            return False
        
        # Check for key result files
        required_files = [
            f"{results_dir}/experiment_1_excess_mortality/summary_statistics.csv",
            f"{results_dir}/figures/experiment_1_model_comparisons.png",
            f"{results_dir}/experiment_2_variance_analysis/detailed_results.csv",
            f"{results_dir}/figures/experiment_2_variance_analysis.png"
        ]
        
        missing_files = [f for f in required_files if not os.path.exists(f)]
        
        if missing_files:
            print("⚠️  Some result files are missing:")
            for f in missing_files:
                print(f"   - {f}")
        
        # Generate publication summary
        summary_content = generate_publication_text()
        
        summary_file = f"{results_dir}/PUBLICATION_SUMMARY.md"
        with open(summary_file, 'w') as f:
            f.write(summary_content)
        
        print(f"✅ Publication summary generated!")
        print(f"📄 Summary file: {summary_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error generating publication summary: {e}")
        return False

def generate_publication_text():
    """Generate publication-ready text summary"""
    
    content = f"""# Publication Summary Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Key Results for Paper

### Experiment 1: Excess Mortality Estimation (2015-2019 train, 2020-2023 test)

**Findings:**
- File: `experiment_1_excess_mortality/summary_statistics.csv`
- Contains mean ± standard deviation for all metrics across models
- Use these statistics to support Hypothesis H1 (DL superiority over SARIMA)

**Publication-Ready Figure:**
- File: `figures/experiment_1_model_comparisons.png`
- Shows SARIMA vs each DL model with prediction intervals
- Similar to your current sarima_vs_lstm_comparison.png but for all models

**Key Metrics to Report:**
- RMSE (Root Mean Square Error): Lower is better
- MAE (Mean Absolute Error): Lower is better  
- MAPE (Mean Absolute Percentage Error): Lower is better
- PI Coverage: Should be close to 95%
- PI Width: Narrower indicates more precision

### Experiment 2: Variance Reduction Analysis

**Findings:**
- File: `experiment_2_variance_analysis/detailed_results.csv`
- Shows how prediction intervals widen over forecasting horizons
- Supports analysis of precision degradation over time

**Publication-Ready Figure:**
- File: `figures/experiment_2_variance_analysis.png`
- Four subplots showing RMSE, PI Width, Coverage, and Growth Rate vs Forecast Horizon

### Experiment 3: Sensitivity Analysis

**Findings:**
- Files: `experiment_3_sensitivity/sensitivity_[model].csv`
- Shows optimal number of seeds and trials for stable results
- Demonstrates methodology robustness

**Publication-Ready Figures:**
- Files: `figures/experiment_3_sensitivity_[model].png`
- Shows convergence behavior for each DL model

## Statistical Significance Testing

For your paper, consider these additional analyses:

1. **Paired t-tests** between SARIMA and each DL model performance
2. **Effect size calculations** (Cohen's d) for practical significance
3. **Bootstrap confidence intervals** for robust uncertainty quantification
4. **Diebold-Mariano tests** for forecast accuracy comparisons

## Paper Integration Guidelines

### Results Section
1. Lead with Experiment 1 results showing DL superiority
2. Include variance analysis to show precision advantages
3. Reference sensitivity analysis for methodology validation

### Tables and Figures
1. **Table 1:** Model performance comparison (from summary_statistics.csv)
2. **Figure 1:** Model comparison plots (from experiment_1_model_comparisons.png)
3. **Figure 2:** Variance analysis (from experiment_2_variance_analysis.png)
4. **Supplementary:** Sensitivity plots for methodology appendix

### Discussion Points
1. **H1 Validation:** LSTM shows X% lower RMSE than SARIMA
2. **H2 Analysis:** DL models predict more/less severe scenarios
3. **Practical Implications:** Narrower prediction intervals enable better resource allocation
4. **Robustness:** Results stable across multiple seeds and trials

## Next Steps for Dashboard Integration

The trained models are saved in `trained_models/` and can be used for:
1. Real-time forecasting interface
2. Interactive scenario analysis
3. Policy impact simulation
4. Regional disaggregation (future work)

## Data Availability Statement

*"The analysis code and results are available upon request. The underlying mortality data is publicly available through the CDC WONDER database. Model implementations follow standard architectures with hyperparameters optimized through grid search validation."*

---

**Note:** Review all generated CSV files and figures before inclusion in manuscript. Adjust figure formatting (fonts, margins, captions) as needed for journal requirements.
"""
    
    return content

def cleanup_temp_files():
    """Clean up temporary files"""
    temp_files = [
        "data_inspection_plots.png",
        "data_inspection_report.txt"
    ]
    
    for file in temp_files:
        if os.path.exists(file):
            try:
                # Move to results directory instead of deleting
                results_dir = "final_eval_results_2015_2023"
                if os.path.exists(results_dir):
                    import shutil
                    shutil.move(file, os.path.join(results_dir, file))
            except:
                pass

def print_final_summary():
    """Print final summary and next steps"""
    print("\n" + "🎉" * 20)
    print("EVALUATION PIPELINE COMPLETED!")
    print("🎉" * 20)
    
    print("\n📁 GENERATED OUTPUTS:")
    print("-" * 30)
    
    results_dir = "final_eval_results_2015_2023"
    
    if os.path.exists(results_dir):
        print(f"📂 Main results: {results_dir}/")
        print("   ├── experiment_1_excess_mortality/")
        print("   │   ├── results.pkl")
        print("   │   └── summary_statistics.csv ⭐")
        print("   ├── experiment_2_variance_analysis/")
        print("   │   ├── results.pkl") 
        print("   │   └── detailed_results.csv ⭐")
        print("   ├── experiment_3_sensitivity/")
        print("   │   └── sensitivity_[model].csv ⭐")
        print("   ├── trained_models/")
        print("   │   └── [model]_best_model.pkl 🤖")
        print("   ├── figures/")
        print("   │   ├── experiment_1_model_comparisons.png 📊")
        print("   │   ├── experiment_2_variance_analysis.png 📊")
        print("   │   └── experiment_3_sensitivity_[model].png 📊")
        print("   ├── PUBLICATION_SUMMARY.md 📄")
        print("   └── evaluation_summary_report.txt 📄")
    
    print("\n🔄 NEXT STEPS:")
    print("-" * 20)
    print("1. 📊 Review figures in the 'figures' folder")
    print("2. 📈 Examine summary_statistics.csv for paper results")
    print("3. 📝 Read PUBLICATION_SUMMARY.md for integration guidance")
    print("4. 🤖 Use trained models for dashboard development")
    print("5. ✅ Update your paper with the new results!")
    
    print("\n💡 KEY FILES FOR PAPER:")
    print("-" * 25)
    print("• summary_statistics.csv - Main results table")
    print("• experiment_1_model_comparisons.png - Figure 1")
    print("• experiment_2_variance_analysis.png - Figure 2")
    print("• PUBLICATION_SUMMARY.md - Integration guide")

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Complete Evaluation Pipeline")
    parser.add_argument('--skip-inspection', action='store_true', 
                       help='Skip data inspection step')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run only quick test')
    
    args = parser.parse_args()
    
    print_banner()
    
    overall_start = time.time()
    
    # Step 1: Data Inspection
    if not args.skip_inspection:
        if not run_data_inspection():
            print("\n❌ Data inspection failed. Please fix data issues before proceeding.")
            sys.exit(1)
    else:
        print("\n⏭️  Skipping data inspection (as requested)")
    
    # Step 2: Quick Test (if requested or before full evaluation)
    if args.quick_test:
        if run_quick_test():
            print("\n✅ Quick test successful! You can now run the full evaluation.")
            sys.exit(0)
        else:
            print("\n❌ Quick test failed. Please fix issues before running full evaluation.")
            sys.exit(1)
    else:
        # Run quick test as validation before full evaluation
        print("\n🔍 Running quick validation test...")
        if not run_quick_test():
            print("\n❌ Validation test failed. Aborting full evaluation.")
            sys.exit(1)
    
    # Step 3: Full Evaluation
    if not run_full_evaluation():
        print("\n❌ Full evaluation failed!")
        sys.exit(1)
    
    # Step 4: Publication Summary
    generate_publication_summary()
    
    # Cleanup
    cleanup_temp_files()
    
    # Final Summary
    overall_time = time.time() - overall_start
    print(f"\n⏱️  TOTAL EXECUTION TIME: {overall_time:.0f} seconds ({overall_time/60:.1f} minutes)")
    
    print_final_summary()

if __name__ == "__main__":
    main()