"""Module for creating plots from model results."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator

def plot_model_vs_sarima(model_name, output_dir, test_start_date=None):
    """
    Generate plots comparing a model against SARIMA model.
    
    Parameters
    ----------
    model_name : str
        Name of the model to compare against SARIMA
    output_dir : str
        Directory containing model outputs
    test_start_date : str, optional
        Start date for test data, used to mark the beginning of forecasting period
    """
    model_dir = os.path.join(output_dir, model_name)
    sarima_dir = os.path.join(output_dir, 'sarima')
    
    # Create comparisons directory
    comparisons_dir = os.path.join(output_dir, 'comparisons')
    os.makedirs(comparisons_dir, exist_ok=True)
    
    # Get all parameter combinations for the model
    combo_dirs = [d for d in os.listdir(model_dir) 
                 if os.path.isdir(os.path.join(model_dir, d))]
    
    for combo in combo_dirs:
        # Ensure the same combo exists for SARIMA
        model_combo_dir = os.path.join(model_dir, combo)
        sarima_combo_dir = os.path.join(sarima_dir, combo)
        
        if not os.path.exists(sarima_combo_dir):
            print(f"Skipping {combo} - not found in SARIMA results")
            continue
        
        # Get all trial directories for this combo
        trial_dirs = [d for d in os.listdir(model_combo_dir) 
                     if os.path.isdir(os.path.join(model_combo_dir, d))]
        
        for trial in trial_dirs:
            # Ensure the same trial exists for SARIMA
            model_trial_dir = os.path.join(model_combo_dir, trial)
            sarima_trial_dir = os.path.join(sarima_combo_dir, trial)
            
            if not os.path.exists(sarima_trial_dir):
                print(f"Skipping {combo}/{trial} - not found in SARIMA results")
                continue
            
            # Load predictions
            model_pred_file = os.path.join(model_trial_dir, "predictions.csv")
            sarima_pred_file = os.path.join(sarima_trial_dir, "predictions.csv")
            
            if not os.path.exists(model_pred_file) or not os.path.exists(sarima_pred_file):
                print(f"Skipping {combo}/{trial} - prediction files not found")
                continue
                
            model_df = pd.read_csv(model_pred_file)
            sarima_df = pd.read_csv(sarima_pred_file)
            
            # Ensure Month column is datetime
            model_df['Month'] = pd.to_datetime(model_df['Month'])
            sarima_df['Month'] = pd.to_datetime(sarima_df['Month'])
            
            # Create the plot
            fig, ax = plt.figure(figsize=(15, 8)), plt.gca()
            
            # Plot actual values
            ax.plot(model_df['Month'], model_df['Deaths'], 
                    label='Actual Data', color='blue', linewidth=2)
            
            # Plot model predictions with prediction intervals
            ax.plot(model_df['Month'], model_df['Predictions'], 
                    label=f'{model_name.upper()} Predictions', color='red', linewidth=2)
            ax.fill_between(model_df['Month'], model_df['Lower PI'], model_df['Upper PI'], 
                           alpha=0.2, color='red', label=f'{model_name.upper()} 95% PI')
            
            # Plot SARIMA predictions with prediction intervals
            ax.plot(sarima_df['Month'], sarima_df['Predictions'], 
                    label='SARIMA Predictions', color='green', linewidth=2)
            ax.fill_between(sarima_df['Month'], sarima_df['Lower PI'], sarima_df['Upper PI'], 
                           alpha=0.2, color='green', label='SARIMA 95% PI')
            
            # Mark the start of the forecasting period if provided
            if test_start_date:
                ax.axvline(pd.to_datetime(test_start_date), color='black', linestyle='--',
                          label='Start of Forecasting')
            
            # Format the plot
            plt.title(f'Mortality: Actual vs {model_name.upper()} vs SARIMA Predictions with Prediction Intervals', 
                     fontsize=16)
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Deaths', fontsize=12)
            
            # Format x-axis to show dates properly
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.xticks(rotation=45)
            
            # Add legend and adjust layout
            plt.legend(fontsize=12)
            plt.tight_layout()
            
            # Add a caption with parameter details
            caption = (f"Figure: Comparison of actual substance overdose mortality data with "
                      f"{model_name.upper()} and SARIMA model predictions. "
                      f"Parameters: {combo.replace('_', ', ')}, {trial}.")
            plt.figtext(0.5, 0.01, caption, wrap=True, horizontalalignment='center', fontsize=10)
            
            # Save the figure
            fig_path = os.path.join(comparisons_dir, f'{model_name}_vs_sarima_{combo}_{trial}.png')
            plt.savefig(fig_path, bbox_inches='tight', dpi=300)
            plt.close()
            
            print(f"Created comparison plot: {fig_path}")

def plot_metric_comparison(output_dir):
    """
    Create bar plots comparing metrics across models.
    
    Parameters
    ----------
    output_dir : str
        Directory containing aggregated metrics
    """
    # Load aggregated metrics
    metrics_file = os.path.join(output_dir, 'aggregated_metrics.csv')
    if not os.path.exists(metrics_file):
        print(f"Metrics file not found: {metrics_file}")
        return
        
    metrics_df = pd.read_csv(metrics_file)
    
    # Create comparisons directory
    comparisons_dir = os.path.join(output_dir, 'comparisons')
    os.makedirs(comparisons_dir, exist_ok=True)
    
    # Plot RMSE comparison
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    
    # Group by model and create bar plots
    models = metrics_df['Model'].unique()
    x = np.arange(len(models))
    width = 0.25
    
    # Get average metrics for each model
    rmse_means = [metrics_df[metrics_df['Model'] == model]['RMSE Mean'].mean() for model in models]
    rmse_stds = [metrics_df[metrics_df['Model'] == model]['RMSE Mean'].std() for model in models]
    
    # Plot RMSE
    plt.bar(x, rmse_means, width, label='RMSE', yerr=rmse_stds, capsize=5)
    
    plt.xlabel('Model')
    plt.ylabel('Root Mean Squared Error (RMSE)')
    plt.title('RMSE Comparison Across Models')
    plt.xticks(x, models)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(comparisons_dir, 'rmse_comparison.png'), dpi=300)
    plt.close()
    
    # Plot MAPE comparison
    plt.figure(figsize=(12, 8))
    
    # Get average MAPE for each model
    mape_means = [metrics_df[metrics_df['Model'] == model]['MAPE Mean'].mean() for model in models]
    mape_stds = [metrics_df[metrics_df['Model'] == model]['MAPE Mean'].std() for model in models]
    
    # Plot MAPE
    plt.bar(x, mape_means, width, label='MAPE', yerr=mape_stds, capsize=5)
    
    plt.xlabel('Model')
    plt.ylabel('Mean Absolute Percentage Error (%)')
    plt.title('MAPE Comparison Across Models')
    plt.xticks(x, models)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(comparisons_dir, 'mape_comparison.png'), dpi=300)
    plt.close()
    
    # Plot PI Width and Coverage
    plt.figure(figsize=(12, 8))
    
    # Get average PI width and coverage for each model
    pi_width_means = [metrics_df[metrics_df['Model'] == model]['PI Width Mean'].mean() for model in models]
    pi_coverage_means = [metrics_df[metrics_df['Model'] == model]['PI GT Overlap % Mean'].mean() for model in models]
    
    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Plot PI width on left y-axis
    color = 'tab:blue'
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Prediction Interval Width', color=color)
    ax1.bar(x - width/2, pi_width_means, width, label='PI Width', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Create second y-axis for coverage
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('% of Actual Values in 95% PI', color=color)
    ax2.bar(x + width/2, pi_coverage_means, width, label='PI Coverage', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Add title and adjust layout
    plt.title('Prediction Interval Comparison')
    plt.xticks(x, models)
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(comparisons_dir, 'pi_comparison.png'), dpi=300)
    plt.close()
    
    print("Created metric comparison plots in", comparisons_dir)

def plot_parameter_effects(output_dir):
    """
    Create plots showing the effect of different parameters on model performance.
    
    Parameters
    ----------
    output_dir : str
        Directory containing aggregated metrics
    """
    # Load aggregated metrics
    metrics_file = os.path.join(output_dir, 'aggregated_metrics.csv')
    if not os.path.exists(metrics_file):
        print(f"Metrics file not found: {metrics_file}")
        return
        
    metrics_df = pd.read_csv(metrics_file)
    
    # Create comparisons directory
    comparisons_dir = os.path.join(output_dir, 'comparisons')
    os.makedirs(comparisons_dir, exist_ok=True)
    
    # For each model, plot effect of lookback period on RMSE
    for model in metrics_df['Model'].unique():
        model_df = metrics_df[metrics_df['Model'] == model]
        
        # Group by lookback and calculate mean metrics
        lookback_groups = model_df.groupby('Lookback')
        lookbacks = sorted(model_df['Lookback'].unique())
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot RMSE vs lookback period
        rmse_means = [lookback_groups.get_group(lb)['RMSE Mean'].mean() for lb in lookbacks]
        rmse_stds = [lookback_groups.get_group(lb)['RMSE Mean'].std() for lb in lookbacks]
        
        plt.errorbar(lookbacks, rmse_means, yerr=rmse_stds, marker='o', linestyle='-', 
                    linewidth=2, markersize=8, label='RMSE')
        
        plt.xlabel('Lookback Period')
        plt.ylabel('Root Mean Squared Error (RMSE)')
        plt.title(f'Effect of Lookback Period on {model.upper()} Performance')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(lookbacks)
        
        # Format y-axis to show fixed decimal places
        plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=10, integer=False))
        
        plt.legend()
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(comparisons_dir, f'{model}_lookback_effect.png'), dpi=300)
        plt.close()
        
        # If model has batch size, plot effect of batch size on RMSE
        if 'lstm' in model.lower() or 'tcn' in model.lower():
            # Group by batch size
            batch_groups = model_df.groupby('Batch')
            batches = sorted(model_df['Batch'].unique())
            
            # Create figure
            plt.figure(figsize=(12, 8))
            
            # Plot RMSE vs batch size
            rmse_means = [batch_groups.get_group(b)['RMSE Mean'].mean() for b in batches]
            rmse_stds = [batch_groups.get_group(b)['RMSE Mean'].std() for b in batches]
            
            plt.errorbar(batches, rmse_means, yerr=rmse_stds, marker='o', linestyle='-', 
                        linewidth=2, markersize=8, label='RMSE')
            
            plt.xlabel('Batch Size')
            plt.ylabel('Root Mean Squared Error (RMSE)')
            plt.title(f'Effect of Batch Size on {model.upper()} Performance')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xticks(batches)
            
            # Format y-axis to show fixed decimal places
            plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=10, integer=False))
            
            plt.legend()
            plt.tight_layout()
            
            # Save figure
            plt.savefig(os.path.join(comparisons_dir, f'{model}_batch_effect.png'), dpi=300)
            plt.close()
    
    print("Created parameter effect plots in", comparisons_dir)