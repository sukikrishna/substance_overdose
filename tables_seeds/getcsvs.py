import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# Set the base directory
base_dir = "seed42"

# Function to compute prediction intervals
def calculate_prediction_intervals(actual, predictions, alpha=0.05):
    residuals = actual - predictions
    std_residual = np.std(residuals, axis=0)
    z_score = 1.96  # 95% confidence
    margin_of_error = z_score * std_residual
    lower_bound = predictions - margin_of_error
    upper_bound = predictions + margin_of_error
    return lower_bound.mean(), upper_bound.mean(), std_residual.mean()

# Loop over all batch directories
for batch_folder in os.listdir(base_dir):
    batch_path = os.path.join(base_dir, batch_folder)
    if not os.path.isdir(batch_path):
        continue
    
    results_dict = {}
    
    # Traverse the directory
    for file in os.listdir(batch_path):
        if file.endswith(".csv"):
            file_parts = file.split("_")
            trial_num = int(file_parts[1])
            lookback = int(file_parts[2][:-5])  # Extract lookback period
            batch_size = int(file_parts[5])
            epochs = int(file_parts[-1][:-4])  # Extract epochs from filename
            
            # Load CSV
            file_path = os.path.join(batch_path, file)
            df = pd.read_csv(file_path)
            original_test = df["Deaths"].iloc[48:].values
            lstm_test = df["LSTM Predictions"].iloc[48:].values
            sarima_test = df["SARIMA Predictions"].iloc[48:].values
            
            # Store trial data
            if lookback not in results_dict:
                results_dict[lookback] = {
                    "batch_size": batch_size, "epochs": epochs,
                    "lstm_rmse": [], "lstm_mape": [], "sarima_rmse": [], "sarima_mape": [],
                    "lstm_lower": [], "lstm_upper": [], "sarima_lower": [], "sarima_upper": [],
                    "lstm_std": [], "sarima_std": []
                }
            
            results_dict[lookback]["lstm_rmse"].append(np.sqrt(mean_squared_error(original_test, lstm_test)))
            results_dict[lookback]["lstm_mape"].append(mean_absolute_percentage_error(original_test, lstm_test) * 100)
            results_dict[lookback]["sarima_rmse"].append(np.sqrt(mean_squared_error(original_test, sarima_test)))
            results_dict[lookback]["sarima_mape"].append(mean_absolute_percentage_error(original_test, sarima_test) * 100)
            
            lower_lstm, upper_lstm, lstm_std = calculate_prediction_intervals(original_test, lstm_test)
            lower_sarima, upper_sarima, sarima_std = calculate_prediction_intervals(original_test, sarima_test)
            results_dict[lookback]["lstm_lower"].append(lower_lstm)
            results_dict[lookback]["lstm_upper"].append(upper_lstm)
            results_dict[lookback]["sarima_lower"].append(lower_sarima)
            results_dict[lookback]["sarima_upper"].append(upper_sarima)
            results_dict[lookback]["lstm_std"].append(lstm_std)
            results_dict[lookback]["sarima_std"].append(sarima_std)
    
    # Compute averages and standard deviations
    final_results = []
    for lookback, data in sorted(results_dict.items()):
        final_results.append({
            "Lookback Period": lookback,
            "Batch Size": data["batch_size"],
            "Epochs": data["epochs"],
            "LSTM RMSE": np.mean(data["lstm_rmse"]),
            "LSTM RMSE Std": np.std(data["lstm_rmse"]),
            "LSTM MAPE": np.mean(data["lstm_mape"]),
            "LSTM MAPE Std": np.std(data["lstm_mape"]),
            "SARIMA RMSE": np.mean(data["sarima_rmse"]),
            "SARIMA RMSE Std": np.std(data["sarima_rmse"]),
            "SARIMA MAPE": np.mean(data["sarima_mape"]),
            "SARIMA MAPE Std": np.std(data["sarima_mape"]),
            "LSTM Prediction Interval Lower": np.mean(data["lstm_lower"]),
            "LSTM Prediction Interval Upper": np.mean(data["lstm_upper"]),
            "SARIMA Prediction Interval Lower": np.mean(data["sarima_lower"]),
            "SARIMA Prediction Interval Upper": np.mean(data["sarima_upper"]),
            "LSTM Std": np.mean(data["lstm_std"]),
            "SARIMA Std": np.mean(data["sarima_std"])
        })
    
    # Save results to CSV
    output_file = os.path.join(base_dir, f"hyper_csv_{batch_size}_{epochs}.csv")
    results_df = pd.DataFrame(final_results)
    results_df.to_csv(output_file, index=False)
    print(f"Aggregated results saved to {output_file}")