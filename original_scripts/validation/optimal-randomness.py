import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

seed = 42

# Directory containing the folders
base_dir = f"../tables_seeds/seed{seed}"

# Function to calculate confidence intervals
def calculate_confidence_intervals(predictions, alpha=0.05):
    mean_pred = np.mean(predictions, axis=0)
    std_pred = np.std(predictions, axis=0)
    z_score = 1.96  # 95% confidence
    margin_of_error = z_score * (std_pred / np.sqrt(predictions.shape[0]))
    lower_bound = mean_pred - margin_of_error
    upper_bound = mean_pred + margin_of_error
    return lower_bound, upper_bound, std_pred

# Function to calculate prediction intervals
def calculate_prediction_intervals(actual, predictions, alpha=0.05):
    residuals = actual - predictions
    std_residual = np.std(residuals, axis=0)
    z_score = 1.96  # 95% confidence
    margin_of_error = z_score * std_residual
    lower_bound = predictions - margin_of_error
    upper_bound = predictions + margin_of_error
    return lower_bound, upper_bound, std_residual

# Function to calculate percent overlap
def calculate_overlap(lower1, upper1, lower2, upper2):
    overlap_count = np.sum((upper1 >= lower2) & (lower1 <= upper2), axis=1)
    percent_overlap = (overlap_count / lower1.shape[1]) * 100
    return np.mean(percent_overlap)

# Initialize a dictionary to store results
results_dict = {}

# Traverse the directory structure
for root, dirs, files in os.walk(base_dir):
    trial_data = []
    for file in files:
        if file.endswith(".csv"):
            # Extract hyperparameters from folder and file names
            folder_name = os.path.basename(root)
            # Skip invalid folders (like .ipynb_checkpoints)
            if not folder_name.startswith("batch_"):
                continue
                
            print(folder_name)
            file_parts = file.split("_")
            trial_num = int(file_parts[1])
            batch_size = int(file_parts[5])
            # loss_type = folder_name.split("_")[3]
            epochs = int(folder_name.split("_")[-1])
            lookback = int(file_parts[2][:-5])
            
            # Load the CSV
            file_path = os.path.join(root, file)
            df = pd.read_csv(file_path)

            # Extract data for calculations
            original_test = df["Deaths"].iloc[49:].values
            print(df.iloc[48]["Month"])
            lstm_test = df["LSTM Predictions"].iloc[49:].values
            sarima_test = df["SARIMA Predictions"].iloc[49:].values

            # Store trial data
            trial_data.append((original_test, lstm_test, sarima_test))
    
    if len(trial_data) == 10:
        original_tests = np.array([x[0] for x in trial_data])
        lstm_tests = np.array([x[1] for x in trial_data])
        sarima_tests = np.array([x[2] for x in trial_data])

        # Calculate averaged metrics
        lstm_test_rmse = np.mean([np.sqrt(mean_squared_error(original_tests[i], lstm_tests[i])) for i in range(10)])
        lstm_test_rmse_std = np.std([np.sqrt(mean_squared_error(original_tests[i], lstm_tests[i])) for i in range(10)])
        lstm_test_mape = np.mean([mean_absolute_percentage_error(original_tests[i], lstm_tests[i]) * 100 for i in range(10)])
        lstm_test_mape_std = np.std([mean_absolute_percentage_error(original_tests[i], lstm_tests[i]) * 100 for i in range(10)])
        sarima_test_rmse = np.mean([np.sqrt(mean_squared_error(original_tests[i], sarima_tests[i])) for i in range(10)])
        sarima_test_rmse_std = np.std([np.sqrt(mean_squared_error(original_tests[i], sarima_tests[i])) for i in range(10)])
        sarima_test_mape = np.mean([mean_absolute_percentage_error(original_tests[i], sarima_tests[i]) * 100 for i in range(10)])
        sarima_test_mape_std = np.std([mean_absolute_percentage_error(original_tests[i], sarima_tests[i]) * 100 for i in range(10)])

        lower_lstm, upper_lstm, lstm_std = calculate_prediction_intervals(original_tests, lstm_tests)
        lower_sarima, upper_sarima, sarima_std = calculate_prediction_intervals(original_tests, sarima_tests)
        percent_overlap = calculate_overlap(lower_lstm, upper_lstm, lower_sarima, upper_sarima)

        # Store results
        results_dict[(lookback, batch_size, loss_type, epochs)] = {
            "Lookback Period": lookback,
            "Batch Size": batch_size,
            # "Loss Type": loss_type,
            "Epochs": epochs,
            "LSTM RMSE": lstm_test_rmse,
            "LSTM RMSE Std": lstm_test_rmse_std,
            "LSTM MAPE": lstm_test_mape,
            "LSTM MAPE Std": lstm_test_mape_std,
            "SARIMA RMSE": sarima_test_rmse,
            "SARIMA RMSE Std": sarima_test_rmse_std,
            "SARIMA MAPE": sarima_test_mape,
            "SARIMA MAPE Std": sarima_test_mape_std,
            "Percent Overlap": percent_overlap,
            "LSTM Prediction Interval Lower": lower_lstm.mean(),
            "LSTM Prediction Interval Upper": upper_lstm.mean(),
            "SARIMA Prediction Interval Lower": lower_sarima.mean(),
            "SARIMA Prediction Interval Upper": upper_sarima.mean()
        }

output_file = f"{base_dir}/hyper_csv_metrics_{batch_size}_{epochs}.csv"

# Save results to CSV
results_df = pd.DataFrame(list(results_dict.values()))
results_df.to_csv(output_file, index=False)
print(f"Hyper CSV saved to {output_file}")