import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# Directory containing the folders
base_dir = "tables"

# Output file
output_file = "hyper_csv_metrics.csv"

# Function to calculate confidence intervals
def calculate_confidence_intervals(predictions, alpha=0.05):
    mean_pred = np.mean(predictions)
    std_pred = np.std(predictions)
    z_score = 1.96  # 95% confidence
    margin_of_error = z_score * (std_pred / np.sqrt(len(predictions)))
    lower_bound = predictions - margin_of_error
    upper_bound = predictions + margin_of_error
    return lower_bound, upper_bound

# Function to calculate prediction intervals
def calculate_prediction_intervals(actual, predictions, alpha=0.05):
    residuals = actual - predictions
    std_residual = np.std(residuals)
    z_score = 1.96  # 95% confidence
    margin_of_error = z_score * std_residual
    lower_bound = predictions - margin_of_error
    upper_bound = predictions + margin_of_error
    return lower_bound, upper_bound

# Function to calculate percent overlap
def calculate_overlap(lower1, upper1, lower2, upper2):
    overlap_count = sum((u1 >= l2) & (l1 <= u2) for l1, u1, l2, u2 in zip(lower1, upper1, lower2, upper2))
    percent_overlap = (overlap_count / len(lower1)) * 100
    return percent_overlap

# Initialize a list to store results
results = []

# Traverse the directory structure
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.endswith(".csv"):
            # Extract hyperparameters from folder and file names
            folder_name = os.path.basename(root)
            file_name = file
            batch_size = int(folder_name.split("_")[1])
            loss_type = folder_name.split("_")[3]
            epochs = int(folder_name.split("_")[-1])
            lookback = int(file_name.split("month")[0])
            
            # Load the CSV
            file_path = os.path.join(root, file)
            df = pd.read_csv(file_path)

            # Extract data for calculations
            original_train = df["Deaths"].iloc[lookback:62]
            original_test = df["Deaths"].iloc[62:]
            lstm_train = df["LSTM Predictions"].iloc[lookback:62]
            lstm_test = df["LSTM Predictions"].iloc[62:]
            sarima_train = df["SARIMA Predictions"].iloc[lookback:62]
            sarima_test = df["SARIMA Predictions"].iloc[62:]

            # Calculate metrics
            lstm_test_rmse = np.sqrt(mean_squared_error(original_test, lstm_test))
            lstm_test_mape = mean_absolute_percentage_error(original_test, lstm_test) * 100
            sarima_test_rmse = np.sqrt(mean_squared_error(original_test, sarima_test))
            sarima_test_mape = mean_absolute_percentage_error(original_test, sarima_test) * 100

            lower_lstm, upper_lstm = calculate_prediction_intervals(original_test, lstm_test)
            lower_sarima, upper_sarima = calculate_prediction_intervals(original_test, sarima_test)
            percent_overlap = calculate_overlap(lower_lstm, upper_lstm, lower_sarima, upper_sarima)

            # Append results
            results.append({
                "Lookback Period": lookback,
                "Batch Size": batch_size,
                "Loss Type": loss_type,
                "Epochs": epochs,
                "LSTM RMSE": lstm_test_rmse,
                "LSTM MAPE": lstm_test_mape,
                "SARIMA RMSE": sarima_test_rmse,
                "SARIMA MAPE": sarima_test_mape,
                "Percent Overlap": percent_overlap
            })

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv('../tables/' + output_file, index=False)
print(f"Hyper CSV saved to {output_file}")
