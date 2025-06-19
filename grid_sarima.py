import pandas as pd
import numpy as np
from itertools import product
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm
import warnings
import random
import os
warnings.filterwarnings("ignore")

# Load and preprocess data
df = pd.read_excel('data/state_month_overdose.xlsx')
df['Deaths'] = df['Deaths'].apply(lambda x: 0 if x == 'Suppressed' else int(x))
df['Month'] = pd.to_datetime(df['Month'])

def create_train_val_test_split(df, train_end='2019-01-01', val_end='2020-01-01', test_end='2020-12-01'):
    train = df[df['Month'] < train_end]
    val = df[(df['Month'] >= train_end) & (df['Month'] < val_end)]
    test = df[(df['Month'] >= val_end) & (df['Month'] <= test_end)]
    return train, val, test

train_df, val_df, _ = create_train_val_test_split(df)
train_series = train_df['Deaths'].reset_index(drop=True)
val_series = val_df['Deaths'].reset_index(drop=True)

# SARIMA parameter grid
p = d = q = range(0, 3)
P = D = Q = range(0, 3)
s = 12
param_grid = list(product(p, d, q))
seasonal_grid = list(product(P, D, Q))

# How many times to repeat each combination
N_TRIALS = 30

# Output files
output_file = 'sarima_gridsearch_results.csv'

# Check if output file exists and load completed combinations
completed_combinations = set()
if os.path.exists(output_file):
    try:
        existing_df = pd.read_csv(output_file)
        for _, row in existing_df.iterrows():
            # Parse the order and seasonal_order back from string representation
            order = eval(row['order'])
            seasonal_order = eval(row['seasonal_order'])
            completed_combinations.add((order, seasonal_order))
        print(f"Found {len(completed_combinations)} completed combinations. Resuming...")
    except Exception as e:
        print(f"Error reading existing file: {e}. Starting fresh...")
        completed_combinations = set()

# Initialize CSV file with headers if it doesn't exist
if not os.path.exists(output_file):
    headers = ['order', 'seasonal_order', 'mean_RMSE', 'std_RMSE', 'mean_MAE', 'std_MAE', 
               'mean_AIC', 'std_AIC', 'mean_BIC', 'std_BIC', 'num_successful_trials']
    pd.DataFrame(columns=headers).to_csv(output_file, index=False)

print("Running SARIMA grid search with 30 trials per configuration...")

# Calculate total combinations for progress tracking
total_combinations = len(param_grid) * len(seasonal_grid)
completed_count = len(completed_combinations)
remaining_combinations = total_combinations - completed_count

print(f"Total combinations: {total_combinations}")
print(f"Completed: {completed_count}")
print(f"Remaining: {remaining_combinations}")

# Track progress
progress_bar = tqdm(total=remaining_combinations, desc="Grid Search Progress")

for order in param_grid:
    for seasonal_order in seasonal_grid:
        # Skip if already completed
        if (order, seasonal_order) in completed_combinations:
            continue
            
        metrics = {'AIC': [], 'BIC': [], 'MAE': [], 'RMSE': []}
        
        for trial in range(N_TRIALS):
            try:
                # Randomized seed to introduce variability
                random_seed = random.randint(0, 10000)
                np.random.seed(random_seed)
                
                model = SARIMAX(train_series,
                                order=order,
                                seasonal_order=(seasonal_order[0], seasonal_order[1], seasonal_order[2], s),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
                results = model.fit(disp=False)
                
                preds = results.predict(start=len(train_series), end=len(train_series)+len(val_series)-1)
                preds = preds[:len(val_series)]  # safeguard
                
                metrics['AIC'].append(results.aic)
                metrics['BIC'].append(results.bic)
                metrics['MAE'].append(mean_absolute_error(val_series, preds))
                metrics['RMSE'].append(np.sqrt(mean_squared_error(val_series, preds)))
                
            except Exception:
                continue
        
        # Only save if we have enough valid runs
        if len(metrics['RMSE']) >= 5:
            result_row = {
                'order': str(order),  # Convert to string for CSV storage
                'seasonal_order': str(seasonal_order),
                'mean_RMSE': np.mean(metrics['RMSE']),
                'std_RMSE': np.std(metrics['RMSE']),
                'mean_MAE': np.mean(metrics['MAE']),
                'std_MAE': np.std(metrics['MAE']),
                'mean_AIC': np.mean(metrics['AIC']),
                'std_AIC': np.std(metrics['AIC']),
                'mean_BIC': np.mean(metrics['BIC']),
                'std_BIC': np.std(metrics['BIC']),
                'num_successful_trials': len(metrics['RMSE'])
            }
            
            # Append to CSV file immediately
            result_df = pd.DataFrame([result_row])
            result_df.to_csv(output_file, mode='a', header=False, index=False)
            
        progress_bar.update(1)

progress_bar.close()

# Create sorted output files
print("Creating sorted output files...")
final_df = pd.read_csv(output_file)

# Sort by RMSE and save
final_df_rmse = final_df.sort_values(by='mean_RMSE')
final_df_rmse.to_csv('sarima_gridsearch_30trials_rmse.csv', index=False)

# Sort by BIC and save  
final_df_bic = final_df.sort_values(by='mean_BIC')
final_df_bic.to_csv('sarima_gridsearch_30trials_bic.csv', index=False)

print("Grid search completed!")
print(f"Results saved to: {output_file}")
print(f"Sorted results saved to: sarima_gridsearch_30trials_rmse.csv and sarima_gridsearch_30trials_bic.csv")

# Display top 10 results
print("\nTop 10 results by RMSE:")
print(final_df_rmse.head(10))