import pandas as pd
import numpy as np
from itertools import product
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tqdm import tqdm
import warnings
import random

warnings.filterwarnings("ignore")

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

# Aggregate metrics for each configuration
all_results = []

print("Running SARIMA grid search with 30 trials per configuration...")
for order in tqdm(param_grid):
    for seasonal_order in seasonal_grid:
        metrics = {'AIC': [], 'BIC': [], 'MAE': [], 'RMSE': []}
        for _ in range(N_TRIALS):
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

        if len(metrics['RMSE']) >= 5:  # Keep only those with enough valid runs
            all_results.append({
                'order': order,
                'seasonal_order': seasonal_order,
                'mean_RMSE': np.mean(metrics['RMSE']),
                'std_RMSE': np.std(metrics['RMSE']),
                'mean_MAE': np.mean(metrics['MAE']),
                'std_MAE': np.std(metrics['MAE']),
                'mean_AIC': np.mean(metrics['AIC']),
                'std_AIC': np.std(metrics['AIC']),
                'mean_BIC': np.mean(metrics['BIC']),
                'std_BIC': np.std(metrics['BIC']),
                'num_successful_trials': len(metrics['RMSE'])
            })

# Save and display sorted results
results_df_agg = pd.DataFrame(all_results)
results_df_agg = results_df.sort_values(by='mean_RMSE')  # or use 'mean_BIC'

results_df_agg.to_csv('sarima_gridsearch_30trials_rmse.csv', index=False)
# print(results_df_agg.head(10))

results_df_agg = pd.DataFrame(all_results)
results_df_agg = results_df.sort_values(by='mean_BIC')  # or use 'mean_BIC'

results_df_agg.to_csv('sarima_gridsearch_30trials_bic.csv', index=False)