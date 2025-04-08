# example_usage.py
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import os
import itertools
from time_series_pipeline import Pipeline, LSTMModel, SARIMAModel

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Path to the data
DATA_PATH = 'data/state_month_overdose.xlsx'
OUTPUT_DIR = 'results'

# Create the pipeline
pipeline = Pipeline(
    data_path=DATA_PATH,
    output_dir=OUTPUT_DIR,
    config={
        'random_seeds': [42, 123, 456, 789, 101],
        'validation_periods': [
            ('2019-11-01', '2020-01-01'),
            ('2019-09-01', '2020-01-01'),
            ('2019-07-01', '2020-01-01'),
            ('2019-01-01', '2020-01-01'),
            ('2018-07-01', '2020-01-01'),
            ('2018-01-01', '2020-01-01')
        ],
        'look_back_periods': [3, 5, 7, 9, 11],
        'final_test_period': ('2020-01-01', None)
    }
)

# Run LSTM experiment
lstm_hyperparams = {
    'look_back': [3, 5, 7, 9, 11],
    'lstm_units': [50, 100],
    'batch_size': [1, 8, 16],
    'epochs': [50, 100]
}

lstm_results = pipeline.run_experiment(
    model_class=LSTMModel,
    hyperparameter_grid=lstm_hyperparams
)

# Run SARIMA experiment
sarima_hyperparams = {
    'order': [(1, 1, 1), (2, 1, 2)],
    'seasonal_order': [(1, 1, 1, 12), (0, 1, 1, 12)]
}

sarima_results = pipeline.run_experiment(
    model_class=SARIMAModel,
    hyperparameter_grid=sarima_hyperparams
)

# Compare models
comparison = pipeline.compare_models([lstm_results, sarima_results])
print("Model Comparison:")
print(comparison)


#############

# # Run TFT experiment
# tft_hyperparams = {
#     'look_back': [3, 5, 7, 9, 11],
#     'hidden_size': [64, 128],
#     'num_attention_heads': [4, 8],
#     'dropout_rate': [0.1, 0.2],
#     'batch_size': [16, 32],
#     'epochs': [100]
# }

# tft_results = pipeline.run_experiment(
#     model_class=TFTModel,
#     hyperparameter_grid=tft_hyperparams
# )

# # Compare all models
# comparison = pipeline.compare_models([lstm_results, sarima_results, tft_results])
# print("Model Comparison (All Models):")
# print(comparison)