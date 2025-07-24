#!/usr/bin/env python3
"""
Streamlined Evaluation Pipeline for Substance Overdose Mortality Forecasting
=============================================================================

This simplified version focuses on Experiments 1 and 2 with reduced computational requirements:
- Single random seed (42) instead of multiple seeds
- 5 trials per model instead of 30
- Still provides comprehensive metrics and visualizations
- Saves all data for later figure editing

Experiments:
1. Excess mortality estimation (2015-2019 train, 2020-2023 test)
2. Variance analysis across different forecast horizons

Usage:
    python streamlined_evaluation.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, GRU, Attention
from tensorflow.keras.layers import RepeatVector, Concatenate, Input, Add
from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
import pickle
import json
from datetime import datetime
import math
from typing import Dict, List, Tuple, Any
import time

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Configuration
RESULTS_DIR = 'streamlined_eval_results_2015_2023'
DATA_PATH = 'data_updated/state_month_overdose_2015_2023.xlsx'
TRIALS_PER_MODEL = 30

# Optimal hyperparameters from previous grid search
OPTIMAL_PARAMS = {
    'sarima': {'order': (1, 0, 0), 'seasonal_order': (2, 2, 2, 12)},
    'lstm': {'lookback': 9, 'batch_size': 8, 'epochs': 100, 'units': 50, 'dropout': 0.1},
    'tcn': {'lookback': 5, 'batch_size': 32, 'epochs': 50, 'filters': 64, 'kernel_size': 3},
    'seq2seq_attn': {'lookback': 11, 'batch_size': 16, 'epochs': 50, 'encoder_units': 128, 'decoder_units': 128},
    'transformer': {'lookback': 5, 'batch_size': 32, 'epochs': 100, 'd_model': 64, 'n_heads': 2}
}

# Create results directory structure
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(f'{RESULTS_DIR}/experiment_1_excess_mortality', exist_ok=True)
os.makedirs(f'{RESULTS_DIR}/experiment_2_variance_analysis', exist_ok=True)
os.makedirs(f'{RESULTS_DIR}/trained_models', exist_ok=True)
os.makedirs(f'{RESULTS_DIR}/figures', exist_ok=True)
os.makedirs(f'{RESULTS_DIR}/data_exports', exist_ok=True)

class StreamlinedEvaluationPipeline:
    """Streamlined evaluation pipeline for rapid experimentation"""
    
    def __init__(self, data_path: str, results_dir: str):
        self.data_path = data_path
        self.results_dir = results_dir
        self.trained_models = {}
        
    def load_and_preprocess_data(self):
        """Load and preprocess the new format overdose dataset (2015-2023)"""
        print("Loading and preprocessing data...")
        
        # Load the Excel file
        df = pd.read_excel(self.data_path)
        
        print("Data columns:", df.columns.tolist())
        print("Data shape:", df.shape)
        
        # Handle the new data format
        # Expected: Row Labels, Month, Month_Code, Year_Code, Sum of Deaths
        
        # Create proper datetime from Row Labels
        if 'Row Labels' in df.columns:
            df['Date'] = pd.to_datetime(df['Row Labels'])
        else:
            # Fallback: construct from Year_Code and Month_Code
            df['Date'] = pd.to_datetime(df['Year_Code'].astype(str) + '-' + 
                                      df['Month_Code'].astype(str).str.zfill(2) + '-01')
        
        # Get deaths column
        if 'Sum of Deaths' in df.columns:
            df['Deaths'] = df['Sum of Deaths']
        elif 'Deaths' in df.columns:
            df['Deaths'] = df['Deaths']
        else:
            raise ValueError("Could not find deaths column in data")
        
        # Clean and sort data
        df = df[['Date', 'Deaths']].copy()
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Handle suppressed values if any
        df['Deaths'] = df['Deaths'].apply(lambda x: 0 if str(x).lower() == 'suppressed' else int(x))
        
        print(f"Processed data shape: {df.shape}")
        print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"Deaths range: {df['Deaths'].min()} to {df['Deaths'].max()}")
        
        return df
    
    def create_data_splits(self, df: pd.DataFrame):
        """Create all necessary data splits for both experiments"""
        
        splits = {}
        
        # Training data (2015-2019)
        train_data = df[df['Date'] <= '2019-12-31'].copy()
        
        # Experiment 1: Full test period (2020-2023)
        full_test_data = df[df['Date'] >= '2020-01-01'].copy()
        
        splits['experiment_1'] = {
            'train': train_data,
            'test': full_test_data,
            'description': 'Train: 2015-2019, Test: 2020-2023'
        }
        
        # Experiment 2: Different forecast horizons
        test_endpoints = ['2020-12-31', '2021-12-31', '2022-12-31', '2023-12-31']
        
        for i, end_date in enumerate(test_endpoints):
            test_data = df[(df['Date'] >= '2020-01-01') & (df['Date'] <= end_date)].copy()
            
            splits[f'horizon_{i+1}'] = {
                'train': train_data,
                'test': test_data,
                'description': f'Train: 2015-2019, Test: 2020-{end_date[:4]}',
                'test_length_months': len(test_data)
            }
        
        return splits
    
    def create_sequences(self, data: np.ndarray, lookback: int):
        """Create sequences for deep learning models"""
        X, y = [], []
        for i in range(len(data) - lookback):
            X.append(data[i:(i + lookback)])
            y.append(data[i + lookback])
        return np.array(X), np.array(y)
    
    def calculate_comprehensive_metrics(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Calculate comprehensive evaluation metrics with uncertainty quantification"""
        
        # Basic accuracy metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1))) * 100
        mse = mean_squared_error(y_true, y_pred)
        
        # Prediction intervals (95% confidence)
        residuals = y_true - y_pred
        std_residual = np.std(residuals)
        z_score = 1.96  # 95% confidence interval
        margin_of_error = z_score * std_residual
        
        lower_bound = y_pred - margin_of_error
        upper_bound = y_pred + margin_of_error
        
        # Coverage (what percentage of actual values fall within prediction interval)
        coverage = np.mean((y_true >= lower_bound) & (y_true <= upper_bound)) * 100
        
        # Interval width (precision measure)
        interval_width = np.mean(upper_bound - lower_bound)
        
        return {
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'MSE': mse,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'coverage': coverage,
            'interval_width': interval_width,
            'residuals': residuals
        }
    
    def train_sarima_model(self, train_data: pd.DataFrame, params: Dict):
        """Train SARIMA model"""
        deaths_series = train_data['Deaths'].values.astype(float)
        
        try:
            model = SARIMAX(deaths_series, 
                           order=params['order'], 
                           seasonal_order=params['seasonal_order'],
                           enforce_stationarity=False, 
                           enforce_invertibility=False)
            fitted_model = model.fit(disp=False, maxiter=200)
            return fitted_model
        except Exception as e:
            print(f"SARIMA training failed: {e}")
            return None
    
    def train_lstm_model(self, train_data: pd.DataFrame, params: Dict):
        """Train LSTM model"""
        deaths_series = train_data['Deaths'].values.astype(float)
        
        # Create sequences
        X, y = self.create_sequences(deaths_series, params['lookback'])
        X = X.reshape((X.shape[0], params['lookback'], 1))
        
        # Build model
        model = Sequential([
            LSTM(params['units'], activation='relu', input_shape=(params['lookback'], 1), 
                 return_sequences=True, dropout=params['dropout']),
            LSTM(params['units'], activation='relu', dropout=params['dropout']),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Train model
        model.fit(X, y, epochs=params['epochs'], batch_size=params['batch_size'], 
                 verbose=0, validation_split=0.1)
        
        return model
    
    def train_tcn_model(self, train_data: pd.DataFrame, params: Dict):
        """Train TCN model (simplified implementation)"""
        deaths_series = train_data['Deaths'].values.astype(float)
        
        # Create sequences
        X, y = self.create_sequences(deaths_series, params['lookback'])
        X = X.reshape((X.shape[0], params['lookback'], 1))
        
        # Build simplified TCN-like model using dilated convolutions
        try:
            # Try to use keras-tcn if available
            from tcn import TCN
            model = Sequential([
                TCN(input_shape=(params['lookback'], 1), 
                    dilations=[1, 2, 4, 8], 
                    nb_filters=params['filters'], 
                    kernel_size=params['kernel_size'], 
                    dropout_rate=0.1),
                Dense(1)
            ])
        except ImportError:
            # Fallback to basic dilated convolution implementation
            print("  TCN package not found, using simplified dilated CNN...")
            model = Sequential([
                Conv1D(filters=params['filters'], kernel_size=params['kernel_size'], 
                       activation='relu', input_shape=(params['lookback'], 1), 
                       padding='causal', dilation_rate=1),
                Conv1D(filters=params['filters'], kernel_size=params['kernel_size'], 
                       activation='relu', padding='causal', dilation_rate=2),
                Conv1D(filters=params['filters'], kernel_size=params['kernel_size'], 
                       activation='relu', padding='causal', dilation_rate=4),
                tf.keras.layers.GlobalAveragePooling1D(),
                Dense(50, activation='relu'),
                tf.keras.layers.Dropout(0.1),
                Dense(1)
            ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Train model
        model.fit(X, y, epochs=params['epochs'], batch_size=params['batch_size'], 
                 verbose=0, validation_split=0.1)
        
        return model
    
    def train_seq2seq_model(self, train_data: pd.DataFrame, params: Dict):
        """Train Seq2Seq with attention model"""
        deaths_series = train_data['Deaths'].values.astype(float)
        
        # Scale data
        scaler = MinMaxScaler()
        scaled_deaths = scaler.fit_transform(deaths_series.reshape(-1, 1)).flatten()
        
        # Create sequences
        X, y = self.create_sequences(scaled_deaths, params['lookback'])
        X = X.reshape((X.shape[0], params['lookback'], 1))
        
        # Build seq2seq model
        encoder_inputs = Input(shape=(params['lookback'], 1))
        encoder_gru = GRU(params['encoder_units'], return_sequences=True, return_state=True)
        encoder_outputs, encoder_state = encoder_gru(encoder_inputs)
        
        decoder_inputs = Input(shape=(1, 1))
        decoder_gru = GRU(params['decoder_units'], return_sequences=True, return_state=True)
        decoder_outputs, _ = decoder_gru(decoder_inputs, initial_state=encoder_state)
        
        # Attention mechanism (simplified)
        attention = tf.keras.layers.Attention()
        context_vector = attention([decoder_outputs, encoder_outputs])
        
        # Combine context and decoder output
        decoder_combined = Concatenate(axis=-1)([decoder_outputs, context_vector])
        decoder_dense = Dense(params['decoder_units'], activation='relu')(decoder_combined)
        outputs = Dense(1)(decoder_dense)
        
        model = Model([encoder_inputs, decoder_inputs], outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        # Prepare training data
        decoder_input_data = np.zeros((X.shape[0], 1, 1))
        y_reshaped = y.reshape(-1, 1, 1)
        
        # Train model
        model.fit([X, decoder_input_data], y_reshaped, 
                 epochs=params['epochs'], batch_size=params['batch_size'], 
                 verbose=0, validation_split=0.1)
        
        return model, scaler
    
    def train_transformer_model(self, train_data: pd.DataFrame, params: Dict):
        """Train Transformer model"""
        deaths_series = train_data['Deaths'].values.astype(float)
        
        # Scale data
        scaler = MinMaxScaler()
        scaled_deaths = scaler.fit_transform(deaths_series.reshape(-1, 1)).flatten()
        
        # Create sequences
        X, y = self.create_sequences(scaled_deaths, params['lookback'])
        X = X.reshape((X.shape[0], params['lookback'], 1))
        
        # Build transformer model
        inputs = Input(shape=(params['lookback'], 1))
        x = Dense(params['d_model'])(inputs)
        
        # Positional encoding (simplified)
        positions = tf.range(start=0, limit=params['lookback'], delta=1)
        positions = tf.cast(positions, tf.float32)
        position_encoding = tf.expand_dims(positions, -1) / 10000.0
        x = x + position_encoding
        
        # Multi-head attention
        attn_output = MultiHeadAttention(num_heads=params['n_heads'], 
                                        key_dim=params['d_model'])(x, x)
        x = Add()([x, attn_output])
        x = LayerNormalization()(x)
        
        # Feed forward
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = Dense(64, activation='relu')(x)
        outputs = Dense(1)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Train model
        model.fit(X, y, epochs=params['epochs'], batch_size=params['batch_size'], 
                 verbose=0, validation_split=0.1)
        
        return model, scaler
    
    def generate_predictions(self, model, model_type: str, train_data: pd.DataFrame, 
                           test_data: pd.DataFrame, params: Dict, scaler=None):
        """Generate predictions for different model types"""
        
        train_deaths = train_data['Deaths'].values.astype(float)
        test_deaths = test_data['Deaths'].values.astype(float)
        
        if model_type == 'sarima':
            # SARIMA predictions
            train_pred = model.fittedvalues
            test_pred = model.predict(start=len(train_deaths), 
                                    end=len(train_deaths) + len(test_deaths) - 1)
            
            return train_deaths, train_pred, test_deaths, test_pred
        
        elif model_type in ['lstm', 'tcn']:
            # LSTM/TCN predictions
            lookback = params['lookback']
            
            # Training predictions (in-sample)
            train_pred = []
            for i in range(lookback, len(train_deaths)):
                input_seq = train_deaths[i-lookback:i].reshape(1, lookback, 1)
                pred = model.predict(input_seq, verbose=0)[0][0]
                train_pred.append(pred)
            
            # Test predictions (out-of-sample, autoregressive)
            test_pred = []
            current_seq = train_deaths[-lookback:].copy()
            
            for _ in range(len(test_deaths)):
                input_seq = current_seq.reshape(1, lookback, 1)
                pred = model.predict(input_seq, verbose=0)[0][0]
                test_pred.append(pred)
                
                # Update sequence
                current_seq = np.append(current_seq[1:], pred)
            
            return (train_deaths[lookback:], np.array(train_pred), 
                   test_deaths, np.array(test_pred))
        
        elif model_type in ['seq2seq_attn', 'transformer']:
            # Seq2Seq/Transformer predictions with scaling
            lookback = params['lookback']
            
            # Scale data if scaler provided
            if scaler is not None:
                train_scaled = scaler.transform(train_deaths.reshape(-1, 1)).flatten()
                test_scaled = scaler.transform(test_deaths.reshape(-1, 1)).flatten()
            else:
                train_scaled = train_deaths
                test_scaled = test_deaths
            
            # Generate predictions based on model type
            if model_type == 'seq2seq_attn':
                # Training predictions
                train_pred_scaled = []
                for i in range(lookback, len(train_scaled)):
                    encoder_input = train_scaled[i-lookback:i].reshape(1, lookback, 1)
                    decoder_input = np.zeros((1, 1, 1))
                    pred_scaled = model.predict([encoder_input, decoder_input], verbose=0)[0, 0, 0]
                    train_pred_scaled.append(pred_scaled)
                
                # Test predictions
                test_pred_scaled = []
                current_seq = train_scaled[-lookback:].copy()
                
                for _ in range(len(test_scaled)):
                    encoder_input = current_seq.reshape(1, lookback, 1)
                    decoder_input = np.zeros((1, 1, 1))
                    pred_scaled = model.predict([encoder_input, decoder_input], verbose=0)[0, 0, 0]
                    test_pred_scaled.append(pred_scaled)
                    current_seq = np.append(current_seq[1:], pred_scaled)
            
            else:  # transformer
                # Training predictions
                train_pred_scaled = []
                for i in range(lookback, len(train_scaled)):
                    input_seq = train_scaled[i-lookback:i].reshape(1, lookback, 1)
                    pred_scaled = model.predict(input_seq, verbose=0)[0][0]
                    train_pred_scaled.append(pred_scaled)
                
                # Test predictions
                test_pred_scaled = []
                current_seq = train_scaled[-lookback:].copy()
                
                for _ in range(len(test_scaled)):
                    input_seq = current_seq.reshape(1, lookback, 1)
                    pred_scaled = model.predict(input_seq, verbose=0)[0][0]
                    test_pred_scaled.append(pred_scaled)
                    current_seq = np.append(current_seq[1:], pred_scaled)
            
            # Inverse transform if scaler used
            if scaler is not None:
                train_pred = scaler.inverse_transform(
                    np.array(train_pred_scaled).reshape(-1, 1)).flatten()
                test_pred = scaler.inverse_transform(
                    np.array(test_pred_scaled).reshape(-1, 1)).flatten()
            else:
                train_pred = np.array(train_pred_scaled)
                test_pred = np.array(test_pred_scaled)
            
            return (train_deaths[lookback:], train_pred, test_deaths, test_pred)
    
    def run_single_trial(self, model_name: str, train_data: pd.DataFrame, 
                        test_data: pd.DataFrame, trial_num: int = 1):
        """Run a single trial for a given model"""
        
        print(f"  Trial {trial_num}/{TRIALS_PER_MODEL}")
        
        params = OPTIMAL_PARAMS[model_name]
        
        # Set seed for this trial (consistent but different for each trial)
        trial_seed = RANDOM_SEED + trial_num
        np.random.seed(trial_seed)
        tf.random.set_seed(trial_seed)
        
        try:
            if model_name == 'sarima':
                model = self.train_sarima_model(train_data, params)
                if model is None:
                    return None
                
                train_true, train_pred, test_true, test_pred = self.generate_predictions(
                    model, 'sarima', train_data, test_data, params)
                
                model_obj = model
                
            elif model_name == 'lstm':
                model = self.train_lstm_model(train_data, params)
                train_true, train_pred, test_true, test_pred = self.generate_predictions(
                    model, 'lstm', train_data, test_data, params)
                
                model_obj = model
                
            elif model_name == 'tcn':
                model = self.train_tcn_model(train_data, params)
                train_true, train_pred, test_true, test_pred = self.generate_predictions(
                    model, 'tcn', train_data, test_data, params)
                
                model_obj = model
                
            elif model_name == 'seq2seq_attn':
                model, scaler = self.train_seq2seq_model(train_data, params)
                train_true, train_pred, test_true, test_pred = self.generate_predictions(
                    model, 'seq2seq_attn', train_data, test_data, params, scaler)
                
                model_obj = {'model': model, 'scaler': scaler}
                
            elif model_name == 'transformer':
                model, scaler = self.train_transformer_model(train_data, params)
                train_true, train_pred, test_true, test_pred = self.generate_predictions(
                    model, 'transformer', train_data, test_data, params, scaler)
                
                model_obj = {'model': model, 'scaler': scaler}
            
            # Calculate comprehensive metrics
            train_metrics = self.calculate_comprehensive_metrics(train_true, train_pred)
            test_metrics = self.calculate_comprehensive_metrics(test_true, test_pred)
            
            results = {
                'trial': trial_num,
                'model_name': model_name,
                'train_true': train_true,
                'train_pred': train_pred,
                'test_true': test_true,
                'test_pred': test_pred,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'model_object': model_obj if trial_num == 1 else None  # Save only first model
            }
            
            return results
            
        except Exception as e:
            print(f"    Error in trial {trial_num}: {e}")
            return None
    
    def experiment_1_excess_mortality(self, data_splits: Dict, models: List[str]):
        """
        Experiment 1: Excess mortality estimation (2015-2019 train, 2020-2023 test)
        """
        print("\n" + "="*80)
        print("EXPERIMENT 1: EXCESS MORTALITY ESTIMATION")
        print("="*80)
        print("Training: 2015-2019, Testing: 2020-2023")
        
        split = data_splits['experiment_1']
        train_data = split['train']
        test_data = split['test']
        
        print(f"Train data: {len(train_data)} months")
        print(f"Test data: {len(test_data)} months")
        
        all_results = {}
        
        for model_name in models:
            print(f"\nTraining {model_name.upper()}...")
            model_results = []
            
            for trial in range(1, TRIALS_PER_MODEL + 1):
                result = self.run_single_trial(model_name, train_data, test_data, trial)
                
                if result is not None:
                    model_results.append(result)
                    
                    # Save best model from first trial
                    if trial == 1 and result['model_object'] is not None:
                        model_save_path = f"{self.results_dir}/trained_models/{model_name}_best_model.pkl"
                        with open(model_save_path, 'wb') as f:
                            pickle.dump(result['model_object'], f)
                        print(f"    Saved model to: {model_save_path}")
            
            all_results[model_name] = model_results
            print(f"  Completed {len(model_results)}/{TRIALS_PER_MODEL} trials")
        
        # Save results
        experiment_1_data = {
            'results': all_results,
            'data_splits': {
                'train_data': train_data,
                'test_data': test_data
            },
            'parameters': OPTIMAL_PARAMS,
            'trials_per_model': TRIALS_PER_MODEL
        }
        
        results_path = f"{self.results_dir}/experiment_1_excess_mortality/results.pkl"
        with open(results_path, 'wb') as f:
            pickle.dump(experiment_1_data, f)
        
        # Generate summary statistics
        self.analyze_experiment_1_results(all_results, train_data, test_data)
        
        return all_results
    
    def experiment_2_variance_analysis(self, data_splits: Dict, models: List[str]):
        """
        Experiment 2: Variance analysis across different forecast horizons
        """
        print("\n" + "="*80)
        print("EXPERIMENT 2: VARIANCE ANALYSIS ACROSS FORECAST HORIZONS")
        print("="*80)
        
        horizon_results = {}
        
        for horizon_name in ['horizon_1', 'horizon_2', 'horizon_3', 'horizon_4']:
            if horizon_name not in data_splits:
                continue
                
            split = data_splits[horizon_name]
            train_data = split['train']
            test_data = split['test']
            
            print(f"\n{split['description']}")
            print(f"Test length: {split['test_length_months']} months")
            
            horizon_model_results = {}
            
            for model_name in models:
                print(f"  {model_name.upper()}...")
                model_results = []
                
                for trial in range(1, TRIALS_PER_MODEL + 1):
                    result = self.run_single_trial(model_name, train_data, test_data, trial)
                    
                    if result is not None:
                        model_results.append(result)
                
                horizon_model_results[model_name] = model_results
                print(f"    Completed {len(model_results)}/{TRIALS_PER_MODEL} trials")
            
            horizon_results[horizon_name] = {
                'results': horizon_model_results,
                'split_info': split
            }
        
        # Save results
        experiment_2_data = {
            'horizon_results': horizon_results,
            'parameters': OPTIMAL_PARAMS,
            'trials_per_model': TRIALS_PER_MODEL
        }
        
        results_path = f"{self.results_dir}/experiment_2_variance_analysis/results.pkl"
        with open(results_path, 'wb') as f:
            pickle.dump(experiment_2_data, f)
        
        # Analyze variance across horizons
        self.analyze_variance_across_horizons(horizon_results)
        
        return horizon_results
    
    def analyze_experiment_1_results(self, results: Dict, train_data: pd.DataFrame, test_data: pd.DataFrame):
        """Analyze and visualize Experiment 1 results"""
        print("\nAnalyzing Experiment 1 results...")
        
        # Calculate summary statistics
        summary_stats = []
        
        for model_name, model_results in results.items():
            if not model_results:
                continue
            
            # Extract metrics across all trials
            train_rmse = [r['train_metrics']['RMSE'] for r in model_results]
            train_mae = [r['train_metrics']['MAE'] for r in model_results]
            train_mape = [r['train_metrics']['MAPE'] for r in model_results]
            train_coverage = [r['train_metrics']['coverage'] for r in model_results]
            train_width = [r['train_metrics']['interval_width'] for r in model_results]
            
            test_rmse = [r['test_metrics']['RMSE'] for r in model_results]
            test_mae = [r['test_metrics']['MAE'] for r in model_results]
            test_mape = [r['test_metrics']['MAPE'] for r in model_results]
            test_coverage = [r['test_metrics']['coverage'] for r in model_results]
            test_width = [r['test_metrics']['interval_width'] for r in model_results]
            
            summary_stats.append({
                'Model': model_name.upper(),
                'Train_RMSE_Mean': np.mean(train_rmse),
                'Train_RMSE_Std': np.std(train_rmse),
                'Train_MAE_Mean': np.mean(train_mae),
                'Train_MAE_Std': np.std(train_mae),
                'Train_MAPE_Mean': np.mean(train_mape),
                'Train_MAPE_Std': np.std(train_mape),
                'Train_Coverage_Mean': np.mean(train_coverage),
                'Train_Coverage_Std': np.std(train_coverage),
                'Train_Width_Mean': np.mean(train_width),
                'Train_Width_Std': np.std(train_width),
                'Test_RMSE_Mean': np.mean(test_rmse),
                'Test_RMSE_Std': np.std(test_rmse),
                'Test_MAE_Mean': np.mean(test_mae),
                'Test_MAE_Std': np.std(test_mae),
                'Test_MAPE_Mean': np.mean(test_mape),
                'Test_MAPE_Std': np.std(test_mape),
                'Test_Coverage_Mean': np.mean(test_coverage),
                'Test_Coverage_Std': np.std(test_coverage),
                'Test_Width_Mean': np.mean(test_width),
                'Test_Width_Std': np.std(test_width),
                'N_Trials': len(model_results)
            })
        
        # Save summary statistics
        summary_df = pd.DataFrame(summary_stats)
        summary_path = f"{self.results_dir}/experiment_1_excess_mortality/summary_statistics.csv"
        summary_df.to_csv(summary_path, index=False)
        
        print(f"Summary statistics saved to: {summary_path}")
        
        # Create model comparison plots (similar to sarima_vs_lstm_comparison.png)
        self.create_model_comparison_plots(results, train_data, test_data)
        
        # Export detailed prediction data for later figure editing
        self.export_prediction_data(results, train_data, test_data, 'experiment_1')
    
    def create_model_comparison_plots(self, results: Dict, train_data: pd.DataFrame, test_data: pd.DataFrame):
        """Create individual model comparison plots"""
        
        if 'sarima' not in results:
            print("Warning: SARIMA results not found for comparison plots")
            return
        
        # Get SARIMA results for comparison
        sarima_results = results['sarima']
        if not sarima_results:
            return
        
        # Calculate average SARIMA predictions across trials
        sarima_train_preds = np.mean([r['train_pred'] for r in sarima_results], axis=0)
        sarima_test_preds = np.mean([r['test_pred'] for r in sarima_results], axis=0)
        sarima_train_true = sarima_results[0]['train_true']
        sarima_test_true = sarima_results[0]['test_true']
        
        # Calculate SARIMA prediction intervals
        sarima_train_metrics = self.calculate_comprehensive_metrics(sarima_train_true, sarima_train_preds)
        sarima_test_metrics = self.calculate_comprehensive_metrics(sarima_test_true, sarima_test_preds)
        
        # Create comparison plots for each DL model
        dl_models = [m for m in results.keys() if m != 'sarima']
        
        for model_name in dl_models:
            if model_name not in results or not results[model_name]:
                continue
                
            print(f"Creating comparison plot: SARIMA vs {model_name.upper()}")
            
            # Calculate average predictions for this model
            model_results = results[model_name]
            model_train_preds = np.mean([r['train_pred'] for r in model_results], axis=0)
            model_test_preds = np.mean([r['test_pred'] for r in model_results], axis=0)
            model_train_true = model_results[0]['train_true']
            model_test_true = model_results[0]['test_true']
            
            # Calculate model prediction intervals
            model_train_metrics = self.calculate_comprehensive_metrics(model_train_true, model_train_preds)
            model_test_metrics = self.calculate_comprehensive_metrics(model_test_true, model_test_preds)
            
            # Prepare data for plotting
            if model_name != 'sarima':
                # For DL models, account for lookback period
                lookback = OPTIMAL_PARAMS[model_name].get('lookback', 0)
                sarima_train_dates = train_data['Date'].values
                model_train_dates = train_data['Date'].iloc[lookback:].values
                
                # Align SARIMA with model training period for comparison
                sarima_aligned_train_preds = sarima_train_preds[lookback:]
                sarima_aligned_train_lower = sarima_train_metrics['lower_bound'][lookback:]
                sarima_aligned_train_upper = sarima_train_metrics['upper_bound'][lookback:]
            else:
                model_train_dates = train_data['Date'].values
                sarima_aligned_train_preds = sarima_train_preds
                sarima_aligned_train_lower = sarima_train_metrics['lower_bound']
                sarima_aligned_train_upper = sarima_train_metrics['upper_bound']
            
            # Combine training and test data for full timeline
            all_dates = np.concatenate([model_train_dates, test_data['Date'].values])
            all_actual = np.concatenate([model_train_true, model_test_true])
            all_sarima_preds = np.concatenate([sarima_aligned_train_preds, sarima_test_preds])
            all_model_preds = np.concatenate([model_train_preds, model_test_preds])
            
            # Combine prediction intervals
            all_sarima_lower = np.concatenate([sarima_aligned_train_lower, sarima_test_metrics['lower_bound']])
            all_sarima_upper = np.concatenate([sarima_aligned_train_upper, sarima_test_metrics['upper_bound']])
            all_model_lower = np.concatenate([model_train_metrics['lower_bound'], model_test_metrics['lower_bound']])
            all_model_upper = np.concatenate([model_train_metrics['upper_bound'], model_test_metrics['upper_bound']])
            
            # Create the plot
            plt.figure(figsize=(16, 10))
            
            # Plot actual data
            plt.plot(all_dates, all_actual, 'k-', linewidth=3, label='Actual Deaths', zorder=5)
            
            # Plot SARIMA predictions and intervals
            plt.plot(all_dates, all_sarima_preds, '--', color='blue', linewidth=2, 
                    label='SARIMA Predictions', alpha=0.8)
            plt.fill_between(all_dates, all_sarima_lower, all_sarima_upper,
                           color='blue', alpha=0.25, label='SARIMA 95% PI')
            
            # Plot model predictions and intervals
            plt.plot(all_dates, all_model_preds, '-', color='red', linewidth=2,
                    label=f'{model_name.upper()} Predictions', alpha=0.8)
            plt.fill_between(all_dates, all_model_lower, all_model_upper,
                           color='red', alpha=0.25, label=f'{model_name.upper()} 95% PI')
            
            # Add forecast start line
            forecast_start = test_data['Date'].iloc[0]
            plt.axvline(forecast_start, color='green', linestyle='--', linewidth=2, 
                       alpha=0.7, label='Forecast Start (Jan 2020)')
            
            # Formatting for publication readiness
            plt.xlabel('Date', fontsize=16, fontweight='bold')
            plt.ylabel('Deaths', fontsize=16, fontweight='bold')
            plt.title(f'Mortality Forecasting: SARIMA vs {model_name.upper()}', 
                     fontsize=20, fontweight='bold', pad=20)
            
            # Increase tick label sizes
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            
            # Legend with larger font
            plt.legend(loc='upper left', fontsize=14, frameon=True, fancybox=True, shadow=True)
            
            # Grid
            plt.grid(True, alpha=0.3)
            
            # Add performance metrics text box
            sarima_test_rmse = np.mean([r['test_metrics']['RMSE'] for r in sarima_results])
            model_test_rmse = np.mean([r['test_metrics']['RMSE'] for r in model_results])
            
            metrics_text = f'Test RMSE:\nSARIMA: {sarima_test_rmse:.0f}\n{model_name.upper()}: {model_test_rmse:.0f}'
            plt.text(0.02, 0.98, metrics_text, transform=plt.gca().transAxes, fontsize=12,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = f"{self.results_dir}/figures/sarima_vs_{model_name}_comparison.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"  Saved: {plot_path}")
    
    def export_prediction_data(self, results: Dict, train_data: pd.DataFrame, 
                              test_data: pd.DataFrame, experiment_name: str):
        """Export detailed prediction data for later figure editing"""
        
        print("Exporting prediction data for later figure editing...")
        
        export_data = {}
        
        for model_name, model_results in results.items():
            if not model_results:
                continue
            
            # Calculate average predictions across trials
            avg_train_pred = np.mean([r['train_pred'] for r in model_results], axis=0)
            avg_test_pred = np.mean([r['test_pred'] for r in model_results], axis=0)
            std_train_pred = np.std([r['train_pred'] for r in model_results], axis=0)
            std_test_pred = np.std([r['test_pred'] for r in model_results], axis=0)
            
            train_true = model_results[0]['train_true']
            test_true = model_results[0]['test_true']
            
            # Calculate prediction intervals for average predictions
            train_metrics = self.calculate_comprehensive_metrics(train_true, avg_train_pred)
            test_metrics = self.calculate_comprehensive_metrics(test_true, avg_test_pred)
            
            # Prepare dates
            if model_name != 'sarima' and 'lookback' in OPTIMAL_PARAMS[model_name]:
                lookback = OPTIMAL_PARAMS[model_name]['lookback']
                train_dates = train_data['Date'].iloc[lookback:].values
            else:
                train_dates = train_data['Date'].values
                
            test_dates = test_data['Date'].values
            
            export_data[model_name] = {
                'train': {
                    'dates': train_dates,
                    'actual': train_true,
                    'predicted_mean': avg_train_pred,
                    'predicted_std': std_train_pred,
                    'lower_bound': train_metrics['lower_bound'],
                    'upper_bound': train_metrics['upper_bound'],
                    'coverage': train_metrics['coverage'],
                    'interval_width': train_metrics['interval_width']
                },
                'test': {
                    'dates': test_dates,
                    'actual': test_true,
                    'predicted_mean': avg_test_pred,
                    'predicted_std': std_test_pred,
                    'lower_bound': test_metrics['lower_bound'],
                    'upper_bound': test_metrics['upper_bound'],
                    'coverage': test_metrics['coverage'],
                    'interval_width': test_metrics['interval_width']
                }
            }
        
        # Save export data
        export_path = f"{self.results_dir}/data_exports/{experiment_name}_prediction_data.pkl"
        with open(export_path, 'wb') as f:
            pickle.dump(export_data, f)
        
        # Also save as CSV for easy access
        csv_data = []
        for model_name, model_data in export_data.items():
            # Training data
            for i, date in enumerate(model_data['train']['dates']):
                csv_data.append({
                    'Model': model_name.upper(),
                    'Period': 'Train',
                    'Date': date,
                    'Actual': model_data['train']['actual'][i],
                    'Predicted_Mean': model_data['train']['predicted_mean'][i],
                    'Predicted_Std': model_data['train']['predicted_std'][i],
                    'Lower_Bound': model_data['train']['lower_bound'][i],
                    'Upper_Bound': model_data['train']['upper_bound'][i]
                })
            
            # Test data
            for i, date in enumerate(model_data['test']['dates']):
                csv_data.append({
                    'Model': model_name.upper(),
                    'Period': 'Test',
                    'Date': date,
                    'Actual': model_data['test']['actual'][i],
                    'Predicted_Mean': model_data['test']['predicted_mean'][i],
                    'Predicted_Std': model_data['test']['predicted_std'][i],
                    'Lower_Bound': model_data['test']['lower_bound'][i],
                    'Upper_Bound': model_data['test']['upper_bound'][i]
                })
        
        csv_df = pd.DataFrame(csv_data)
        csv_path = f"{self.results_dir}/data_exports/{experiment_name}_prediction_data.csv"
        csv_df.to_csv(csv_path, index=False)
        
        print(f"Prediction data exported to:")
        print(f"  - {export_path}")
        print(f"  - {csv_path}")
    
    def analyze_variance_across_horizons(self, horizon_results: Dict):
        """Analyze variance reduction across different forecast horizons"""
        print("\nAnalyzing variance across forecast horizons...")
        
        # Organize data by horizon
        horizon_summary = []
        
        for horizon_name, horizon_data in horizon_results.items():
            split_info = horizon_data['split_info']
            results = horizon_data['results']
            
            horizon_months = split_info['test_length_months']
            
            for model_name, model_results in results.items():
                if not model_results:
                    continue
                
                # Extract metrics across trials
                test_rmse = [r['test_metrics']['RMSE'] for r in model_results]
                test_mae = [r['test_metrics']['MAE'] for r in model_results]
                test_mape = [r['test_metrics']['MAPE'] for r in model_results]
                test_coverage = [r['test_metrics']['coverage'] for r in model_results]
                test_width = [r['test_metrics']['interval_width'] for r in model_results]
                
                horizon_summary.append({
                    'Horizon': horizon_name,
                    'Forecast_Horizon_Months': horizon_months,
                    'Model': model_name.upper(),
                    'Test_RMSE_Mean': np.mean(test_rmse),
                    'Test_RMSE_Std': np.std(test_rmse),
                    'Test_MAE_Mean': np.mean(test_mae),
                    'Test_MAE_Std': np.std(test_mae),
                    'Test_MAPE_Mean': np.mean(test_mape),
                    'Test_MAPE_Std': np.std(test_mape),
                    'Coverage_Mean': np.mean(test_coverage),
                    'Coverage_Std': np.std(test_coverage),
                    'Interval_Width_Mean': np.mean(test_width),
                    'Interval_Width_Std': np.std(test_width),
                    'N_Trials': len(model_results)
                })
        
        # Save horizon summary
        horizon_df = pd.DataFrame(horizon_summary)
        horizon_path = f"{self.results_dir}/experiment_2_variance_analysis/horizon_summary.csv"
        horizon_df.to_csv(horizon_path, index=False)
        
        print(f"Horizon analysis saved to: {horizon_path}")
        
        # Create variance analysis plots
        self.create_variance_analysis_plots(horizon_df)
        
        # Export horizon prediction data
        self.export_horizon_prediction_data(horizon_results)
    
    def create_variance_analysis_plots(self, horizon_df: pd.DataFrame):
        """Create plots showing variance across forecast horizons"""
        
        print("Creating variance analysis plots...")
        
        # Get unique models and horizons
        models = horizon_df['Model'].unique()
        horizons = sorted(horizon_df['Forecast_Horizon_Months'].unique())
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Variance Analysis Across Forecast Horizons', fontsize=20, fontweight='bold')
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
        
        # Plot 1: RMSE vs Horizon
        ax = axes[0, 0]
        for i, model in enumerate(models):
            model_data = horizon_df[horizon_df['Model'] == model]
            rmse_means = [model_data[model_data['Forecast_Horizon_Months'] == h]['Test_RMSE_Mean'].iloc[0] 
                         for h in horizons if len(model_data[model_data['Forecast_Horizon_Months'] == h]) > 0]
            rmse_stds = [model_data[model_data['Forecast_Horizon_Months'] == h]['Test_RMSE_Std'].iloc[0] 
                        for h in horizons if len(model_data[model_data['Forecast_Horizon_Months'] == h]) > 0]
            
            if rmse_means:
                ax.errorbar(horizons[:len(rmse_means)], rmse_means, yerr=rmse_stds, 
                           label=model, marker='o', color=colors[i], linewidth=2, markersize=8)
        
        ax.set_xlabel('Forecast Horizon (Months)', fontsize=14, fontweight='bold')
        ax.set_ylabel('RMSE', fontsize=14, fontweight='bold')
        ax.set_title('RMSE vs Forecast Horizon', fontsize=16, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=12)
        
        # Plot 2: Coverage vs Horizon
        ax = axes[0, 1]
        for i, model in enumerate(models):
            model_data = horizon_df[horizon_df['Model'] == model]
            coverage_means = [model_data[model_data['Forecast_Horizon_Months'] == h]['Coverage_Mean'].iloc[0] 
                            for h in horizons if len(model_data[model_data['Forecast_Horizon_Months'] == h]) > 0]
            coverage_stds = [model_data[model_data['Forecast_Horizon_Months'] == h]['Coverage_Std'].iloc[0] 
                           for h in horizons if len(model_data[model_data['Forecast_Horizon_Months'] == h]) > 0]
            
            if coverage_means:
                ax.errorbar(horizons[:len(coverage_means)], coverage_means, yerr=coverage_stds, 
                           label=model, marker='s', color=colors[i], linewidth=2, markersize=8)
        
        ax.axhline(y=95, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Target 95%')
        ax.set_xlabel('Forecast Horizon (Months)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Coverage (%)', fontsize=14, fontweight='bold')
        ax.set_title('PI Coverage vs Forecast Horizon', fontsize=16, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=12)
        
        # Plot 3: Interval Width vs Horizon
        ax = axes[1, 0]
        for i, model in enumerate(models):
            model_data = horizon_df[horizon_df['Model'] == model]
            width_means = [model_data[model_data['Forecast_Horizon_Months'] == h]['Interval_Width_Mean'].iloc[0] 
                          for h in horizons if len(model_data[model_data['Forecast_Horizon_Months'] == h]) > 0]
            width_stds = [model_data[model_data['Forecast_Horizon_Months'] == h]['Interval_Width_Std'].iloc[0] 
                         for h in horizons if len(model_data[model_data['Forecast_Horizon_Months'] == h]) > 0]
            
            if width_means:
                ax.errorbar(horizons[:len(width_means)], width_means, yerr=width_stds, 
                           label=model, marker='^', color=colors[i], linewidth=2, markersize=8)
        
        ax.set_xlabel('Forecast Horizon (Months)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Average PI Width', fontsize=14, fontweight='bold')
        ax.set_title('PI Width vs Forecast Horizon', fontsize=16, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=12)
        
        # Plot 4: MAPE vs Horizon
        ax = axes[1, 1]
        for i, model in enumerate(models):
            model_data = horizon_df[horizon_df['Model'] == model]
            mape_means = [model_data[model_data['Forecast_Horizon_Months'] == h]['Test_MAPE_Mean'].iloc[0] 
                         for h in horizons if len(model_data[model_data['Forecast_Horizon_Months'] == h]) > 0]
            mape_stds = [model_data[model_data['Forecast_Horizon_Months'] == h]['Test_MAPE_Std'].iloc[0] 
                        for h in horizons if len(model_data[model_data['Forecast_Horizon_Months'] == h]) > 0]
            
            if mape_means:
                ax.errorbar(horizons[:len(mape_means)], mape_means, yerr=mape_stds, 
                           label=model, marker='D', color=colors[i], linewidth=2, markersize=8)
        
        ax.set_xlabel('Forecast Horizon (Months)', fontsize=14, fontweight='bold')
        ax.set_ylabel('MAPE (%)', fontsize=14, fontweight='bold')
        ax.set_title('MAPE vs Forecast Horizon', fontsize=16, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=12)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = f"{self.results_dir}/figures/variance_analysis_across_horizons.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Variance analysis plot saved to: {plot_path}")
    
    def export_horizon_prediction_data(self, horizon_results: Dict):
        """Export prediction data for all horizons"""
        
        print("Exporting horizon prediction data...")
        
        all_horizon_data = {}
        
        for horizon_name, horizon_data in horizon_results.items():
            results = horizon_data['results']
            split_info = horizon_data['split_info']
            
            horizon_export = {}
            
            for model_name, model_results in results.items():
                if not model_results:
                    continue
                
                # Calculate average predictions
                avg_test_pred = np.mean([r['test_pred'] for r in model_results], axis=0)
                std_test_pred = np.std([r['test_pred'] for r in model_results], axis=0)
                test_true = model_results[0]['test_true']
                
                # Calculate metrics
                test_metrics = self.calculate_comprehensive_metrics(test_true, avg_test_pred)
                
                horizon_export[model_name] = {
                    'test_dates': split_info['train']['Date'].iloc[-1:].tolist() + split_info['test']['Date'].tolist()[:-1],
                    'actual': test_true,
                    'predicted_mean': avg_test_pred,
                    'predicted_std': std_test_pred,
                    'lower_bound': test_metrics['lower_bound'],
                    'upper_bound': test_metrics['upper_bound'],
                    'coverage': test_metrics['coverage'],
                    'interval_width': test_metrics['interval_width'],
                    'horizon_months': split_info['test_length_months']
                }
            
            all_horizon_data[horizon_name] = horizon_export
        
        # Save horizon data
        horizon_export_path = f"{self.results_dir}/data_exports/horizon_prediction_data.pkl"
        with open(horizon_export_path, 'wb') as f:
            pickle.dump(all_horizon_data, f)
        
        print(f"Horizon prediction data exported to: {horizon_export_path}")
    
    def generate_final_report(self):
        """Generate comprehensive final report"""
        
        print("\n" + "="*80)
        print("GENERATING FINAL REPORT")
        print("="*80)
        
        report_content = []
        report_content.append("STREAMLINED EVALUATION RESULTS SUMMARY")
        report_content.append("=" * 50)
        report_content.append(f"Execution Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_content.append(f"Random Seed: {RANDOM_SEED}")
        report_content.append(f"Trials per Model: {TRIALS_PER_MODEL}")
        report_content.append("")
        
        # Model parameters
        report_content.append("OPTIMAL HYPERPARAMETERS USED:")
        report_content.append("-" * 30)
        for model, params in OPTIMAL_PARAMS.items():
            report_content.append(f"{model.upper()}: {params}")
        report_content.append("")
        
        # Generated files
        report_content.append("GENERATED FILES:")
        report_content.append("-" * 20)
        
        for root, dirs, files in os.walk(self.results_dir):
            level = root.replace(self.results_dir, '').count(os.sep)
            indent = '  ' * level
            report_content.append(f"{indent}{os.path.basename(root)}/")
            
            subindent = '  ' * (level + 1)
            for file in sorted(files):
                if not file.startswith('.'):
                    report_content.append(f"{subindent}{file}")
        
        report_content.append("")
        report_content.append("KEY RESULTS:")
        report_content.append("-" * 15)
        
        # Load and summarize key results
        try:
            summary_path = f"{self.results_dir}/experiment_1_excess_mortality/summary_statistics.csv"
            if os.path.exists(summary_path):
                summary_df = pd.read_csv(summary_path)
                
                report_content.append("Experiment 1 - Test Set Performance (2020-2023):")
                for _, row in summary_df.iterrows():
                    report_content.append(f"  {row['Model']}:")
                    report_content.append(f"    RMSE: {row['Test_RMSE_Mean']:.2f} ± {row['Test_RMSE_Std']:.2f}")
                    report_content.append(f"    MAPE: {row['Test_MAPE_Mean']:.2f}% ± {row['Test_MAPE_Std']:.2f}%")
                    report_content.append(f"    Coverage: {row['Test_Coverage_Mean']:.1f}% ± {row['Test_Coverage_Std']:.1f}%")
                
                # Find best model
                best_model_idx = summary_df['Test_RMSE_Mean'].idxmin()
                best_model = summary_df.iloc[best_model_idx]
                report_content.append("")
                report_content.append(f"BEST PERFORMING MODEL: {best_model['Model']}")
                report_content.append(f"  Test RMSE: {best_model['Test_RMSE_Mean']:.2f}")
                report_content.append(f"  Test MAPE: {best_model['Test_MAPE_Mean']:.2f}%")
        
        except Exception as e:
            report_content.append(f"Error loading summary statistics: {e}")
        
        report_content.append("")
        report_content.append("NEXT STEPS:")
        report_content.append("-" * 15)
        report_content.append("1. Review figures in the 'figures' folder")
        report_content.append("2. Examine CSV files in 'data_exports' for detailed analysis")
        report_content.append("3. Use trained models from 'trained_models' for dashboard")
        report_content.append("4. Edit figure styling using exported prediction data")
        
        # Save report
        report_path = f"{self.results_dir}/EVALUATION_SUMMARY_REPORT.txt"
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_content))
        
        print(f"Final report saved to: {report_path}")
        
        # Print report to console
        print("\n" + "\n".join(report_content))


def main():
    """Main execution function"""
    print("=" * 80)
    print("STREAMLINED EVALUATION PIPELINE")
    print("Advanced Machine Learning for Substance Overdose Mortality Prediction")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  - Random seed: {RANDOM_SEED}")
    print(f"  - Trials per model: {TRIALS_PER_MODEL}")
    print(f"  - Results directory: {RESULTS_DIR}")
    print(f"  - Data path: {DATA_PATH}")
    
    # Check if data file exists
    if not os.path.exists(DATA_PATH):
        print(f"\nError: Data file not found at {DATA_PATH}")
        print("Please ensure the Excel file is in the correct location.")
        return False
    
    start_time = time.time()
    
    # Initialize pipeline
    pipeline = StreamlinedEvaluationPipeline(DATA_PATH, RESULTS_DIR)
    
    try:
        # Step 1: Load and preprocess data
        print("\n" + "="*60)
        print("STEP 1: LOADING AND PREPROCESSING DATA")
        print("="*60)
        
        df = pipeline.load_and_preprocess_data()
        
        # Step 2: Create data splits
        print("\n" + "="*60)
        print("STEP 2: CREATING DATA SPLITS")
        print("="*60)
        
        data_splits = pipeline.create_data_splits(df)
        
        print("Data splits created:")
        for split_name, split_info in data_splits.items():
            print(f"  {split_name}: {split_info['description']}")
            if 'test_length_months' in split_info:
                print(f"    Test length: {split_info['test_length_months']} months")
        
        # Define models to evaluate
        models = ['sarima', 'lstm', 'tcn', 'seq2seq_attn', 'transformer']
        print(f"\nModels to evaluate: {models}")
        
        # Step 3: Run Experiment 1
        print("\n" + "="*60)
        print("STEP 3: EXPERIMENT 1 - EXCESS MORTALITY ESTIMATION")
        print("="*60)
        
        exp1_results = pipeline.experiment_1_excess_mortality(data_splits, models)
        
        # Step 4: Run Experiment 2
        print("\n" + "="*60)
        print("STEP 4: EXPERIMENT 2 - VARIANCE ANALYSIS")
        print("="*60)
        
        exp2_results = pipeline.experiment_2_variance_analysis(data_splits, models)
        
        # Step 5: Generate final report
        pipeline.generate_final_report()
        
        # Execution summary
        total_time = time.time() - start_time
        
        print("\n" + "🎉" * 20)
        print("STREAMLINED EVALUATION COMPLETED SUCCESSFULLY!")
        print("🎉" * 20)
        
        print(f"\nExecution Summary:")
        print(f"  Total time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        print(f"  Models evaluated: {len(models)}")
        print(f"  Trials per model: {TRIALS_PER_MODEL}")
        print(f"  Random seed: {RANDOM_SEED}")
        
        print(f"\n📁 Results saved to: {RESULTS_DIR}")
        print("\n📊 Key outputs:")
        print("  - Model comparison figures: figures/sarima_vs_[model]_comparison.png")
        print("  - Variance analysis plot: figures/variance_analysis_across_horizons.png")
        print("  - Summary statistics: experiment_1_excess_mortality/summary_statistics.csv")
        print("  - Horizon analysis: experiment_2_variance_analysis/horizon_summary.csv")
        print("  - Trained models: trained_models/[model]_best_model.pkl")
        print("  - Prediction data: data_exports/[experiment]_prediction_data.csv")
        
        print("\n🔄 Next steps:")
        print("  1. Review generated figures and adjust styling as needed")
        print("  2. Use exported CSV data for custom visualizations")
        print("  3. Integrate trained models into dashboard backend")
        print("  4. Analyze summary statistics for paper results")
        
        return True
        
    except KeyboardInterrupt:
        print("\n⚠️  Evaluation interrupted by user")
        return False
        
    except Exception as e:
        print(f"\n❌ Evaluation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
