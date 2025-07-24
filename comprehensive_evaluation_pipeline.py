#!/usr/bin/env python3
"""
Comprehensive Evaluation Pipeline for Substance Overdose Mortality Forecasting
================================================================================

This script runs all three experiments:
1. Excess mortality estimation with full trained models (2015-2019 train, 2020-2023 test)
2. Variance reduction analysis over different forecasting horizons
3. Sensitivity analysis for random seeds and trial numbers

Updated for new data format (2015-2023) with proper data structure handling.
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
import itertools
from typing import Dict, List, Tuple, Any
import math

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration
RESULTS_DIR = 'final_eval_results_2015_2023'
DATA_PATH = 'data_updated/state_month_overdose_2015_2023.xlsx'

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
os.makedirs(f'{RESULTS_DIR}/experiment_3_sensitivity', exist_ok=True)
os.makedirs(f'{RESULTS_DIR}/trained_models', exist_ok=True)
os.makedirs(f'{RESULTS_DIR}/figures', exist_ok=True)

class ComprehensiveEvaluationPipeline:
    """Comprehensive evaluation pipeline for all experiments"""
    
    def __init__(self, data_path: str, results_dir: str):
        self.data_path = data_path
        self.results_dir = results_dir
        self.trained_models = {}
        
    def load_and_preprocess_data(self):
        """Load and preprocess the new format overdose dataset (2015-2023)"""
        print("Loading and preprocessing data...")
        
        # Load the Excel file
        df = pd.read_excel(self.data_path)
        
        # Print data structure for debugging
        print("Data columns:", df.columns.tolist())
        print("Data shape:", df.shape)
        print("First few rows:")
        print(df.head())
        
        # Handle the new data format
        # Expected columns: Row Labels, Month, Month_Code, Year_Code, Sum of Deaths
        
        # Create proper datetime from Row Labels or construct from Year_Code and Month_Code
        if 'Row Labels' in df.columns:
            df['Date'] = pd.to_datetime(df['Row Labels'])
        else:
            # Construct date from Year_Code and Month_Code
            df['Date'] = pd.to_datetime(df['Year_Code'].astype(str) + '-' + 
                                      df['Month_Code'].astype(str).str.zfill(2) + '-01')
        
        # Get deaths column (should be 'Sum of Deaths' in new format)
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
    
    def create_train_test_splits(self, df: pd.DataFrame, train_end: str = '2019-12-31', 
                                test_periods: List[str] = None):
        """Create train/test splits for different experimental conditions"""
        
        if test_periods is None:
            test_periods = ['2020-12-31', '2021-12-31', '2022-12-31', '2023-12-31']
        
        splits = {}
        
        # Training data (2015-2019)
        train_data = df[df['Date'] <= train_end].copy()
        
        # Different test periods for variance analysis
        for i, test_end in enumerate(test_periods):
            test_start = '2020-01-01'
            test_data = df[(df['Date'] >= test_start) & (df['Date'] <= test_end)].copy()
            
            splits[f'test_period_{i+1}'] = {
                'train': train_data,
                'test': test_data,
                'test_end': test_end,
                'test_length': len(test_data)
            }
        
        return splits
    
    def create_sequences(self, data: np.ndarray, lookback: int):
        """Create sequences for deep learning models"""
        X, y = [], []
        for i in range(len(data) - lookback):
            X.append(data[i:(i + lookback)])
            y.append(data[i + lookback])
        return np.array(X), np.array(y)
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Calculate comprehensive evaluation metrics"""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1))) * 100
        mse = mean_squared_error(y_true, y_pred)
        
        return {
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'MSE': mse
        }
    
    def calculate_prediction_intervals(self, y_true: np.ndarray, y_pred: np.ndarray, alpha: float = 0.05):
        """Calculate prediction intervals and coverage"""
        residuals = y_true - y_pred
        std_residual = np.std(residuals)
        z_score = 1.96  # 95% confidence interval
        margin_of_error = z_score * std_residual
        
        lower_bound = y_pred - margin_of_error
        upper_bound = y_pred + margin_of_error
        
        # Calculate coverage
        coverage = np.mean((y_true >= lower_bound) & (y_true <= upper_bound)) * 100
        
        # Calculate interval width
        width = np.mean(upper_bound - lower_bound)
        
        return lower_bound, upper_bound, coverage, width
    
    def train_sarima_model(self, train_data: pd.DataFrame, params: Dict, seed: int = 42):
        """Train SARIMA model"""
        np.random.seed(seed)
        
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
    
    def train_lstm_model(self, train_data: pd.DataFrame, params: Dict, seed: int = 42):
        """Train LSTM model"""
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
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
    
    def train_tcn_model(self, train_data: pd.DataFrame, params: Dict, seed: int = 42):
        """Train TCN model (simplified implementation without keras-tcn dependency)"""
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
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
            print("  TCN package not found, using simplified dilated CNN implementation...")
            model = Sequential([
                Conv1D(filters=params['filters'], kernel_size=params['kernel_size'], 
                       activation='relu', input_shape=(params['lookback'], 1), 
                       padding='causal', dilation_rate=1),
                Conv1D(filters=params['filters'], kernel_size=params['kernel_size'], 
                       activation='relu', padding='causal', dilation_rate=2),
                Conv1D(filters=params['filters'], kernel_size=params['kernel_size'], 
                       activation='relu', padding='causal', dilation_rate=4),
                Conv1D(filters=params['filters']//2, kernel_size=1, 
                       activation='relu', padding='causal'),
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
    
    def train_seq2seq_model(self, train_data: pd.DataFrame, params: Dict, seed: int = 42):
        """Train Seq2Seq with attention model"""
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
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
    
    def train_transformer_model(self, train_data: pd.DataFrame, params: Dict, seed: int = 42):
        """Train Transformer model"""
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
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
        
        elif model_type == 'seq2seq_attn':
            # Seq2Seq predictions
            lookback = params['lookback']
            
            # Scale data if scaler provided
            if scaler is not None:
                train_scaled = scaler.transform(train_deaths.reshape(-1, 1)).flatten()
                test_scaled = scaler.transform(test_deaths.reshape(-1, 1)).flatten()
            else:
                train_scaled = train_deaths
                test_scaled = test_deaths
            
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
        
        elif model_type == 'transformer':
            # Transformer predictions (similar to seq2seq)
            lookback = params['lookback']
            
            if scaler is not None:
                train_scaled = scaler.transform(train_deaths.reshape(-1, 1)).flatten()
                test_scaled = scaler.transform(test_deaths.reshape(-1, 1)).flatten()
            else:
                train_scaled = train_deaths
                test_scaled = test_deaths
            
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
                        test_data: pd.DataFrame, seed: int = 42):
        """Run a single trial for a given model"""
        
        params = OPTIMAL_PARAMS[model_name]
        
        try:
            if model_name == 'sarima':
                model = self.train_sarima_model(train_data, params, seed)
                if model is None:
                    return None
                
                train_true, train_pred, test_true, test_pred = self.generate_predictions(
                    model, 'sarima', train_data, test_data, params)
                
                additional_objects = {'model': model}
                
            elif model_name == 'lstm':
                model = self.train_lstm_model(train_data, params, seed)
                train_true, train_pred, test_true, test_pred = self.generate_predictions(
                    model, 'lstm', train_data, test_data, params)
                
                additional_objects = {'model': model}
                
            elif model_name == 'tcn':
                model = self.train_tcn_model(train_data, params, seed)
                train_true, train_pred, test_true, test_pred = self.generate_predictions(
                    model, 'tcn', train_data, test_data, params)
                
                additional_objects = {'model': model}
                
            elif model_name == 'seq2seq_attn':
                model, scaler = self.train_seq2seq_model(train_data, params, seed)
                train_true, train_pred, test_true, test_pred = self.generate_predictions(
                    model, 'seq2seq_attn', train_data, test_data, params, scaler)
                
                additional_objects = {'model': model, 'scaler': scaler}
                
            elif model_name == 'transformer':
                model, scaler = self.train_transformer_model(train_data, params, seed)
                train_true, train_pred, test_true, test_pred = self.generate_predictions(
                    model, 'transformer', train_data, test_data, params, scaler)
                
                additional_objects = {'model': model, 'scaler': scaler}
            
            # Calculate metrics
            train_metrics = self.calculate_metrics(train_true, train_pred)
            test_metrics = self.calculate_metrics(test_true, test_pred)
            
            # Calculate prediction intervals
            train_lower, train_upper, train_coverage, train_width = \
                self.calculate_prediction_intervals(train_true, train_pred)
            test_lower, test_upper, test_coverage, test_width = \
                self.calculate_prediction_intervals(test_true, test_pred)
            
            results = {
                'model_name': model_name,
                'seed': seed,
                'train_true': train_true,
                'train_pred': train_pred,
                'train_lower': train_lower,
                'train_upper': train_upper,
                'test_true': test_true,
                'test_pred': test_pred,
                'test_lower': test_lower,
                'test_upper': test_upper,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'train_coverage': train_coverage,
                'train_width': train_width,
                'test_coverage': test_coverage,
                'test_width': test_width,
                'additional_objects': additional_objects
            }
            
            return results
            
        except Exception as e:
            print(f"Error in trial for {model_name} with seed {seed}: {e}")
            return None
        
    def experiment_1_excess_mortality(self, data_splits: Dict, models: List[str], 
                                    seeds: List[int], trials_per_seed: int = 30):
        """
        Experiment 1: Excess mortality estimation (2015-2019 train, 2020-2023 test)
        """
        print("\n" + "="*80)
        print("EXPERIMENT 1: EXCESS MORTALITY ESTIMATION")
        print("="*80)
        
        # Use the full test period (2020-2023)
        split = data_splits['test_period_4']  # 2020-2023
        train_data = split['train']
        test_data = split['test']
        
        results = {}
        
        for model_name in models:
            print(f"\nTraining {model_name.upper()}...")
            model_results = {}
            
            for seed in seeds:
                print(f"  Seed {seed}...")
                seed_results = []
                
                for trial in range(trials_per_seed):
                    trial_seed = seed + trial * 1000
                    result = self.run_single_trial(model_name, train_data, test_data, trial_seed)
                    
                    if result is not None:
                        seed_results.append(result)
                        
                        # Save best model from first trial of first seed
                        if seed == seeds[0] and trial == 0:
                            model_save_path = f"{self.results_dir}/trained_models/{model_name}_best_model.pkl"
                            with open(model_save_path, 'wb') as f:
                                pickle.dump(result['additional_objects'], f)
                
                model_results[seed] = seed_results
            
            results[model_name] = model_results
        
        # Save results
        results_path = f"{self.results_dir}/experiment_1_excess_mortality/results.pkl"
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        
        # Generate summary statistics and plots
        self.analyze_experiment_1_results(results, train_data, test_data)
        
        return results
    
    def experiment_2_variance_analysis(self, data_splits: Dict, models: List[str], 
                                     seeds: List[int], trials_per_seed: int = 30):
        """
        Experiment 2: Variance reduction analysis over different forecasting horizons
        """
        print("\n" + "="*80)
        print("EXPERIMENT 2: VARIANCE REDUCTION ANALYSIS")
        print("="*80)
        
        variance_results = {}
        
        for period_name, split in data_splits.items():
            print(f"\nAnalyzing {period_name} (test length: {split['test_length']} months)...")
            
            train_data = split['train']
            test_data = split['test']
            
            period_results = {}
            
            for model_name in models:
                print(f"  Training {model_name.upper()}...")
                model_results = {}
                
                for seed in seeds:
                    seed_results = []
                    
                    for trial in range(trials_per_seed):
                        trial_seed = seed + trial * 1000
                        result = self.run_single_trial(model_name, train_data, test_data, trial_seed)
                        
                        if result is not None:
                            seed_results.append(result)
                    
                    model_results[seed] = seed_results
                
                period_results[model_name] = model_results
            
            variance_results[period_name] = period_results
        
        # Save results
        results_path = f"{self.results_dir}/experiment_2_variance_analysis/results.pkl"
        with open(results_path, 'wb') as f:
            pickle.dump(variance_results, f)
        
        # Analyze variance reduction
        self.analyze_variance_reduction(variance_results, data_splits)
        
        return variance_results
    
    def experiment_3_sensitivity_analysis(self, data_splits: Dict, models: List[str]):
        """
        Experiment 3: Sensitivity analysis for seeds and trials
        """
        print("\n" + "="*80)
        print("EXPERIMENT 3: SENSITIVITY ANALYSIS")
        print("="*80)
        
        # Use the full test period for sensitivity analysis
        split = data_splits['test_period_4']
        train_data = split['train']
        test_data = split['test']
        
        # Test different numbers of seeds and trials
        seed_tests = [5, 10, 20, 50]
        trial_tests = [10, 20, 30, 50]
        
        sensitivity_results = {}
        
        # Focus on deep learning models for sensitivity analysis
        dl_models = [m for m in models if m != 'sarima']
        
        for model_name in dl_models:
            print(f"\nSensitivity analysis for {model_name.upper()}...")
            model_sensitivity = {}
            
            # Test different numbers of seeds
            print("  Testing seed sensitivity...")
            for n_seeds in seed_tests:
                seeds = list(range(42, 42 + n_seeds))
                seed_results = []
                
                for seed in seeds:
                    # Use fixed number of trials (30) for seed testing
                    trials = 30
                    trial_results = []
                    
                    for trial in range(trials):
                        trial_seed = seed + trial * 1000
                        result = self.run_single_trial(model_name, train_data, test_data, trial_seed)
                        
                        if result is not None:
                            trial_results.append(result)
                    
                    if trial_results:
                        # Calculate average metrics for this seed
                        avg_test_rmse = np.mean([r['test_metrics']['RMSE'] for r in trial_results])
                        avg_test_mae = np.mean([r['test_metrics']['MAE'] for r in trial_results])
                        avg_test_mape = np.mean([r['test_metrics']['MAPE'] for r in trial_results])
                        
                        seed_results.append({
                            'seed': seed,
                            'avg_test_rmse': avg_test_rmse,
                            'avg_test_mae': avg_test_mae,
                            'avg_test_mape': avg_test_mape,
                            'n_trials': len(trial_results)
                        })
                
                model_sensitivity[f'seeds_{n_seeds}'] = seed_results
            
            # Test different numbers of trials
            print("  Testing trial sensitivity...")
            for n_trials in trial_tests:
                trial_results = []
                
                # Use fixed seed (42) for trial testing
                seed = 42
                
                for trial in range(n_trials):
                    trial_seed = seed + trial * 1000
                    result = self.run_single_trial(model_name, train_data, test_data, trial_seed)
                    
                    if result is not None:
                        trial_results.append(result)
                        
                        # Save cumulative results every 10 trials
                        if (trial + 1) % 10 == 0:
                            cumulative_rmse = np.mean([r['test_metrics']['RMSE'] for r in trial_results])
                            cumulative_mae = np.mean([r['test_metrics']['MAE'] for r in trial_results])
                            cumulative_mape = np.mean([r['test_metrics']['MAPE'] for r in trial_results])
                            
                            if f'trials_{n_trials}' not in model_sensitivity:
                                model_sensitivity[f'trials_{n_trials}'] = []
                            
                            model_sensitivity[f'trials_{n_trials}'].append({
                                'n_trials_completed': trial + 1,
                                'cumulative_test_rmse': cumulative_rmse,
                                'cumulative_test_mae': cumulative_mae,
                                'cumulative_test_mape': cumulative_mape,
                                'rmse_std': np.std([r['test_metrics']['RMSE'] for r in trial_results])
                            })
            
            sensitivity_results[model_name] = model_sensitivity
        
        # Save results
        results_path = f"{self.results_dir}/experiment_3_sensitivity/results.pkl"
        with open(results_path, 'wb') as f:
            pickle.dump(sensitivity_results, f)
        
        # Analyze sensitivity
        self.analyze_sensitivity_results(sensitivity_results)
        
        return sensitivity_results
    
    def analyze_experiment_1_results(self, results: Dict, train_data: pd.DataFrame, test_data: pd.DataFrame):
        """Analyze and visualize Experiment 1 results"""
        print("\nAnalyzing Experiment 1 results...")
        
        # Create comprehensive comparison plots similar to sarima_vs_lstm_comparison.png
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Excess Mortality Forecasting: Model Comparisons (2015-2019 train, 2020-2023 test)', 
                     fontsize=16, fontweight='bold')
        
        # Get SARIMA results for comparison
        sarima_results = results.get('sarima', {})
        if not sarima_results:
            print("Warning: No SARIMA results found")
            return
        
        # Use first seed, first trial for plotting
        first_seed = list(sarima_results.keys())[0]
        sarima_data = sarima_results[first_seed][0]
        
        # Create date ranges for plotting
        train_dates = pd.date_range(start=train_data['Date'].min(), 
                                   end=train_data['Date'].max(), freq='M')
        test_dates = pd.date_range(start=test_data['Date'].min(), 
                                  end=test_data['Date'].max(), freq='M')
        
        # Adjust for model-specific lookback
        sarima_train_dates = train_dates
        sarima_train_true = sarima_data['train_true']
        sarima_train_pred = sarima_data['train_pred']
        
        # Plot comparisons
        models_to_compare = ['lstm', 'tcn', 'seq2seq_attn', 'transformer']
        
        for i, model_name in enumerate(models_to_compare):
            if model_name not in results:
                continue
                
            ax = axes[i//2, i%2]
            
            # Get model results
            model_results = results[model_name][first_seed][0]
            
            # Adjust dates for lookback
            if model_name != 'sarima':
                lookback = OPTIMAL_PARAMS[model_name]['lookback']
                model_train_dates = train_dates[lookback:]
            else:
                model_train_dates = train_dates
                
            model_train_true = model_results['train_true']
            model_train_pred = model_results['train_pred']
            model_test_true = model_results['test_true']
            model_test_pred = model_results['test_pred']
            
            # Combine all data for plotting
            all_dates = np.concatenate([model_train_dates[:len(model_train_true)], test_dates[:len(model_test_true)]])
            all_actual = np.concatenate([model_train_true, model_test_true])
            all_sarima_pred = np.concatenate([sarima_train_pred[:len(model_train_true)], 
                                            sarima_data['test_pred'][:len(model_test_true)]])
            all_model_pred = np.concatenate([model_train_pred, model_test_pred])
            
            # Plot actual data
            ax.plot(all_dates, all_actual, 'k-', linewidth=2, label='Actual Deaths', zorder=5)
            
            # Plot SARIMA predictions
            ax.plot(all_dates, all_sarima_pred, '--', color='blue', linewidth=1.5, 
                   label='SARIMA Predictions', alpha=0.8)
            
            # Plot model predictions
            ax.plot(all_dates, all_model_pred, '-', color='red', linewidth=1.5,
                   label=f'{model_name.upper()} Predictions', alpha=0.8)
            
            # Add prediction intervals
            sarima_lower = np.concatenate([sarima_data['train_lower'][:len(model_train_true)], 
                                         sarima_data['test_lower'][:len(model_test_true)]])
            sarima_upper = np.concatenate([sarima_data['train_upper'][:len(model_train_true)], 
                                         sarima_data['test_upper'][:len(model_test_true)]])
            
            model_lower = np.concatenate([model_results['train_lower'], model_results['test_lower']])
            model_upper = np.concatenate([model_results['train_upper'], model_results['test_upper']])
            
            ax.fill_between(all_dates, sarima_lower, sarima_upper, alpha=0.2, color='blue', 
                           label='SARIMA 95% PI')
            ax.fill_between(all_dates, model_lower, model_upper, alpha=0.2, color='red',
                           label=f'{model_name.upper()} 95% PI')
            
            # Add forecast start line
            forecast_start = test_data['Date'].min()
            ax.axvline(forecast_start, color='green', linestyle='--', alpha=0.7, linewidth=2)
            
            # Formatting
            ax.set_title(f'SARIMA vs {model_name.upper()}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Deaths', fontsize=12)
            ax.legend(loc='upper left', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Rotate x-axis labels
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/figures/experiment_1_model_comparisons.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create summary statistics table
        summary_stats = []
        
        for model_name, model_results in results.items():
            all_test_rmse = []
            all_test_mae = []
            all_test_mape = []
            all_test_coverage = []
            all_test_width = []
            
            for seed_results in model_results.values():
                for trial_result in seed_results:
                    all_test_rmse.append(trial_result['test_metrics']['RMSE'])
                    all_test_mae.append(trial_result['test_metrics']['MAE'])
                    all_test_mape.append(trial_result['test_metrics']['MAPE'])
                    all_test_coverage.append(trial_result['test_coverage'])
                    all_test_width.append(trial_result['test_width'])
            
            summary_stats.append({
                'Model': model_name.upper(),
                'Test_RMSE_Mean': np.mean(all_test_rmse),
                'Test_RMSE_Std': np.std(all_test_rmse),
                'Test_MAE_Mean': np.mean(all_test_mae),
                'Test_MAE_Std': np.std(all_test_mae),
                'Test_MAPE_Mean': np.mean(all_test_mape),
                'Test_MAPE_Std': np.std(all_test_mape),
                'Test_Coverage_Mean': np.mean(all_test_coverage),
                'Test_Coverage_Std': np.std(all_test_coverage),
                'Test_Width_Mean': np.mean(all_test_width),
                'Test_Width_Std': np.std(all_test_width),
                'N_Trials': len(all_test_rmse)
            })
        
        summary_df = pd.DataFrame(summary_stats)
        summary_df.to_csv(f'{self.results_dir}/experiment_1_excess_mortality/summary_statistics.csv', 
                         index=False)
        
        print("Experiment 1 analysis complete!")
        print(f"Summary statistics saved to: {self.results_dir}/experiment_1_excess_mortality/summary_statistics.csv")
        print(f"Comparison plots saved to: {self.results_dir}/figures/experiment_1_model_comparisons.png")
    
    def analyze_variance_reduction(self, variance_results: Dict, data_splits: Dict):
        """Analyze variance reduction across different forecasting horizons"""
        print("\nAnalyzing variance reduction across forecasting horizons...")
        
        # Extract metrics for each time horizon
        horizon_analysis = {}
        
        for period_name, period_results in variance_results.items():
            test_length = data_splits[period_name]['test_length']
            
            horizon_metrics = {}
            
            for model_name, model_results in period_results.items():
                all_rmse = []
                all_width = []
                all_coverage = []
                
                for seed_results in model_results.values():
                    for trial_result in seed_results:
                        all_rmse.append(trial_result['test_metrics']['RMSE'])
                        all_width.append(trial_result['test_width'])
                        all_coverage.append(trial_result['test_coverage'])
                
                horizon_metrics[model_name] = {
                    'rmse_mean': np.mean(all_rmse),
                    'rmse_std': np.std(all_rmse),
                    'width_mean': np.mean(all_width),
                    'width_std': np.std(all_width),
                    'coverage_mean': np.mean(all_coverage),
                    'coverage_std': np.std(all_coverage)
                }
            
            horizon_analysis[test_length] = horizon_metrics
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Variance Analysis Across Forecasting Horizons', fontsize=16, fontweight='bold')
        
        horizons = sorted(horizon_analysis.keys())
        models = list(next(iter(horizon_analysis.values())).keys())
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
        
        # RMSE vs Horizon
        ax = axes[0, 0]
        for i, model in enumerate(models):
            rmse_means = [horizon_analysis[h][model]['rmse_mean'] for h in horizons]
            rmse_stds = [horizon_analysis[h][model]['rmse_std'] for h in horizons]
            
            ax.errorbar(horizons, rmse_means, yerr=rmse_stds, 
                       label=model.upper(), marker='o', color=colors[i])
        
        ax.set_xlabel('Forecast Horizon (months)')
        ax.set_ylabel('RMSE')
        ax.set_title('RMSE vs Forecast Horizon')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # PI Width vs Horizon
        ax = axes[0, 1]
        for i, model in enumerate(models):
            width_means = [horizon_analysis[h][model]['width_mean'] for h in horizons]
            width_stds = [horizon_analysis[h][model]['width_std'] for h in horizons]
            
            ax.errorbar(horizons, width_means, yerr=width_stds, 
                       label=model.upper(), marker='s', color=colors[i])
        
        ax.set_xlabel('Forecast Horizon (months)')
        ax.set_ylabel('Prediction Interval Width')
        ax.set_title('PI Width vs Forecast Horizon')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Coverage vs Horizon
        ax = axes[1, 0]
        for i, model in enumerate(models):
            coverage_means = [horizon_analysis[h][model]['coverage_mean'] for h in horizons]
            coverage_stds = [horizon_analysis[h][model]['coverage_std'] for h in horizons]
            
            ax.errorbar(horizons, coverage_means, yerr=coverage_stds, 
                       label=model.upper(), marker='^', color=colors[i])
        
        ax.axhline(y=95, color='black', linestyle='--', alpha=0.5, label='Target 95%')
        ax.set_xlabel('Forecast Horizon (months)')
        ax.set_ylabel('Coverage (%)')
        ax.set_title('PI Coverage vs Forecast Horizon')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # RMSE Growth Rate
        ax = axes[1, 1]
        for i, model in enumerate(models):
            rmse_means = [horizon_analysis[h][model]['rmse_mean'] for h in horizons]
            if len(rmse_means) > 1:
                growth_rates = [(rmse_means[j] - rmse_means[0]) / rmse_means[0] * 100 
                               for j in range(len(rmse_means))]
                ax.plot(horizons, growth_rates, label=model.upper(), 
                       marker='D', color=colors[i])
        
        ax.set_xlabel('Forecast Horizon (months)')
        ax.set_ylabel('RMSE Growth Rate (%)')
        ax.set_title('RMSE Growth Rate vs Forecast Horizon')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/figures/experiment_2_variance_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save detailed results
        detailed_results = []
        for horizon in horizons:
            for model_name, metrics in horizon_analysis[horizon].items():
                detailed_results.append({
                    'Forecast_Horizon_Months': horizon,
                    'Model': model_name.upper(),
                    'RMSE_Mean': metrics['rmse_mean'],
                    'RMSE_Std': metrics['rmse_std'],
                    'PI_Width_Mean': metrics['width_mean'],
                    'PI_Width_Std': metrics['width_std'],
                    'Coverage_Mean': metrics['coverage_mean'],
                    'Coverage_Std': metrics['coverage_std']
                })
        
        detailed_df = pd.DataFrame(detailed_results)
        detailed_df.to_csv(f'{self.results_dir}/experiment_2_variance_analysis/detailed_results.csv', 
                          index=False)
        
        print("Variance reduction analysis complete!")
        print(f"Detailed results saved to: {self.results_dir}/experiment_2_variance_analysis/detailed_results.csv")
        print(f"Variance plots saved to: {self.results_dir}/figures/experiment_2_variance_analysis.png")
    
    def analyze_sensitivity_results(self, sensitivity_results: Dict):
        """Analyze sensitivity to seeds and trials"""
        print("\nAnalyzing sensitivity results...")
        
        for model_name, model_sensitivity in sensitivity_results.items():
            print(f"\nAnalyzing {model_name.upper()} sensitivity...")
            
            # Create sensitivity plots
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'{model_name.upper()} Sensitivity Analysis', fontsize=16, fontweight='bold')
            
            # Seed sensitivity
            seed_data = {k: v for k, v in model_sensitivity.items() if k.startswith('seeds_')}
            
            if seed_data:
                n_seeds_list = []
                rmse_means = []
                rmse_stds = []
                
                for key, results in seed_data.items():
                    n_seeds = int(key.split('_')[1])
                    rmse_values = [r['avg_test_rmse'] for r in results]
                    
                    n_seeds_list.append(n_seeds)
                    rmse_means.append(np.mean(rmse_values))
                    rmse_stds.append(np.std(rmse_values))
                
                # Plot RMSE vs number of seeds
                ax = axes[0, 0]
                ax.errorbar(n_seeds_list, rmse_means, yerr=rmse_stds, 
                           marker='o', capsize=5, capthick=2)
                ax.set_xlabel('Number of Seeds')
                ax.set_ylabel('Average Test RMSE')
                ax.set_title('RMSE vs Number of Seeds')
                ax.grid(True, alpha=0.3)
                
                # Plot RMSE standard deviation vs number of seeds
                ax = axes[0, 1]
                ax.plot(n_seeds_list, rmse_stds, marker='s', color='red')
                ax.set_xlabel('Number of Seeds')
                ax.set_ylabel('RMSE Standard Deviation')
                ax.set_title('RMSE Variability vs Number of Seeds')
                ax.grid(True, alpha=0.3)
            
            # Trial sensitivity
            trial_data = {k: v for k, v in model_sensitivity.items() if k.startswith('trials_')}
            
            if trial_data:
                # Plot convergence for different trial numbers
                ax = axes[1, 0]
                
                for key, results in trial_data.items():
                    n_trials_max = int(key.split('_')[1])
                    trials_completed = [r['n_trials_completed'] for r in results]
                    cumulative_rmse = [r['cumulative_test_rmse'] for r in results]
                    
                    ax.plot(trials_completed, cumulative_rmse, 
                           marker='o', label=f'Max {n_trials_max} trials')
                
                ax.set_xlabel('Number of Trials Completed')
                ax.set_ylabel('Cumulative Average RMSE')
                ax.set_title('RMSE Convergence vs Number of Trials')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Plot RMSE standard deviation convergence
                ax = axes[1, 1]
                
                for key, results in trial_data.items():
                    n_trials_max = int(key.split('_')[1])
                    trials_completed = [r['n_trials_completed'] for r in results]
                    rmse_std = [r['rmse_std'] for r in results]
                    
                    ax.plot(trials_completed, rmse_std, 
                           marker='s', label=f'Max {n_trials_max} trials')
                
                ax.set_xlabel('Number of Trials Completed')
                ax.set_ylabel('RMSE Standard Deviation')
                ax.set_title('RMSE Std Convergence vs Number of Trials')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{self.results_dir}/figures/experiment_3_sensitivity_{model_name}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save sensitivity data
            sensitivity_df_data = []
            
            # Add seed sensitivity data
            for key, results in seed_data.items():
                n_seeds = int(key.split('_')[1])
                for result in results:
                    sensitivity_df_data.append({
                        'Analysis_Type': 'Seed_Sensitivity',
                        'N_Seeds': n_seeds,
                        'N_Trials': result['n_trials'],
                        'Seed': result['seed'],
                        'Test_RMSE': result['avg_test_rmse'],
                        'Test_MAE': result['avg_test_mae'],
                        'Test_MAPE': result['avg_test_mape']
                    })
            
            # Add trial sensitivity data
            for key, results in trial_data.items():
                n_trials_max = int(key.split('_')[1])
                for result in results:
                    sensitivity_df_data.append({
                        'Analysis_Type': 'Trial_Sensitivity',
                        'N_Trials_Max': n_trials_max,
                        'N_Trials_Completed': result['n_trials_completed'],
                        'Cumulative_RMSE': result['cumulative_test_rmse'],
                        'Cumulative_MAE': result['cumulative_test_mae'],
                        'Cumulative_MAPE': result['cumulative_test_mape'],
                        'RMSE_Std': result['rmse_std']
                    })
            
            if sensitivity_df_data:
                sensitivity_df = pd.DataFrame(sensitivity_df_data)
                sensitivity_df.to_csv(f'{self.results_dir}/experiment_3_sensitivity/sensitivity_{model_name}.csv', 
                                     index=False)
        
        print("Sensitivity analysis complete!")
        print(f"Sensitivity plots saved to: {self.results_dir}/figures/experiment_3_sensitivity_[model].png")
        print(f"Sensitivity data saved to: {self.results_dir}/experiment_3_sensitivity/sensitivity_[model].csv")


def main():
    """Main execution function"""
    print("="*80)
    print("COMPREHENSIVE EVALUATION PIPELINE")
    print("Advanced Machine Learning for Substance Overdose Mortality Prediction")
    print("="*80)
    
    # Initialize pipeline
    pipeline = ComprehensiveEvaluationPipeline(DATA_PATH, RESULTS_DIR)
    
    # Load and preprocess data
    df = pipeline.load_and_preprocess_data()
    
    # Create train/test splits for different experiments
    data_splits = pipeline.create_train_test_splits(df)
    
    print(f"\nData splits created:")
    for split_name, split_info in data_splits.items():
        print(f"  {split_name}: Train {len(split_info['train'])} months, Test {len(split_info['test'])} months")
    
    # Define models and experimental parameters
    models = ['sarima', 'lstm', 'tcn', 'seq2seq_attn', 'transformer']
    seeds = [42, 123, 456, 789, 1000]  # 5 seeds as baseline
    trials_per_seed = 30
    
    print(f"\nExperimental setup:")
    print(f"  Models: {models}")
    print(f"  Seeds: {len(seeds)} ({min(seeds)} to {max(seeds)})")
    print(f"  Trials per seed: {trials_per_seed}")
    
    # Run experiments
    try:
        # Experiment 1: Excess mortality estimation
        exp1_results = pipeline.experiment_1_excess_mortality(
            data_splits, models, seeds, trials_per_seed)
        
        # Experiment 2: Variance reduction analysis
        exp2_results = pipeline.experiment_2_variance_analysis(
            data_splits, models, seeds, trials_per_seed)
        
        # Experiment 3: Sensitivity analysis
        exp3_results = pipeline.experiment_3_sensitivity_analysis(
            data_splits, models)
        
        print("\n" + "="*80)
        print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"\nResults saved to: {RESULTS_DIR}/")
        print("\nGenerated files:")
        print("├── experiment_1_excess_mortality/")
        print("│   ├── results.pkl")
        print("│   └── summary_statistics.csv")
        print("├── experiment_2_variance_analysis/")
        print("│   ├── results.pkl")
        print("│   └── detailed_results.csv")
        print("├── experiment_3_sensitivity/")
        print("│   ├── results.pkl")
        print("│   └── sensitivity_[model].csv")
        print("├── trained_models/")
        print("│   └── [model]_best_model.pkl")
        print("└── figures/")
        print("    ├── experiment_1_model_comparisons.png")
        print("    ├── experiment_2_variance_analysis.png")
        print("    └── experiment_3_sensitivity_[model].png")
        
        print(f"\nTotal execution time: Please check individual experiment outputs for timing.")
        
    except Exception as e:
        print(f"\nERROR: Experiment failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()