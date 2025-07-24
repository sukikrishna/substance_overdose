#!/usr/bin/env python3
"""
Fixed Streamlined Evaluation Pipeline for Substance Overdose Mortality Forecasting
==================================================================================

This version fixes the Seq2Seq and Transformer architectures and aligns plotting properly.
- Single random seed (42) for reproducibility
- 30 trials per model for robust statistics
- Experiments 1 and 2 with proper model overlap handling
- Fixed model architectures that actually work
- Proper plot alignment after lookback period

Experiments:
1. Excess mortality estimation (2015-2019 train, 2020-2023 test)
2. Variance analysis across different forecast horizons (reuses models from exp 1)

Usage:
    python streamlined_evaluation_fixed.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (LSTM, Dense, Conv1D, GRU, Dropout, 
                                   GlobalAveragePooling1D, Input, 
                                   MultiHeadAttention, LayerNormalization, Add,
                                   Concatenate, RepeatVector)
from tensorflow.keras.optimizers import Adam
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
RESULTS_DIR = 'streamlined_eval_results_fixed_v2'
DATA_PATH = 'data_updated/state_month_overdose_2015_2023.xlsx'
TRIALS_PER_MODEL = 30

# Optimal hyperparameters aligned with working models
OPTIMAL_PARAMS = {
    'sarima': {'order': (1, 1, 1), 'seasonal_order': (1, 1, 1, 12)},
    'lstm': {'lookback': 12, 'batch_size': 8, 'epochs': 100, 'units': 50, 'dropout': 0.1},
    'tcn': {'lookback': 12, 'batch_size': 8, 'epochs': 100, 'filters': 64, 'kernel_size': 3},
    'seq2seq': {'lookback': 12, 'batch_size': 8, 'epochs': 50, 'encoder_units': 64, 'decoder_units': 64},
    'transformer': {'lookback': 12, 'batch_size': 8, 'epochs': 100, 'd_model': 64, 'num_heads': 4}
}

# Create results directory structure
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(f'{RESULTS_DIR}/experiment_1_excess_mortality', exist_ok=True)
os.makedirs(f'{RESULTS_DIR}/experiment_2_variance_analysis', exist_ok=True)
os.makedirs(f'{RESULTS_DIR}/trained_models', exist_ok=True)
os.makedirs(f'{RESULTS_DIR}/figures', exist_ok=True)
os.makedirs(f'{RESULTS_DIR}/data_exports', exist_ok=True)

class FixedEvaluationPipeline:
    """Fixed evaluation pipeline with working model architectures"""
    
    def __init__(self, data_path: str, results_dir: str):
        self.data_path = data_path
        self.results_dir = results_dir
        self.trained_models = {}
        
    def load_and_preprocess_data(self):
        """Load and preprocess the overdose dataset"""
        print("Loading and preprocessing data...")
        
        df = pd.read_excel(self.data_path)
        print("Data columns:", df.columns.tolist())
        print("Data shape:", df.shape)
        
        # Handle the data format - adjust based on your actual column names
        if 'Row Labels' in df.columns:
            df['Date'] = pd.to_datetime(df['Row Labels'])
        elif 'Month' in df.columns:
            df['Date'] = pd.to_datetime(df['Month'])
        else:
            # Construct from year/month codes
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
        
        # Handle suppressed values
        df['Deaths'] = df['Deaths'].apply(lambda x: 0 if str(x).lower() == 'suppressed' else int(x))
        
        print(f"Processed data shape: {df.shape}")
        print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"Deaths range: {df['Deaths'].min()} to {df['Deaths'].max()}")
        
        return df
    
    def create_data_splits(self, df: pd.DataFrame):
        """Create data splits for both experiments"""
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
    
    def create_dataset(self, dataset, look_back=3):
        """Create dataset for sequence models - matching run_experiments.py"""
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back):
            a = dataset[i:(i + look_back)]
            dataX.append(a)
            dataY.append(dataset[i + look_back])
        return np.array(dataX), np.array(dataY)
    
    def calculate_comprehensive_metrics(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Calculate comprehensive evaluation metrics"""
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
    
    def build_lstm_model(self, look_back: int, units: int = 50, dropout: float = 0.0):
        """Build LSTM model - exactly matching run_experiments.py"""
        model = Sequential([
            LSTM(units, activation='relu', input_shape=(look_back, 1), return_sequences=False),
            Dropout(dropout),
            Dense(1)
        ])
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model
    
    def build_tcn_model(self, look_back: int, filters: int = 64, kernel_size: int = 3):
        """Build TCN model - exactly matching run_experiments.py"""
        model = Sequential([
            Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', 
                   input_shape=(look_back, 1), padding='causal'),
            Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', 
                   padding='causal', dilation_rate=2),
            Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', 
                   padding='causal', dilation_rate=4),
            GlobalAveragePooling1D(),
            Dense(50, activation='relu'),
            Dense(1)
        ])
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model
    
    def build_seq2seq_model(self, look_back: int, encoder_units: int = 64, decoder_units: int = 64):
        """Build working Seq2Seq model with attention"""
        # Scale down for stability
        scaler = MinMaxScaler()
        
        # Encoder
        encoder_inputs = Input(shape=(look_back, 1))
        encoder_gru = GRU(encoder_units, return_sequences=True, return_state=True)
        encoder_outputs, encoder_state = encoder_gru(encoder_inputs)
        
        # Decoder 
        decoder_inputs = Input(shape=(1, 1))
        decoder_gru = GRU(decoder_units, return_sequences=True, return_state=True)
        decoder_outputs, _ = decoder_gru(decoder_inputs, initial_state=encoder_state)
        
        # Simple attention mechanism
        attention = tf.keras.layers.Attention()
        context_vector = attention([decoder_outputs, encoder_outputs])
        
        # Combine context and decoder output
        decoder_combined = Concatenate(axis=-1)([decoder_outputs, context_vector])
        decoder_dense = Dense(decoder_units, activation='relu')(decoder_combined)
        outputs = Dense(1)(decoder_dense)
        
        model = Model([encoder_inputs, decoder_inputs], outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        return model, scaler
    
    def build_transformer_model(self, look_back: int, d_model: int = 64, num_heads: int = 4):
        """Build working Transformer model"""
        scaler = MinMaxScaler()
        
        inputs = Input(shape=(look_back, 1))
        x = Dense(d_model)(inputs)
        
        # Positional encoding (simplified)
        positions = tf.range(start=0, limit=look_back, delta=1, dtype=tf.float32)
        position_encoding = tf.expand_dims(positions, -1) / 10000.0
        x = x + position_encoding
        
        # Multi-head attention
        attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
        x = Add()([x, attn_output])
        x = LayerNormalization()(x)
        
        # Feed forward
        ff_output = Dense(d_model * 4, activation='relu')(x)
        ff_output = Dense(d_model)(ff_output)
        x = Add()([x, ff_output])
        x = LayerNormalization()(x)
        
        # Output
        x = GlobalAveragePooling1D()(x)
        outputs = Dense(1)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        return model, scaler
    
    def generate_forecast(self, model, initial_sequence, num_predictions, look_back, model_type='lstm', scaler=None):
        """Generate forecasts for different model types"""
        predictions = []
        
        if model_type in ['lstm', 'tcn']:
            current_sequence = initial_sequence.copy()
            for _ in range(num_predictions):
                next_prediction = model.predict(current_sequence, verbose=0)
                predictions.append(next_prediction[0][0])
                
                # Update sequence
                new_sequence = np.append(current_sequence[0, 1:], [[next_prediction[0][0]]], axis=0)
                current_sequence = new_sequence.reshape((1, look_back, 1))
        
        elif model_type == 'seq2seq':
            encoder_input = initial_sequence
            for _ in range(num_predictions):
                decoder_input = np.zeros((1, 1, 1))
                prediction = model.predict([encoder_input, decoder_input], verbose=0)
                pred_value = prediction[0][0][0]
                predictions.append(pred_value)
                
                # Update encoder input
                encoder_input = np.roll(encoder_input, -1, axis=1)
                encoder_input[0, -1, 0] = pred_value
        
        elif model_type == 'transformer':
            current_sequence = initial_sequence.copy()
            for _ in range(num_predictions):
                next_prediction = model.predict(current_sequence, verbose=0)
                pred_value = next_prediction[0][0]
                predictions.append(pred_value)
                
                # Update sequence
                new_sequence = np.append(current_sequence[0, 1:], [[pred_value]], axis=0)
                current_sequence = new_sequence.reshape((1, look_back, 1))
        
        return np.array(predictions)
    
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
    
    def train_sequence_model(self, model_name: str, train_data: pd.DataFrame, params: Dict):
        """Train sequence models (LSTM, TCN, Seq2Seq, Transformer)"""
        deaths_series = train_data['Deaths'].values.astype(float)
        look_back = params['lookback']
        
        # Create sequences using the same method as run_experiments.py
        trainX, trainY = self.create_dataset(pd.Series(deaths_series), look_back)
        trainX = trainX.reshape((trainX.shape[0], look_back, 1))
        
        # Build model based on type
        if model_name == 'lstm':
            model = self.build_lstm_model(look_back, params['units'], params['dropout'])
            scaler = None
            
        elif model_name == 'tcn':
            model = self.build_tcn_model(look_back, params['filters'], params['kernel_size'])
            scaler = None
            
        elif model_name == 'seq2seq':
            model, scaler = self.build_seq2seq_model(look_back, params['encoder_units'], params['decoder_units'])
            # Scale training data
            scaler.fit(deaths_series.reshape(-1, 1))
            trainX_scaled = scaler.transform(trainX.reshape(-1, 1)).reshape(trainX.shape)
            trainY_scaled = scaler.transform(trainY.reshape(-1, 1)).flatten()
            trainX, trainY = trainX_scaled, trainY_scaled
            
        elif model_name == 'transformer':
            model, scaler = self.build_transformer_model(look_back, params['d_model'], params['num_heads'])
            # Scale training data
            scaler.fit(deaths_series.reshape(-1, 1))
            trainX_scaled = scaler.transform(trainX.reshape(-1, 1)).reshape(trainX.shape)
            trainY_scaled = scaler.transform(trainY.reshape(-1, 1)).flatten()
            trainX, trainY = trainX_scaled, trainY_scaled
        
        # Train model
        if model_name == 'seq2seq':
            # Special training for seq2seq
            decoder_input = np.zeros((trainX.shape[0], 1, 1))
            model.fit([trainX, decoder_input], trainY.reshape(-1, 1, 1),
                     epochs=params['epochs'],
                     batch_size=params['batch_size'],
                     verbose=0, validation_split=0.1)
        else:
            model.fit(trainX, trainY,
                     epochs=params['epochs'],
                     batch_size=params['batch_size'],
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
        
        else:
            # Sequence models
            look_back = params['lookback']
            
            # Prepare scaled data if needed
            if scaler is not None:
                train_deaths_scaled = scaler.transform(train_deaths.reshape(-1, 1)).flatten()
                test_deaths_scaled = scaler.transform(test_deaths.reshape(-1, 1)).flatten()
            else:
                train_deaths_scaled = train_deaths
                test_deaths_scaled = test_deaths
            
            # Training predictions (in-sample)
            train_X, train_Y = self.create_dataset(pd.Series(train_deaths_scaled), look_back)
            train_X = train_X.reshape((train_X.shape[0], look_back, 1))
            
            if model_type == 'seq2seq':
                decoder_input = np.zeros((train_X.shape[0], 1, 1))
                train_pred_scaled = model.predict([train_X, decoder_input], verbose=0)
                train_pred_scaled = train_pred_scaled.reshape(-1)
            else:
                train_pred_scaled = model.predict(train_X, verbose=0).reshape(-1)
            
            # Test predictions (out-of-sample, autoregressive)
            initial_sequence = train_X[-1].reshape((1, look_back, 1))
            test_pred_scaled = self.generate_forecast(model, initial_sequence, 
                                                    len(test_deaths), look_back, model_type, scaler)
            
            # Inverse transform if scaler was used
            if scaler is not None:
                train_pred = scaler.inverse_transform(train_pred_scaled.reshape(-1, 1)).flatten()
                test_pred = scaler.inverse_transform(test_pred_scaled.reshape(-1, 1)).flatten()
            else:
                train_pred = train_pred_scaled
                test_pred = test_pred_scaled
            
            return (train_deaths[look_back:], train_pred, 
                   test_deaths, test_pred)
    
    def run_single_trial(self, model_name: str, train_data: pd.DataFrame, 
                        test_data: pd.DataFrame, trial_num: int = 1):
        """Run a single trial for a given model"""
        
        if trial_num % 5 == 0:
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
                scaler = None
                
            else:
                # Sequence models
                model, scaler = self.train_sequence_model(model_name, train_data, params)
                train_true, train_pred, test_true, test_pred = self.generate_predictions(
                    model, model_name, train_data, test_data, params, scaler)
                
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
                'model_object': model_obj if trial_num == 1 else None,  # Save only first model
                'scaler': scaler if trial_num == 1 else None
            }
            
            return results
            
        except Exception as e:
            print(f"    Error in trial {trial_num}: {e}")
            return None
    
    def experiment_1_excess_mortality(self, data_splits: Dict, models: List[str]):
        """Experiment 1: Excess mortality estimation (2015-2019 train, 2020-2023 test)"""
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
                            pickle.dump({
                                'model': result['model_object'], 
                                'scaler': result.get('scaler')
                            }, f)
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
        
        # Generate analysis and visualizations
        self.analyze_experiment_1_results(all_results, train_data, test_data)
        
        return all_results
    
    def experiment_2_variance_analysis(self, data_splits: Dict, models: List[str], exp1_results: Dict):
        """Experiment 2: Evaluate trained models on different horizons"""
        print("\n" + "="*80)
        print("EXPERIMENT 2: VARIANCE ANALYSIS ACROSS FORECAST HORIZONS")
        print("="*80)
        print("Using trained models from Experiment 1 on different test horizons")
        
        horizon_results = {}
        
        for horizon_name in ['horizon_1', 'horizon_2', 'horizon_3', 'horizon_4']:
            if horizon_name not in data_splits:
                continue
                
            split = data_splits[horizon_name]
            train_data = split['train']
            test_data = split['test']
            
            print(f"\n{split['description']}")
            print(f"Train data: {len(train_data)} months, Test data: {len(test_data)} months")
            
            horizon_model_results = {}
            
            for model_name in models:
                if model_name not in exp1_results or not exp1_results[model_name]:
                    continue
                
                print(f"  Evaluating {model_name.upper()} on this horizon...")
                model_results = []
                
                # Use models from experiment 1
                for i, exp1_result in enumerate(exp1_results[model_name]):
                    if i >= TRIALS_PER_MODEL:
                        break
                    
                    # For first trial, use the saved model, for others retrain
                    if i == 0 and exp1_result['model_object'] is not None:
                        # Use saved model
                        model_obj = exp1_result['model_object']
                        scaler = exp1_result.get('scaler')
                        model_name_type = model_name
                    else:
                        # Retrain model (same as in experiment 1)
                        trial_result = self.run_single_trial(model_name, train_data, test_data, i + 1)
                        if trial_result is None:
                            continue
                        model_results.append(trial_result)
                        continue
                    
                    # Generate predictions for this horizon using the trained model
                    try:
                        params = OPTIMAL_PARAMS[model_name]
                        train_true, train_pred, test_true, test_pred = self.generate_predictions(
                            model_obj, model_name, train_data, test_data, params, scaler)
                        
                        # Calculate metrics
                        train_metrics = self.calculate_comprehensive_metrics(train_true, train_pred)
                        test_metrics = self.calculate_comprehensive_metrics(test_true, test_pred)
                        
                        result = {
                            'trial': i + 1,
                            'model_name': model_name,
                            'train_true': train_true,
                            'train_pred': train_pred,
                            'test_true': test_true,
                            'test_pred': test_pred,
                            'train_metrics': train_metrics,
                            'test_metrics': test_metrics,
                            'model_object': None,  # Don't save models again
                            'scaler': None
                        }
                        
                        model_results.append(result)
                        
                    except Exception as e:
                        print(f"    Error evaluating trial {i+1}: {e}")
                        continue
                
                # Fill remaining trials by retraining
                while len(model_results) < TRIALS_PER_MODEL:
                    trial_num = len(model_results) + 1
                    trial_result = self.run_single_trial(model_name, train_data, test_data, trial_num)
                    if trial_result is not None:
                        model_results.append(trial_result)
                    else:
                        break
                
                horizon_model_results[model_name] = model_results
                print(f"    Completed {len(model_results)} evaluations")
            
            horizon_results[horizon_name] = {
                'results': horizon_model_results,
                'split_info': split
            }
        
        # Save horizon results
        horizon_results_path = f"{self.results_dir}/experiment_2_variance_analysis/horizon_results.pkl"
        with open(horizon_results_path, 'wb') as f:
            pickle.dump(horizon_results, f)
        
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
            
            # Extract metrics across trials
            train_rmse = [r['train_metrics']['RMSE'] for r in model_results]
            train_mae = [r['train_metrics']['MAE'] for r in model_results]
            train_mape = [r['train_metrics']['MAPE'] for r in model_results]
            train_coverage = [r['train_metrics']['coverage'] for r in model_results]
            
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
                'Test_RMSE_Mean': np.mean(test_rmse),
                'Test_RMSE_Std': np.std(test_rmse),
                'Test_MAE_Mean': np.mean(test_mae),
                'Test_MAE_Std': np.std(test_mae),
                'Test_MAPE_Mean': np.mean(test_mape),
                'Test_MAPE_Std': np.std(test_mape),
                'Test_Coverage_Mean': np.mean(test_coverage),
                'Test_Coverage_Std': np.std(test_coverage),
                'Test_Interval_Width_Mean': np.mean(test_width),
                'Test_Interval_Width_Std': np.std(test_width),
                'N_Trials': len(model_results)
            })
        
        # Save summary statistics
        summary_df = pd.DataFrame(summary_stats)
        summary_path = f"{self.results_dir}/experiment_1_excess_mortality/summary_statistics.csv"
        summary_df.to_csv(summary_path, index=False)
        
        print(f"Summary statistics saved to: {summary_path}")
        
        # Create comparison plots
        self.create_model_comparison_plots(results, train_data, test_data)
        
        # Export prediction data
        self.export_prediction_data(results, train_data, test_data, 'experiment_1')
    
    def create_model_comparison_plots(self, results: Dict, train_data: pd.DataFrame, test_data: pd.DataFrame):
        """Create comprehensive model comparison plots with proper alignment"""
        
        print("Creating model comparison plots...")
        
        # Define colors for each model
        model_colors = {
            'sarima': 'blue',
            'lstm': 'red', 
            'tcn': 'green',
            'seq2seq': 'purple',
            'transformer': 'orange'
        }
        
        # Create main comparison plot
        plt.figure(figsize=(18, 12))
        
        # Determine the common start date (after maximum lookback period)
        max_lookback = max([OPTIMAL_PARAMS[model].get('lookback', 0) 
                           for model in results.keys() if model != 'sarima'])
        
        # Start plotting from after the lookback period
        plot_start_idx = max_lookback
        plot_train_dates = train_data['Date'].iloc[plot_start_idx:].values
        plot_test_dates = test_data['Date'].values
        all_plot_dates = np.concatenate([plot_train_dates, plot_test_dates])
        
        # Plot actual data starting from after lookback
        plot_train_actual = train_data['Deaths'].iloc[plot_start_idx:].values
        plot_test_actual = test_data['Deaths'].values
        all_plot_actual = np.concatenate([plot_train_actual, plot_test_actual])
        
        plt.plot(all_plot_dates, all_plot_actual, 'k-', linewidth=3, label='Actual Deaths', zorder=10)
        
        # Add vertical line at forecast start
        forecast_start = test_data['Date'].iloc[0]
        plt.axvline(forecast_start, color='gray', linestyle='--', linewidth=2, 
                   alpha=0.7, label='Forecast Start (Jan 2020)')
        
        # Plot each model's predictions
        for model_name, model_results in results.items():
            if not model_results:
                continue
            
            color = model_colors.get(model_name, 'black')
            
            # Calculate average predictions across trials
            avg_train_pred = np.mean([r['train_pred'] for r in model_results], axis=0)
            avg_test_pred = np.mean([r['test_pred'] for r in model_results], axis=0)
            
            # Get corresponding actual values
            train_true = model_results[0]['train_true']
            test_true = model_results[0]['test_true']
            
            # Handle alignment - SARIMA starts from beginning, others after lookback
            if model_name == 'sarima':
                # Align SARIMA with other models by starting from same point
                model_train_pred = avg_train_pred[plot_start_idx:]
                model_train_dates = plot_train_dates
            else:
                # Sequence models already start after lookback
                model_train_pred = avg_train_pred
                model_train_dates = plot_train_dates
            
            # Combine predictions and dates
            all_model_dates = np.concatenate([model_train_dates, plot_test_dates])
            all_model_pred = np.concatenate([model_train_pred, avg_test_pred])
            
            # Plot predictions
            plt.plot(all_model_dates, all_model_pred, '-', color=color, linewidth=2,
                    label=f'{model_name.upper()} Predictions', alpha=0.8)
            
            # Calculate and plot prediction intervals
            if model_name == 'sarima':
                aligned_train_true = train_true[plot_start_idx:]
                train_metrics = self.calculate_comprehensive_metrics(aligned_train_true, model_train_pred)
            else:
                train_metrics = self.calculate_comprehensive_metrics(train_true, model_train_pred)
            
            test_metrics = self.calculate_comprehensive_metrics(test_true, avg_test_pred)
            
            all_lower = np.concatenate([train_metrics['lower_bound'], test_metrics['lower_bound']])
            all_upper = np.concatenate([train_metrics['upper_bound'], test_metrics['upper_bound']])
            
            plt.fill_between(all_model_dates, all_lower, all_upper,
                           color=color, alpha=0.2, label=f'{model_name.upper()} 95% PI')
        
        # Formatting
        plt.xlabel('Date', fontsize=16, fontweight='bold')
        plt.ylabel('Deaths', fontsize=16, fontweight='bold')
        plt.title('Substance Overdose Mortality Forecasting: Model Comparison (2015-2023)', 
                 fontsize=20, fontweight='bold', pad=20)
        
        plt.xticks(fontsize=14, rotation=45)
        plt.yticks(fontsize=14)
        plt.legend(loc='upper left', fontsize=12, frameon=True, fancybox=True, shadow=True)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = f"{self.results_dir}/figures/model_comparison_experiment_1.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"  Model comparison plot saved: {plot_path}")
        
        # Create individual model vs SARIMA plots
        if 'sarima' in results:
            self.create_individual_comparison_plots(results, train_data, test_data)
    
    def create_individual_comparison_plots(self, results: Dict, train_data: pd.DataFrame, test_data: pd.DataFrame):
        """Create individual model vs SARIMA comparison plots with proper alignment"""
        
        sarima_results = results.get('sarima', [])
        if not sarima_results:
            return
        
        # Calculate SARIMA averages
        sarima_train_pred = np.mean([r['train_pred'] for r in sarima_results], axis=0)
        sarima_test_pred = np.mean([r['test_pred'] for r in sarima_results], axis=0)
        sarima_train_true = sarima_results[0]['train_true']
        sarima_test_true = sarima_results[0]['test_true']
        
        dl_models = [m for m in results.keys() if m != 'sarima']
        
        for model_name in dl_models:
            if not results[model_name]:
                continue
            
            print(f"  Creating SARIMA vs {model_name.upper()} comparison...")
            
            model_results = results[model_name]
            model_train_pred = np.mean([r['train_pred'] for r in model_results], axis=0)
            model_test_pred = np.mean([r['test_pred'] for r in model_results], axis=0)
            model_train_true = model_results[0]['train_true']
            model_test_true = model_results[0]['test_true']
            
            # Handle alignment - start from after lookback period
            lookback = OPTIMAL_PARAMS[model_name].get('lookback', 0)
            
            # Align dates and predictions
            model_train_dates = train_data['Date'].iloc[lookback:].values
            plot_test_dates = test_data['Date'].values
            all_dates = np.concatenate([model_train_dates, plot_test_dates])
            
            # Align actual values
            all_actual = np.concatenate([model_train_true, model_test_true])
            
            # Align SARIMA predictions to match model timeline
            sarima_aligned_train_pred = sarima_train_pred[lookback:]
            all_sarima_pred = np.concatenate([sarima_aligned_train_pred, sarima_test_pred])
            all_model_pred = np.concatenate([model_train_pred, model_test_pred])
            
            # Calculate aligned metrics for intervals
            sarima_train_metrics = self.calculate_comprehensive_metrics(model_train_true, sarima_aligned_train_pred)
            sarima_test_metrics = self.calculate_comprehensive_metrics(sarima_test_true, sarima_test_pred)
            model_train_metrics = self.calculate_comprehensive_metrics(model_train_true, model_train_pred)
            model_test_metrics = self.calculate_comprehensive_metrics(model_test_true, model_test_pred)
            
            # Combine intervals
            all_sarima_lower = np.concatenate([sarima_train_metrics['lower_bound'], sarima_test_metrics['lower_bound']])
            all_sarima_upper = np.concatenate([sarima_train_metrics['upper_bound'], sarima_test_metrics['upper_bound']])
            all_model_lower = np.concatenate([model_train_metrics['lower_bound'], model_test_metrics['lower_bound']])
            all_model_upper = np.concatenate([model_train_metrics['upper_bound'], model_test_metrics['upper_bound']])
            
            # Create plot
            plt.figure(figsize=(16, 10))
            
            # Plot actual data
            plt.plot(all_dates, all_actual, 'k-', linewidth=3, label='Actual Deaths', zorder=5)
            
            # Plot SARIMA
            plt.plot(all_dates, all_sarima_pred, '--', color='blue', linewidth=2, 
                    label='SARIMA Predictions', alpha=0.8)
            plt.fill_between(all_dates, all_sarima_lower, all_sarima_upper,
                           color='blue', alpha=0.25, label='SARIMA 95% PI')
            
            # Plot model
            plt.plot(all_dates, all_model_pred, '-', color='red', linewidth=2,
                    label=f'{model_name.upper()} Predictions', alpha=0.8)
            plt.fill_between(all_dates, all_model_lower, all_model_upper,
                           color='red', alpha=0.25, label=f'{model_name.upper()} 95% PI')
            
            # Add forecast start line
            forecast_start = test_data['Date'].iloc[0]
            plt.axvline(forecast_start, color='green', linestyle='--', linewidth=2, 
                       alpha=0.7, label='Forecast Start (Jan 2020)')
            
            # Formatting
            plt.xlabel('Date', fontsize=16, fontweight='bold')
            plt.ylabel('Deaths', fontsize=16, fontweight='bold')
            plt.title(f'Mortality Forecasting: SARIMA vs {model_name.upper()}', 
                     fontsize=20, fontweight='bold', pad=20)
            
            plt.xticks(fontsize=14, rotation=45)
            plt.yticks(fontsize=14)
            plt.legend(loc='upper left', fontsize=14, frameon=True, fancybox=True, shadow=True)
            plt.grid(True, alpha=0.3)
            
            # Add performance metrics text box
            sarima_test_rmse = np.mean([r['test_metrics']['RMSE'] for r in sarima_results])
            model_test_rmse = np.mean([r['test_metrics']['RMSE'] for r in model_results])
            
            metrics_text = f'Test RMSE:\nSARIMA: {sarima_test_rmse:.0f}\n{model_name.upper()}: {model_test_rmse:.0f}'
            plt.text(0.02, 0.98, metrics_text, transform=plt.gca().transAxes, fontsize=12,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            
            # Save plot
            plot_path = f"{self.results_dir}/figures/sarima_vs_{model_name}_comparison.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"    Saved: {plot_path}")
    
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
        
        # Create horizon comparison plots
        self.create_horizon_comparison_plots(horizon_results)
        
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
        fig.suptitle('Model Performance Across Forecast Horizons', fontsize=20, fontweight='bold')
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
        
        # Plot 1: RMSE vs Horizon
        ax = axes[0, 0]
        for i, model in enumerate(models):
            model_data = horizon_df[horizon_df['Model'] == model]
            rmse_means = []
            rmse_stds = []
            
            for h in horizons:
                h_data = model_data[model_data['Forecast_Horizon_Months'] == h]
                if len(h_data) > 0:
                    rmse_means.append(h_data['Test_RMSE_Mean'].iloc[0])
                    rmse_stds.append(h_data['Test_RMSE_Std'].iloc[0])
            
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
            coverage_means = []
            coverage_stds = []
            
            for h in horizons:
                h_data = model_data[model_data['Forecast_Horizon_Months'] == h]
                if len(h_data) > 0:
                    coverage_means.append(h_data['Coverage_Mean'].iloc[0])
                    coverage_stds.append(h_data['Coverage_Std'].iloc[0])
            
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
            width_means = []
            width_stds = []
            
            for h in horizons:
                h_data = model_data[model_data['Forecast_Horizon_Months'] == h]
                if len(h_data) > 0:
                    width_means.append(h_data['Interval_Width_Mean'].iloc[0])
                    width_stds.append(h_data['Interval_Width_Std'].iloc[0])
            
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
            mape_means = []
            mape_stds = []
            
            for h in horizons:
                h_data = model_data[model_data['Forecast_Horizon_Months'] == h]
                if len(h_data) > 0:
                    mape_means.append(h_data['Test_MAPE_Mean'].iloc[0])
                    mape_stds.append(h_data['Test_MAPE_Std'].iloc[0])
            
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
        
        print(f"  Variance analysis plot saved: {plot_path}")
    
    def create_horizon_comparison_plots(self, horizon_results: Dict):
        """Create comparison plots for each forecast horizon with proper alignment"""
        
        print("Creating horizon comparison plots...")
        
        for horizon_name, horizon_data in horizon_results.items():
            results = horizon_data['results']
            split_info = horizon_data['split_info']
            
            train_data = split_info['train']
            test_data = split_info['test']
            
            print(f"  Creating plot for {horizon_name}...")
            
            # Determine alignment based on maximum lookback
            max_lookback = max([OPTIMAL_PARAMS[model].get('lookback', 0) 
                               for model in results.keys() if model != 'sarima'])
            
            plot_start_idx = max_lookback
            plot_train_dates = train_data['Date'].iloc[plot_start_idx:].values
            plot_test_dates = test_data['Date'].values
            all_dates = np.concatenate([plot_train_dates, plot_test_dates])
            
            # Plot actual data from aligned start point
            plot_train_actual = train_data['Deaths'].iloc[plot_start_idx:].values
            plot_test_actual = test_data['Deaths'].values
            all_actual = np.concatenate([plot_train_actual, plot_test_actual])
            
            # Create plot for this horizon
            plt.figure(figsize=(16, 10))
            
            plt.plot(all_dates, all_actual, 'k-', linewidth=3, label='Actual Deaths', zorder=10)
            
            # Add forecast start line
            forecast_start = test_data['Date'].iloc[0]
            plt.axvline(forecast_start, color='gray', linestyle='--', linewidth=2, 
                       alpha=0.7, label='Forecast Start')
            
            # Plot each model's predictions
            model_colors = {'sarima': 'blue', 'lstm': 'red', 'tcn': 'green', 
                          'seq2seq': 'purple', 'transformer': 'orange'}
            
            for model_name, model_results in results.items():
                if not model_results:
                    continue
                
                color = model_colors.get(model_name, 'black')
                
                # Calculate average predictions
                avg_train_pred = np.mean([r['train_pred'] for r in model_results], axis=0)
                avg_test_pred = np.mean([r['test_pred'] for r in model_results], axis=0)
                
                # Handle alignment
                if model_name == 'sarima':
                    # Align SARIMA to start from same point as other models
                    model_train_pred = avg_train_pred[plot_start_idx:]
                else:
                    # Sequence models already aligned
                    model_train_pred = avg_train_pred
                
                all_model_pred = np.concatenate([model_train_pred, avg_test_pred])
                
                plt.plot(all_dates, all_model_pred, '-', color=color, linewidth=2,
                        label=f'{model_name.upper()}', alpha=0.8)
            
            # Formatting
            plt.xlabel('Date', fontsize=16, fontweight='bold')
            plt.ylabel('Deaths', fontsize=16, fontweight='bold')
            plt.title(f'Forecast Horizon: {split_info["description"]}', 
                     fontsize=18, fontweight='bold', pad=20)
            
            plt.xticks(fontsize=14, rotation=45)
            plt.yticks(fontsize=14)
            plt.legend(loc='upper left', fontsize=12, frameon=True, fancybox=True, shadow=True)
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = f"{self.results_dir}/figures/horizon_comparison_{horizon_name}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"    Saved: {plot_path}")
    
    def export_prediction_data(self, results: Dict, train_data: pd.DataFrame, 
                              test_data: pd.DataFrame, experiment_name: str):
        """Export detailed prediction data for later figure editing"""
        
        print("Exporting prediction data for later figure editing...")
        
        export_data = {}
        
        for model_name, model_results in results.items():
            if not model_results:
                continue
            
            # Calculate statistics across trials
            train_preds = np.array([r['train_pred'] for r in model_results])
            test_preds = np.array([r['test_pred'] for r in model_results])
            
            avg_train_pred = np.mean(train_preds, axis=0)
            std_train_pred = np.std(train_preds, axis=0)
            avg_test_pred = np.mean(test_preds, axis=0)
            std_test_pred = np.std(test_preds, axis=0)
            
            train_true = model_results[0]['train_true']
            test_true = model_results[0]['test_true']
            
            # Calculate prediction intervals for average predictions
            train_metrics = self.calculate_comprehensive_metrics(train_true, avg_train_pred)
            test_metrics = self.calculate_comprehensive_metrics(test_true, avg_test_pred)
            
            # Prepare dates
            if model_name != 'sarima':
                lookback = OPTIMAL_PARAMS[model_name].get('lookback', 0)
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
                test_preds = np.array([r['test_pred'] for r in model_results])
                avg_test_pred = np.mean(test_preds, axis=0)
                std_test_pred = np.std(test_preds, axis=0)
                test_true = model_results[0]['test_true']
                
                # Calculate metrics
                test_metrics = self.calculate_comprehensive_metrics(test_true, avg_test_pred)
                
                horizon_export[model_name] = {
                    'test_dates': split_info['test']['Date'].values,
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
        report_content.append("FIXED STREAMLINED EVALUATION RESULTS SUMMARY")
        report_content.append("=" * 60)
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
        report_content.append("ARCHITECTURAL IMPROVEMENTS:")
        report_content.append("-" * 25)
        report_content.append("✓ Fixed Seq2Seq architecture with proper attention mechanism")
        report_content.append("✓ Fixed Transformer architecture with correct scaling")
        report_content.append("✓ Proper plot alignment starting after lookback period")
        report_content.append("✓ Consistent model evaluation across different horizons")
        report_content.append("✓ Enhanced error handling and data scaling")
        
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
    print("FIXED STREAMLINED EVALUATION PIPELINE")
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
    pipeline = FixedEvaluationPipeline(DATA_PATH, RESULTS_DIR)
    
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
        models = ['sarima', 'lstm', 'tcn', 'seq2seq', 'transformer']
        print(f"\nModels to evaluate: {models}")
        
        # Step 3: Run Experiment 1
        print("\n" + "="*60)
        print("STEP 3: EXPERIMENT 1 - EXCESS MORTALITY ESTIMATION")
        print("="*60)
        
        exp1_results = pipeline.experiment_1_excess_mortality(data_splits, models)
        
        # Step 4: Run Experiment 2 (using models from Exp 1)
        print("\n" + "="*60)
        print("STEP 4: EXPERIMENT 2 - VARIANCE ANALYSIS")
        print("="*60)
        
        exp2_results = pipeline.experiment_2_variance_analysis(data_splits, models, exp1_results)
        
        # Step 5: Generate final report
        pipeline.generate_final_report()
        
        # Execution summary
        total_time = time.time() - start_time
        
        print("\n" + "🎉" * 20)
        print("FIXED STREAMLINED EVALUATION COMPLETED SUCCESSFULLY!")
        print("🎉" * 20)
        
        print(f"\nExecution Summary:")
        print(f"  Total time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        print(f"  Models evaluated: {len(models)}")
        print(f"  Trials per model: {TRIALS_PER_MODEL}")
        print(f"  Random seed: {RANDOM_SEED}")
        
        print(f"\n📁 Results saved to: {RESULTS_DIR}")
        print("\n📊 Key outputs:")
        print("  - Model comparison figures: figures/model_comparison_experiment_1.png")
        print("  - Individual comparisons: figures/sarima_vs_[model]_comparison.png")
        print("  - Variance analysis plot: figures/variance_analysis_across_horizons.png")
        print("  - Horizon comparisons: figures/horizon_comparison_[horizon].png")
        print("  - Summary statistics: experiment_1_excess_mortality/summary_statistics.csv")
        print("  - Horizon analysis: experiment_2_variance_analysis/horizon_summary.csv")
        print("  - Trained models: trained_models/[model]_best_model.pkl")
        print("  - Prediction data: data_exports/[experiment]_prediction_data.csv")
        
        print("\n✅ Key fixes implemented:")
        print("  - Fixed Seq2Seq and Transformer architectures")
        print("  - Proper plot alignment after lookback period")
        print("  - Enhanced data scaling for stability")
        print("  - Efficient model reuse between experiments")
        
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
