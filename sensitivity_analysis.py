#!/usr/bin/env python3
"""
Sensitivity Analysis Script for Model Convergence Studies

This script performs comprehensive sensitivity analysis to study convergence behavior:
1. Sensitivity to number of random seeds (convergence of confidence intervals)
2. Sensitivity to number of trials per seed (stability of point estimates)

The script is designed to be efficient by avoiding unnecessary recomputation and 
provides detailed convergence plots and statistics.

Usage:
    python sensitivity_analysis.py --model lstm --max_seeds 100 --max_trials 50 --seed_step 5 --trial_step 5

Parameters:
    --model: Model to analyze (lstm, tcn, seq2seq_attn, transformer, sarima)
    --max_seeds: Maximum number of random seeds to test (default: 100)
    --max_trials: Maximum number of trials per seed (default: 50)
    --seed_step: Step size for seed convergence analysis (default: 5)
    --trial_step: Step size for trial convergence analysis (default: 5)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Import required libraries for models
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, Flatten, GRU
from tensorflow.keras.layers import RepeatVector, Concatenate
from tensorflow.keras.layers import Input, Add, ReLU, Lambda, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention
from tensorflow.keras.models import Model
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.layers import Attention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import math

try:
    from tcn import TCN
except ImportError:
    print("Warning: TCN not available. Install with: pip install keras-tcn")
    TCN = None

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Configuration
OPTIMAL_PARAMS = {
    'sarima': {'order': (1, 0, 0), 'seasonal_order': (2, 2, 2, 12)},
    'lstm': {'lookback': 9, 'batch_size': 8, 'epochs': 100},
    'tcn': {'lookback': 5, 'batch_size': 32, 'epochs': 50},
    'seq2seq_attn': {'lookback': 11, 'batch_size': 16, 'epochs': 50, 'encoder_units': 128, 'decoder_units': 128},
    'transformer': {'lookback': 5, 'batch_size': 32, 'epochs': 100, 'd_model': 64, 'n_heads': 2}
}

DATA_PATH = 'data/state_month_overdose.xlsx'

# Model colors for plotting
MODEL_COLORS = {
    'sarima': '#2E86AB',
    'lstm': '#A23B72',
    'tcn': '#F18F01',
    'seq2seq_attn': '#C73E1D',
    'transformer': '#2D5016'
}

class SensitivityAnalyzer:
    def __init__(self, model_name, output_dir, max_seeds=100, max_trials=50, 
                 seed_step=5, trial_step=5):
        self.model_name = model_name
        self.output_dir = output_dir
        self.max_seeds = max_seeds
        self.max_trials = max_trials
        self.seed_step = seed_step
        self.trial_step = trial_step
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        self.plots_dir = os.path.join(self.output_dir, 'plots')
        self.data_dir = os.path.join(self.output_dir, 'data')
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Load and preprocess data
        self.load_data()
        
        # Storage for results
        self.all_results = {}
        self.convergence_data = {
            'seed_convergence': [],
            'trial_convergence': []
        }
        
        print(f"Initialized SensitivityAnalyzer for {model_name}")
        print(f"Max seeds: {max_seeds}, Max trials: {max_trials}")
        print(f"Seed step: {seed_step}, Trial step: {trial_step}")
        print(f"Output directory: {output_dir}")

    def load_data(self):
        """Load and preprocess the overdose data"""
        print("Loading and preprocessing data...")
        df = pd.read_excel(DATA_PATH)
        df['Deaths'] = df['Deaths'].apply(lambda x: 0 if x == 'Suppressed' else int(x))
        df['Month'] = pd.to_datetime(df['Month'])
        df = df.groupby('Month').agg({'Deaths': 'sum'}).reset_index()
        
        # Create train/validation/test splits
        train_end = '2019-01-01'
        val_end = '2020-01-01'
        
        self.train_data = df[df['Month'] < train_end]
        self.validation_data = df[(df['Month'] >= train_end) & (df['Month'] < val_end)]
        self.test_data = df[df['Month'] >= val_end]
        self.train_val_data = pd.concat([self.train_data, self.validation_data], ignore_index=True)
        
        print(f"Train+Val samples: {len(self.train_val_data)}")
        print(f"Test samples: {len(self.test_data)}")

    @staticmethod
    def create_dataset(series, look_back):
        """Create dataset for supervised learning"""
        X, y = [], []
        for i in range(len(series) - look_back):
            X.append(series[i:i+look_back])
            y.append(series[i+look_back])
        return np.array(X), np.array(y)

    @staticmethod
    def evaluate_metrics(y_true, y_pred):
        """Calculate evaluation metrics"""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        mse = mean_squared_error(y_true, y_pred)
        return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'MSE': mse}

    @staticmethod
    def calculate_prediction_intervals(actual, predictions, alpha=0.05):
        """Calculate prediction intervals"""
        residuals = actual - predictions
        std_residual = np.std(residuals)
        z_score = 1.96
        margin_of_error = z_score * std_residual
        lower_bound = predictions - margin_of_error
        upper_bound = predictions + margin_of_error
        return lower_bound, upper_bound

    @staticmethod
    def calculate_pi_coverage(actual, lower_bound, upper_bound):
        """Calculate prediction interval coverage"""
        coverage = np.mean((actual >= lower_bound) & (actual <= upper_bound))
        return coverage * 100

    def run_single_trial(self, seed, trial_idx):
        """Run a single trial for the specified model"""
        trial_seed = seed + trial_idx * 1000
        np.random.seed(trial_seed)
        tf.random.set_seed(trial_seed)
        
        params = OPTIMAL_PARAMS[self.model_name]
        
        try:
            if self.model_name == 'sarima':
                return self._run_sarima_trial(trial_seed, params)
            elif self.model_name == 'lstm':
                return self._run_lstm_trial(trial_seed, params)
            elif self.model_name == 'tcn':
                return self._run_tcn_trial(trial_seed, params)
            elif self.model_name == 'seq2seq_attn':
                return self._run_seq2seq_trial(trial_seed, params)
            elif self.model_name == 'transformer':
                return self._run_transformer_trial(trial_seed, params)
            else:
                raise ValueError(f"Unknown model: {self.model_name}")
                
        except Exception as e:
            print(f"Error in trial {trial_idx} with seed {seed}: {str(e)}")
            return None

    def _run_sarima_trial(self, seed, params):
        """Run SARIMA trial"""
        np.random.seed(seed)
        
        train_val_series = self.train_val_data['Deaths'].reset_index(drop=True).astype(float)
        test_series = self.test_data['Deaths'].reset_index(drop=True).astype(float)
        
        model = SARIMAX(train_val_series, 
                        order=params['order'], 
                        seasonal_order=params['seasonal_order'],
                        enforce_stationarity=False, 
                        enforce_invertibility=False)
        results = model.fit(disp=False)
        
        train_predictions = results.fittedvalues.values
        test_predictions = results.predict(start=len(train_val_series), 
                                          end=len(train_val_series) + len(test_series) - 1).values
        
        return {
            'train_true': train_val_series.values,
            'train_pred': train_predictions,
            'test_true': test_series.values,
            'test_pred': test_predictions
        }

    def _run_lstm_trial(self, seed, params):
        """Run LSTM trial"""
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        train_val_data = self.train_val_data['Deaths'].values
        test_data = self.test_data['Deaths'].values
        lookback = params['lookback']
        
        X_train, y_train = self.create_dataset(train_val_data, lookback)
        X_train = X_train.reshape((X_train.shape[0], lookback, 1))
        
        model = Sequential([
            LSTM(50, activation='relu', input_shape=(lookback, 1), return_sequences=True),
            LSTM(50, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=params['epochs'], 
                 batch_size=params['batch_size'], verbose=0)
        
        # Generate training predictions
        train_preds = []
        for i in range(lookback, len(train_val_data)):
            input_seq = train_val_data[i-lookback:i].reshape((1, lookback, 1))
            pred = model.predict(input_seq, verbose=0)[0][0]
            train_preds.append(pred)
        
        # Generate test predictions
        current_input = train_val_data[-lookback:].reshape((1, lookback, 1))
        test_preds = []
        for _ in range(len(test_data)):
            pred = model.predict(current_input, verbose=0)[0][0]
            test_preds.append(pred)
            current_input = np.append(current_input[:, 1:, :], [[[pred]]], axis=1)
        
        return {
            'train_true': train_val_data[lookback:],
            'train_pred': np.array(train_preds),
            'test_true': test_data,
            'test_pred': np.array(test_preds)
        }

    def _run_tcn_trial(self, seed, params):
        """Run TCN trial"""
        if TCN is None:
            raise ImportError("TCN not available")
            
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        train_val_data = self.train_val_data['Deaths'].values
        test_data = self.test_data['Deaths'].values
        lookback = params['lookback']
        
        X_train, y_train = self.create_dataset(train_val_data, lookback)
        X_train = X_train.reshape((X_train.shape[0], lookback, 1))
        
        model = Sequential([
            TCN(input_shape=(lookback, 1), dilations=[1, 2, 4, 8], 
                nb_filters=64, kernel_size=3, dropout_rate=0.1),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=params['epochs'], 
                 batch_size=params['batch_size'], verbose=0)
        
        # Generate predictions (similar to LSTM)
        train_preds = []
        for i in range(lookback, len(train_val_data)):
            input_seq = train_val_data[i-lookback:i].reshape((1, lookback, 1))
            pred = model.predict(input_seq, verbose=0)[0][0]
            train_preds.append(pred)
        
        current_input = train_val_data[-lookback:].reshape((1, lookback, 1))
        test_preds = []
        for _ in range(len(test_data)):
            pred = model.predict(current_input, verbose=0)[0][0]
            test_preds.append(pred)
            current_input = np.append(current_input[:, 1:, :], [[[pred]]], axis=1)
        
        return {
            'train_true': train_val_data[lookback:],
            'train_pred': np.array(train_preds),
            'test_true': test_data,
            'test_pred': np.array(test_preds)
        }

    def _run_seq2seq_trial(self, seed, params):
        """Run Seq2Seq with Attention trial"""
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        train_val_data = self.train_val_data['Deaths'].values
        test_data = self.test_data['Deaths'].values
        lookback = params['lookback']
        
        # Scaling
        full_series = np.concatenate([train_val_data, test_data])
        scaler = MinMaxScaler()
        scaled_full = scaler.fit_transform(full_series.reshape(-1, 1)).flatten()
        
        train_val_scaled = scaled_full[:len(train_val_data)]
        test_scaled = scaled_full[len(train_val_data):]
        
        X_train, y_train = self.create_dataset(train_val_scaled, lookback)
        X_train = X_train.reshape((X_train.shape[0], lookback, 1))
        decoder_input_train = np.zeros((X_train.shape[0], 1, 1))
        y_train = y_train.reshape((-1, 1, 1))
        
        # Build seq2seq model with attention
        encoder_inputs = Input(shape=(lookback, 1))
        encoder_gru = GRU(params['encoder_units'], return_sequences=True, return_state=True)
        encoder_outputs, encoder_state = encoder_gru(encoder_inputs)
        
        decoder_inputs = Input(shape=(1, 1))
        decoder_gru = GRU(params['decoder_units'], return_sequences=True, return_state=True)
        decoder_outputs, _ = decoder_gru(decoder_inputs, initial_state=encoder_state)
        
        attention_layer = Attention()
        context_vector = attention_layer([decoder_outputs, encoder_outputs])
        decoder_combined = Concatenate(axis=-1)([decoder_outputs, context_vector])
        decoder_hidden = Dense(params['decoder_units'], activation='relu')(decoder_combined)
        decoder_outputs = Dense(1)(decoder_hidden)
        
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        model.compile(optimizer=Adam(learning_rate=0.001, clipnorm=1.0), loss='mse')
        
        early_stopping = EarlyStopping(monitor='loss', patience=15, restore_best_weights=True)
        model.fit([X_train, decoder_input_train], y_train, epochs=params['epochs'], 
                 batch_size=params['batch_size'], verbose=0, callbacks=[early_stopping])
        
        # Generate predictions
        train_preds_scaled = []
        for i in range(lookback, len(train_val_data)):
            encoder_input = train_val_scaled[i-lookback:i].reshape((1, lookback, 1))
            decoder_input = np.zeros((1, 1, 1))
            pred_scaled = model.predict([encoder_input, decoder_input], verbose=0)[0, 0, 0]
            train_preds_scaled.append(pred_scaled)
        
        test_preds_scaled = []
        current_sequence = train_val_scaled[-lookback:].copy()
        for _ in range(len(test_data)):
            encoder_input = current_sequence.reshape((1, lookback, 1))
            decoder_input = np.zeros((1, 1, 1))
            pred_scaled = model.predict([encoder_input, decoder_input], verbose=0)[0, 0, 0]
            test_preds_scaled.append(pred_scaled)
            current_sequence = np.append(current_sequence[1:], pred_scaled)
        
        # Inverse transform
        train_preds = scaler.inverse_transform(np.array(train_preds_scaled).reshape(-1, 1)).flatten()
        test_preds = scaler.inverse_transform(np.array(test_preds_scaled).reshape(-1, 1)).flatten()
        
        return {
            'train_true': train_val_data[lookback:],
            'train_pred': train_preds,
            'test_true': test_data,
            'test_pred': test_preds
        }

    def _run_transformer_trial(self, seed, params):
        """Run Transformer trial"""
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        train_val_data = self.train_val_data['Deaths'].values
        test_data = self.test_data['Deaths'].values
        lookback = params['lookback']
        
        # Scaling
        full_series = np.concatenate([train_val_data, test_data])
        scaler = MinMaxScaler()
        scaled_full = scaler.fit_transform(full_series.reshape(-1, 1)).flatten()
        
        train_val_scaled = scaled_full[:len(train_val_data)]
        
        X_train, y_train = self.create_dataset(train_val_scaled, lookback)
        X_train = X_train.reshape((X_train.shape[0], lookback, 1))
        y_train = y_train.reshape((-1, 1))
        
        # Build transformer model
        class PositionalEncoding(tf.keras.layers.Layer):
            def __init__(self, d_model, max_len=5000):
                super().__init__()
                pe = np.zeros((max_len, d_model))
                for pos in range(max_len):
                    for i in range(0, d_model, 2):
                        pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                        if i+1 < d_model:
                            pe[pos, i+1] = math.cos(pos / (10000 ** ((2 * (i+1))/d_model)))
                self.pe = tf.constant(pe[np.newaxis, :, :], dtype=tf.float32)

            def call(self, x):
                return x + self.pe[:, :tf.shape(x)[1], :]
        
        inputs = Input(shape=(lookback, 1))
        x = Dense(params['d_model'])(inputs)
        x = PositionalEncoding(params['d_model'])(x)
        
        attn_output = MultiHeadAttention(num_heads=params['n_heads'], key_dim=params['d_model'])(x, x)
        x = Add()([x, attn_output])
        x = LayerNormalization()(x)
        
        x = Flatten()(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.1)(x)
        outputs = Dense(1)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, batch_size=params['batch_size'], 
                 epochs=params['epochs'], verbose=0)
        
        # Generate predictions
        train_preds_scaled = []
        for i in range(lookback, len(train_val_data)):
            input_seq = train_val_scaled[i-lookback:i].reshape((1, lookback, 1))
            pred_scaled = model.predict(input_seq, verbose=0)[0][0]
            train_preds_scaled.append(pred_scaled)
        
        current_seq = train_val_scaled[-lookback:].copy()
        test_preds_scaled = []
        for _ in range(len(test_data)):
            input_seq = current_seq.reshape((1, lookback, 1))
            pred_scaled = model.predict(input_seq, verbose=0)[0][0]
            test_preds_scaled.append(pred_scaled)
            current_seq = np.append(current_seq[1:], pred_scaled)
        
        # Inverse transform
        train_preds = scaler.inverse_transform(np.array(train_preds_scaled).reshape(-1, 1)).flatten()
        test_preds = scaler.inverse_transform(np.array(test_preds_scaled).reshape(-1, 1)).flatten()
        
        return {
            'train_true': train_val_data[lookback:],
            'train_pred': train_preds,
            'test_true': test_data,
            'test_pred': test_preds
        }

    def calculate_metrics_from_trials(self, trial_results):
        """Calculate aggregated metrics from trial results"""
        if not trial_results:
            return None
            
        # Aggregate predictions
        train_preds = [r['train_pred'] for r in trial_results]
        test_preds = [r['test_pred'] for r in trial_results]
        
        # Calculate mean predictions
        mean_train_pred = np.mean(train_preds, axis=0)
        mean_test_pred = np.mean(test_preds, axis=0)
        
        # Get true values (same across trials)
        train_true = trial_results[0]['train_true']
        test_true = trial_results[0]['test_true']
        
        # Calculate metrics
        train_metrics = self.evaluate_metrics(train_true, mean_train_pred)
        test_metrics = self.evaluate_metrics(test_true, mean_test_pred)
        
        # Calculate prediction intervals
        train_lower, train_upper = self.calculate_prediction_intervals(train_true, mean_train_pred)
        test_lower, test_upper = self.calculate_prediction_intervals(test_true, mean_test_pred)
        
        # Calculate coverage
        train_coverage = self.calculate_pi_coverage(train_true, train_lower, train_upper)
        test_coverage = self.calculate_pi_coverage(test_true, test_lower, test_upper)
        
        # Calculate prediction uncertainty (std across trials)
        train_uncertainty = np.mean(np.std(train_preds, axis=0))
        test_uncertainty = np.mean(np.std(test_preds, axis=0))
        
        return {
            'train_rmse': train_metrics['RMSE'],
            'train_mae': train_metrics['MAE'],
            'train_mape': train_metrics['MAPE'],
            'train_mse': train_metrics['MSE'],
            'train_coverage': train_coverage,
            'test_rmse': test_metrics['RMSE'],
            'test_mae': test_metrics['MAE'],
            'test_mape': test_metrics['MAPE'],
            'test_mse': test_metrics['MSE'],
            'test_coverage': test_coverage,
            'train_uncertainty': train_uncertainty,
            'test_uncertainty': test_uncertainty,
            'num_trials': len(trial_results)
        }

    def run_seed_sensitivity_analysis(self):
        """Run sensitivity analysis for number of seeds"""
        print(f"\n{'='*60}")
        print("SEED SENSITIVITY ANALYSIS")
        print(f"{'='*60}")
        
        # Generate all seeds upfront
        base_seeds = list(range(42, 42 + self.max_seeds))
        
        # Run all trials for all seeds
        print(f"Running trials for {self.max_seeds} seeds with {self.max_trials} trials each...")
        
        all_seed_data = {}
        
        for seed_idx, seed in enumerate(base_seeds):
            print(f"\nProcessing seed {seed_idx + 1}/{len(base_seeds)} (seed={seed})")
            
            seed_trials = []
            for trial_idx in range(self.max_trials):
                if (trial_idx + 1) % 10 == 0:
                    print(f"  Trial {trial_idx + 1}/{self.max_trials}")
                
                result = self.run_single_trial(seed, trial_idx)
                if result is not None:
                    seed_trials.append(result)
            
            all_seed_data[seed] = seed_trials
            print(f"  Completed {len(seed_trials)} successful trials for seed {seed}")
        
        # Analyze convergence at different seed counts
        seed_convergence_results = []
        
        seed_counts = list(range(self.seed_step, self.max_seeds + 1, self.seed_step))
        
        print(f"\nAnalyzing convergence for seed counts: {seed_counts}")
        
        for num_seeds in seed_counts:
            print(f"  Analyzing with {num_seeds} seeds...")
            
            # Use first num_seeds seeds
            selected_seeds = base_seeds[:num_seeds]
            
            # Collect all trials from selected seeds
            all_trials = []
            for seed in selected_seeds:
                all_trials.extend(all_seed_data[seed])
            
            # Calculate metrics
            metrics = self.calculate_metrics_from_trials(all_trials)
            
            if metrics is not None:
                metrics['num_seeds'] = num_seeds
                metrics['total_trials'] = len(all_trials)
                seed_convergence_results.append(metrics)
                
                print(f"    {num_seeds} seeds: Test RMSE = {metrics['test_rmse']:.4f}, "
                      f"Test MAPE = {metrics['test_mape']:.2f}%, Trials = {len(all_trials)}")
        
        # Save results
        seed_df = pd.DataFrame(seed_convergence_results)
        seed_df.to_csv(os.path.join(self.data_dir, 'seed_convergence_analysis.csv'), index=False)
        
        self.convergence_data['seed_convergence'] = seed_convergence_results
        
        # Save all seed data for future use
        with open(os.path.join(self.data_dir, 'all_seed_trial_data.pkl'), 'wb') as f:
            pickle.dump(all_seed_data, f)
        
        print(f"\nSeed sensitivity analysis complete. Results saved to {self.data_dir}")
        
        return seed_convergence_results

    def run_trial_sensitivity_analysis(self):
        """Run sensitivity analysis for number of trials per seed"""
        print(f"\n{'='*60}")
        print("TRIAL SENSITIVITY ANALYSIS")
        print(f"{'='*60}")
        
        # Load seed data if exists, otherwise run with subset of seeds
        seed_data_path = os.path.join(self.data_dir, 'all_seed_trial_data.pkl')
        
        if os.path.exists(seed_data_path):
            print("Loading existing seed data...")
            with open(seed_data_path, 'rb') as f:
                all_seed_data = pickle.load(f)
        else:
            print("No existing seed data found. Running with 20 seeds...")
            # Run with subset of seeds for trial analysis
            base_seeds = list(range(42, 42 + 20))
            all_seed_data = {}
            
            for seed in base_seeds:
                print(f"  Processing seed {seed}...")
                seed_trials = []
                for trial_idx in range(self.max_trials):
                    result = self.run_single_trial(seed, trial_idx)
                    if result is not None:
                        seed_trials.append(result)
                all_seed_data[seed] = seed_trials
        
        # Analyze convergence at different trial counts
        trial_convergence_results = []
        
        trial_counts = list(range(self.trial_step, self.max_trials + 1, self.trial_step))
        
        print(f"\nAnalyzing convergence for trial counts: {trial_counts}")
        
        for num_trials in trial_counts:
            print(f"  Analyzing with {num_trials} trials per seed...")
            
            # Collect first num_trials from each seed
            all_trials = []
            for seed, seed_trials in all_seed_data.items():
                if len(seed_trials) >= num_trials:
                    all_trials.extend(seed_trials[:num_trials])
            
            # Calculate metrics
            metrics = self.calculate_metrics_from_trials(all_trials)
            
            if metrics is not None:
                metrics['num_trials_per_seed'] = num_trials
                metrics['num_seeds_used'] = len(all_seed_data)
                metrics['total_trials'] = len(all_trials)
                trial_convergence_results.append(metrics)
                
                print(f"    {num_trials} trials/seed: Test RMSE = {metrics['test_rmse']:.4f}, "
                      f"Test MAPE = {metrics['test_mape']:.2f}%, Total trials = {len(all_trials)}")
        
        # Save results
        trial_df = pd.DataFrame(trial_convergence_results)
        trial_df.to_csv(os.path.join(self.data_dir, 'trial_convergence_analysis.csv'), index=False)
        
        self.convergence_data['trial_convergence'] = trial_convergence_results
        
        print(f"\nTrial sensitivity analysis complete. Results saved to {self.data_dir}")
        
        return trial_convergence_results

    def create_convergence_plots(self):
        """Create comprehensive convergence plots"""
        print(f"\n{'='*60}")
        print("CREATING CONVERGENCE PLOTS")
        print(f"{'='*60}")
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8-whitegrid')
        color = MODEL_COLORS.get(self.model_name, '#2E86AB')
        
        # 1. Seed Convergence Plot
        if self.convergence_data['seed_convergence']:
            self._create_seed_convergence_plot(color)
        
        # 2. Trial Convergence Plot
        if self.convergence_data['trial_convergence']:
            self._create_trial_convergence_plot(color)
        
        # 3. Combined Convergence Plot
        if (self.convergence_data['seed_convergence'] and 
            self.convergence_data['trial_convergence']):
            self._create_combined_convergence_plot(color)
        
        # 4. Confidence Interval Convergence
        if self.convergence_data['seed_convergence']:
            self._create_confidence_interval_plot(color)
        
        # 5. Uncertainty Analysis Plot
        if self.convergence_data['seed_convergence']:
            self._create_uncertainty_analysis_plot(color)
        
        print(f"All convergence plots saved to {self.plots_dir}")

    def _create_seed_convergence_plot(self, color):
        """Create seed convergence analysis plot"""
        data = pd.DataFrame(self.convergence_data['seed_convergence'])
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{self.model_name.upper()} - Convergence Analysis: Number of Seeds', 
                     fontsize=16, fontweight='bold')
        
        metrics = ['test_rmse', 'test_mae', 'test_mape', 'test_coverage', 'test_uncertainty']
        titles = ['Test RMSE', 'Test MAE', 'Test MAPE (%)', 'Test PI Coverage (%)', 'Test Uncertainty']
        
        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]
            
            ax.plot(data['num_seeds'], data[metric], marker='o', linewidth=2, 
                   markersize=4, color=color, alpha=0.8)
            ax.set_xlabel('Number of Seeds', fontsize=10)
            ax.set_ylabel(title, fontsize=10)
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add convergence info
            if len(data) > 1:
                final_val = data[metric].iloc[-1]
                std_last_few = data[metric].iloc[-min(3, len(data)):].std()
                ax.text(0.02, 0.98, f'Final: {final_val:.4f}\nStd: {std_last_few:.4f}', 
                       transform=ax.transAxes, fontsize=8, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Remove empty subplot
        if len(metrics) < 6:
            axes[1, 2].remove()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, f'{self.model_name}_seed_convergence.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def _create_trial_convergence_plot(self, color):
        """Create trial convergence analysis plot"""
        data = pd.DataFrame(self.convergence_data['trial_convergence'])
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{self.model_name.upper()} - Convergence Analysis: Number of Trials per Seed', 
                     fontsize=16, fontweight='bold')
        
        metrics = ['test_rmse', 'test_mae', 'test_mape', 'test_coverage', 'test_uncertainty']
        titles = ['Test RMSE', 'Test MAE', 'Test MAPE (%)', 'Test PI Coverage (%)', 'Test Uncertainty']
        
        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]
            
            ax.plot(data['num_trials_per_seed'], data[metric], marker='s', linewidth=2, 
                   markersize=4, color=color, alpha=0.8)
            ax.set_xlabel('Number of Trials per Seed', fontsize=10)
            ax.set_ylabel(title, fontsize=10)
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add convergence info
            if len(data) > 1:
                final_val = data[metric].iloc[-1]
                std_last_few = data[metric].iloc[-min(3, len(data)):].std()
                ax.text(0.02, 0.98, f'Final: {final_val:.4f}\nStd: {std_last_few:.4f}', 
                       transform=ax.transAxes, fontsize=8, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Remove empty subplot
        if len(metrics) < 6:
            axes[1, 2].remove()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, f'{self.model_name}_trial_convergence.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def _create_combined_convergence_plot(self, color):
        """Create combined convergence plot comparing seed vs trial effects"""
        seed_data = pd.DataFrame(self.convergence_data['seed_convergence'])
        trial_data = pd.DataFrame(self.convergence_data['trial_convergence'])
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{self.model_name.upper()} - Combined Convergence Analysis', 
                     fontsize=16, fontweight='bold')
        
        metrics = ['test_rmse', 'test_mae', 'test_mape', 'test_coverage']
        titles = ['Test RMSE', 'Test MAE', 'Test MAPE (%)', 'Test PI Coverage (%)']
        
        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            row = idx // 2
            col = idx % 2
            ax = axes[row, col]
            
            # Plot seed convergence
            ax.plot(seed_data['num_seeds'], seed_data[metric], 
                   marker='o', linewidth=2, markersize=4, color=color, 
                   alpha=0.8, label='Seeds')
            
            # Plot trial convergence (normalize x-axis)
            max_seeds = seed_data['num_seeds'].max()
            max_trials = trial_data['num_trials_per_seed'].max()
            normalized_trials = trial_data['num_trials_per_seed'] * (max_seeds / max_trials)
            
            ax2 = ax.twiny()
            ax2.plot(normalized_trials, trial_data[metric], 
                    marker='s', linewidth=2, markersize=4, color='red', 
                    alpha=0.8, label='Trials/Seed')
            
            ax.set_xlabel('Number of Seeds', fontsize=10)
            ax2.set_xlabel('Trials per Seed (normalized)', fontsize=10, color='red')
            ax.set_ylabel(title, fontsize=10)
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Legends
            ax.legend(loc='upper right')
            ax2.legend(loc='upper left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, f'{self.model_name}_combined_convergence.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def _create_confidence_interval_plot(self, color):
        """Create confidence interval convergence plot"""
        data = pd.DataFrame(self.convergence_data['seed_convergence'])
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{self.model_name.upper()} - Confidence Interval Convergence', 
                     fontsize=16, fontweight='bold')
        
        metrics = ['test_rmse', 'test_mae', 'test_mape', 'test_coverage']
        titles = ['Test RMSE', 'Test MAE', 'Test MAPE (%)', 'Test PI Coverage (%)']
        
        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            row = idx // 2
            col = idx % 2
            ax = axes[row, col]
            
            # Calculate rolling statistics
            values = data[metric].values
            means = []
            stds = []
            
            for i in range(len(values)):
                window = values[:i+1]
                means.append(np.mean(window))
                stds.append(np.std(window))
            
            means = np.array(means)
            stds = np.array(stds)
            
            # Plot mean with confidence intervals
            ax.plot(data['num_seeds'], means, color=color, linewidth=2, label='Mean')
            ax.fill_between(data['num_seeds'], means - 1.96*stds, means + 1.96*stds, 
                           color=color, alpha=0.3, label='95% CI')
            
            ax.set_xlabel('Number of Seeds', fontsize=10)
            ax.set_ylabel(title, fontsize=10)
            ax.set_title(f'{title} - Mean ± 95% CI', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Add convergence assessment
            if len(stds) > 3:
                final_std = stds[-1]
                cv = final_std / abs(means[-1]) if means[-1] != 0 else float('inf')
                ax.text(0.02, 0.98, f'Final CV: {cv:.4f}', 
                       transform=ax.transAxes, fontsize=8, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, f'{self.model_name}_confidence_intervals.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def _create_uncertainty_analysis_plot(self, color):
        """Create uncertainty analysis plot"""
        data = pd.DataFrame(self.convergence_data['seed_convergence'])
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{self.model_name.upper()} - Uncertainty Analysis', 
                     fontsize=16, fontweight='bold')
        
        # Plot 1: Test uncertainty vs number of seeds
        ax1 = axes[0, 0]
        ax1.plot(data['num_seeds'], data['test_uncertainty'], 
                marker='o', color=color, linewidth=2, markersize=4)
        ax1.set_xlabel('Number of Seeds')
        ax1.set_ylabel('Test Prediction Uncertainty')
        ax1.set_title('Prediction Uncertainty vs Seeds')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Coverage vs uncertainty
        ax2 = axes[0, 1]
        ax2.scatter(data['test_uncertainty'], data['test_coverage'], 
                   color=color, alpha=0.7, s=50)
        ax2.set_xlabel('Test Prediction Uncertainty')
        ax2.set_ylabel('Test PI Coverage (%)')
        ax2.set_title('Coverage vs Uncertainty')
        ax2.grid(True, alpha=0.3)
        
        # Add trend line
        if len(data) > 2:
            z = np.polyfit(data['test_uncertainty'], data['test_coverage'], 1)
            p = np.poly1d(z)
            ax2.plot(data['test_uncertainty'], p(data['test_uncertainty']), 
                    "r--", alpha=0.8, linewidth=1)
        
        # Plot 3: RMSE vs number of total trials
        ax3 = axes[1, 0]
        ax3.plot(data['total_trials'], data['test_rmse'], 
                marker='o', color=color, linewidth=2, markersize=4)
        ax3.set_xlabel('Total Number of Trials')
        ax3.set_ylabel('Test RMSE')
        ax3.set_title('RMSE vs Total Trials')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Coefficient of variation over seeds
        ax4 = axes[1, 1]
        
        # Calculate rolling CV for RMSE
        rmse_values = data['test_rmse'].values
        cvs = []
        
        for i in range(1, len(rmse_values)):
            window = rmse_values[:i+1]
            cv = np.std(window) / np.mean(window) if np.mean(window) != 0 else 0
            cvs.append(cv)
        
        if cvs:
            ax4.plot(data['num_seeds'].values[1:], cvs, 
                    marker='o', color=color, linewidth=2, markersize=4)
            ax4.set_xlabel('Number of Seeds')
            ax4.set_ylabel('Coefficient of Variation (RMSE)')
            ax4.set_title('RMSE Stability (CV)')
            ax4.grid(True, alpha=0.3)
            
            # Add horizontal line at 5% CV (good convergence threshold)
            ax4.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, 
                       label='5% CV threshold')
            ax4.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, f'{self.model_name}_uncertainty_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def create_summary_report(self):
        """Create a comprehensive summary report"""
        print(f"\n{'='*60}")
        print("CREATING SUMMARY REPORT")
        print(f"{'='*60}")
        
        # Create summary statistics
        summary_stats = {}
        
        if self.convergence_data['seed_convergence']:
            seed_data = pd.DataFrame(self.convergence_data['seed_convergence'])
            
            # Final convergence values
            final_metrics = seed_data.iloc[-1]
            
            # Convergence assessment (CV of last few points)
            last_n = min(5, len(seed_data))
            rmse_cv = seed_data['test_rmse'].iloc[-last_n:].std() / seed_data['test_rmse'].iloc[-last_n:].mean()
            mae_cv = seed_data['test_mae'].iloc[-last_n:].std() / seed_data['test_mae'].iloc[-last_n:].mean()
            
            summary_stats['seed_analysis'] = {
                'max_seeds_tested': int(final_metrics['num_seeds']),
                'total_trials_final': int(final_metrics['total_trials']),
                'final_test_rmse': final_metrics['test_rmse'],
                'final_test_mae': final_metrics['test_mae'],
                'final_test_mape': final_metrics['test_mape'],
                'final_test_coverage': final_metrics['test_coverage'],
                'rmse_coefficient_variation': rmse_cv,
                'mae_coefficient_variation': mae_cv,
                'convergence_assessment': 'Good' if rmse_cv < 0.05 else 'Moderate' if rmse_cv < 0.1 else 'Poor'
            }
        
        if self.convergence_data['trial_convergence']:
            trial_data = pd.DataFrame(self.convergence_data['trial_convergence'])
            
            final_trial_metrics = trial_data.iloc[-1]
            
            # Trial stability assessment
            last_n = min(3, len(trial_data))
            trial_rmse_cv = trial_data['test_rmse'].iloc[-last_n:].std() / trial_data['test_rmse'].iloc[-last_n:].mean()
            
            summary_stats['trial_analysis'] = {
                'max_trials_per_seed': int(final_trial_metrics['num_trials_per_seed']),
                'num_seeds_used': int(final_trial_metrics['num_seeds_used']),
                'final_test_rmse': final_trial_metrics['test_rmse'],
                'final_test_mae': final_trial_metrics['test_mae'],
                'final_test_mape': final_trial_metrics['test_mape'],
                'trial_rmse_cv': trial_rmse_cv,
                'trial_stability': 'Stable' if trial_rmse_cv < 0.02 else 'Moderate' if trial_rmse_cv < 0.05 else 'Unstable'
            }
        
        # Save summary statistics
        summary_df = pd.DataFrame(summary_stats)
        summary_df.to_csv(os.path.join(self.data_dir, 'sensitivity_summary.csv'))
        
        # Create text report
        report_path = os.path.join(self.output_dir, f'{self.model_name}_sensitivity_report.txt')
        
        with open(report_path, 'w') as f:
            f.write(f"SENSITIVITY ANALYSIS REPORT\n")
            f.write(f"{'='*50}\n\n")
            f.write(f"Model: {self.model_name.upper()}\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Output Directory: {self.output_dir}\n\n")
            
            if 'seed_analysis' in summary_stats:
                f.write(f"SEED SENSITIVITY ANALYSIS\n")
                f.write(f"{'-'*30}\n")
                seed_stats = summary_stats['seed_analysis']
                f.write(f"Maximum seeds tested: {seed_stats['max_seeds_tested']}\n")
                f.write(f"Total trials: {seed_stats['total_trials_final']}\n")
                f.write(f"Final Test RMSE: {seed_stats['final_test_rmse']:.6f}\n")
                f.write(f"Final Test MAE: {seed_stats['final_test_mae']:.6f}\n")
                f.write(f"Final Test MAPE: {seed_stats['final_test_mape']:.4f}%\n")
                f.write(f"Final Test Coverage: {seed_stats['final_test_coverage']:.2f}%\n")
                f.write(f"RMSE Coefficient of Variation: {seed_stats['rmse_coefficient_variation']:.6f}\n")
                f.write(f"MAE Coefficient of Variation: {seed_stats['mae_coefficient_variation']:.6f}\n")
                f.write(f"Convergence Assessment: {seed_stats['convergence_assessment']}\n\n")
            
            if 'trial_analysis' in summary_stats:
                f.write(f"TRIAL SENSITIVITY ANALYSIS\n")
                f.write(f"{'-'*30}\n")
                trial_stats = summary_stats['trial_analysis']
                f.write(f"Maximum trials per seed: {trial_stats['max_trials_per_seed']}\n")
                f.write(f"Number of seeds used: {trial_stats['num_seeds_used']}\n")
                f.write(f"Final Test RMSE: {trial_stats['final_test_rmse']:.6f}\n")
                f.write(f"Final Test MAE: {trial_stats['final_test_mae']:.6f}\n")
                f.write(f"Final Test MAPE: {trial_stats['final_test_mape']:.4f}%\n")
                f.write(f"Trial RMSE CV: {trial_stats['trial_rmse_cv']:.6f}\n")
                f.write(f"Trial Stability: {trial_stats['trial_stability']}\n\n")
            
            f.write(f"RECOMMENDATIONS\n")
            f.write(f"{'-'*15}\n")
            
            if 'seed_analysis' in summary_stats:
                seed_stats = summary_stats['seed_analysis']
                if seed_stats['rmse_coefficient_variation'] < 0.05:
                    f.write(f"✓ Seed convergence is good. Current number of seeds ({seed_stats['max_seeds_tested']}) is sufficient.\n")
                elif seed_stats['rmse_coefficient_variation'] < 0.1:
                    f.write(f"⚠ Seed convergence is moderate. Consider increasing seeds to {seed_stats['max_seeds_tested'] * 2}.\n")
                else:
                    f.write(f"✗ Poor seed convergence. Recommend significantly more seeds (>= {seed_stats['max_seeds_tested'] * 3}).\n")
            
            if 'trial_analysis' in summary_stats:
                trial_stats = summary_stats['trial_analysis']
                if trial_stats['trial_rmse_cv'] < 0.02:
                    f.write(f"✓ Trial stability is excellent. {trial_stats['max_trials_per_seed']} trials per seed is sufficient.\n")
                elif trial_stats['trial_rmse_cv'] < 0.05:
                    f.write(f"⚠ Trial stability is moderate. Consider {trial_stats['max_trials_per_seed'] + 10} trials per seed.\n")
                else:
                    f.write(f"✗ Trial instability detected. Recommend >= {trial_stats['max_trials_per_seed'] + 20} trials per seed.\n")
        
        print(f"Summary report saved to: {report_path}")
        
        return summary_stats

    def run_full_analysis(self):
        """Run the complete sensitivity analysis"""
        print(f"\n{'='*80}")
        print(f"STARTING COMPREHENSIVE SENSITIVITY ANALYSIS FOR {self.model_name.upper()}")
        print(f"{'='*80}")
        
        start_time = datetime.now()
        print(f"Analysis started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # 1. Run seed sensitivity analysis
            seed_results = self.run_seed_sensitivity_analysis()
            
            # 2. Run trial sensitivity analysis
            trial_results = self.run_trial_sensitivity_analysis()
            
            # 3. Create all plots
            self.create_convergence_plots()
            
            # 4. Create summary report
            summary_stats = self.create_summary_report()
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            print(f"\n{'='*80}")
            print(f"SENSITIVITY ANALYSIS COMPLETED SUCCESSFULLY")
            print(f"{'='*80}")
            print(f"Duration: {duration}")
            print(f"Model: {self.model_name.upper()}")
            print(f"Output directory: {self.output_dir}")
            print(f"\nGenerated files:")
            print(f"├── data/")
            print(f"│   ├── seed_convergence_analysis.csv")
            print(f"│   ├── trial_convergence_analysis.csv")
            print(f"│   ├── sensitivity_summary.csv")
            print(f"│   └── all_seed_trial_data.pkl")
            print(f"├── plots/")
            print(f"│   ├── {self.model_name}_seed_convergence.png")
            print(f"│   ├── {self.model_name}_trial_convergence.png")
            print(f"│   ├── {self.model_name}_combined_convergence.png")
            print(f"│   ├── {self.model_name}_confidence_intervals.png")
            print(f"│   └── {self.model_name}_uncertainty_analysis.png")
            print(f"└── {self.model_name}_sensitivity_report.txt")
            
            return True
            
        except Exception as e:
            print(f"\nError during sensitivity analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(description='Sensitivity Analysis for Time Series Models')
    
    parser.add_argument('--model', type=str, default='lstm',
                       choices=['lstm', 'tcn', 'seq2seq_attn', 'transformer', 'sarima'],
                       help='Model to analyze (default: lstm)')
    parser.add_argument('--max_seeds', type=int, default=100,
                       help='Maximum number of random seeds (default: 100)')
    parser.add_argument('--max_trials', type=int, default=50,
                       help='Maximum number of trials per seed (default: 50)')
    parser.add_argument('--seed_step', type=int, default=5,
                       help='Step size for seed analysis (default: 5)')
    parser.add_argument('--trial_step', type=int, default=5,
                       help='Step size for trial analysis (default: 5)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: sensitivity_analysis_[model]_[timestamp])')
    
    args = parser.parse_args()
    
    # Create output directory if not specified
    if args.output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = f'sensitivity_analysis_{args.model}_{timestamp}'
    
    # Check if data file exists
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found: {DATA_PATH}")
        print("Please ensure the data file exists before running the analysis.")
        return False
    
    # Check model-specific requirements
    if args.model == 'tcn' and TCN is None:
        print("Error: TCN model requires keras-tcn package.")
        print("Install with: pip install keras-tcn")
        return False
    
    # Initialize and run analysis
    analyzer = SensitivityAnalyzer(
        model_name=args.model,
        output_dir=args.output_dir,
        max_seeds=args.max_seeds,
        max_trials=args.max_trials,
        seed_step=args.seed_step,
        trial_step=args.trial_step
    )
    
    success = analyzer.run_full_analysis()
    
    if success:
        print(f"\n🎉 Analysis completed successfully!")
        print(f"📁 Results saved to: {args.output_dir}")
        print(f"📊 Check the plots directory for visualizations")
        print(f"📋 Read the summary report for recommendations")
    else:
        print(f"\n❌ Analysis failed. Check the error messages above.")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
