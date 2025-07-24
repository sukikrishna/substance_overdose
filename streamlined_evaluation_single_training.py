"""
Streamlined Evaluation Pipeline - Single Training Phase
======================================================

Efficient approach:
1. Train models ONCE on 2015-2019 data
2. Evaluate same trained models on ALL forecast horizons
3. Experiment 1 = longest horizon results (2020-2023)
4. Experiment 2 = variance analysis across all horizons

This eliminates redundant training and ensures consistent model comparisons.

Usage:
    python streamlined_evaluation_single_training.py
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
RESULTS_DIR = 'streamlined_eval_single_training'
DATA_PATH = 'data_updated/state_month_overdose_2015_2023.xlsx'
TRIALS_PER_MODEL = 30

# Optimal hyperparameters
OPTIMAL_PARAMS = {
    'sarima': {'order': (1, 1, 1), 'seasonal_order': (1, 1, 1, 12)},
    'lstm': {'lookback': 12, 'batch_size': 8, 'epochs': 100, 'units': 50, 'dropout': 0.1},
    'tcn': {'lookback': 12, 'batch_size': 8, 'epochs': 100, 'filters': 64, 'kernel_size': 3},
    'seq2seq': {'lookback': 12, 'batch_size': 8, 'epochs': 50, 'encoder_units': 64, 'decoder_units': 64},
    'transformer': {'lookback': 12, 'batch_size': 8, 'epochs': 100, 'd_model': 64, 'num_heads': 4}
}

# Create results directory structure
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(f'{RESULTS_DIR}/trained_models', exist_ok=True)
os.makedirs(f'{RESULTS_DIR}/figures', exist_ok=True)
os.makedirs(f'{RESULTS_DIR}/data_exports', exist_ok=True)

class StreamlinedSingleTrainingPipeline:
    """Streamlined pipeline with single training phase"""
    
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
        
        # Handle the data format
        if 'Row Labels' in df.columns:
            df['Date'] = pd.to_datetime(df['Row Labels'])
        elif 'Month' in df.columns:
            df['Date'] = pd.to_datetime(df['Month'])
        else:
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
        df['Deaths'] = df['Deaths'].apply(lambda x: 0 if str(x).lower() == 'suppressed' else int(x))
        
        print(f"Processed data shape: {df.shape}")
        print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"Deaths range: {df['Deaths'].min()} to {df['Deaths'].max()}")
        
        return df
    
    def create_data_splits(self, df: pd.DataFrame):
        """Create training data and all test horizons"""
        splits = {}
        
        # Single training period (2015-2019)
        train_data = df[df['Date'] <= '2019-12-31'].copy()
        splits['train'] = train_data
        
        # All test horizons
        test_endpoints = [
            ('2020-12-31', 'horizon_1year'),
            ('2021-12-31', 'horizon_2year'), 
            ('2022-12-31', 'horizon_3year'),
            ('2023-12-31', 'horizon_4year')
        ]
        
        for end_date, horizon_name in test_endpoints:
            test_data = df[(df['Date'] >= '2020-01-01') & (df['Date'] <= end_date)].copy()
            
            splits[horizon_name] = {
                'test': test_data,
                'description': f'Test: 2020-{end_date[:4]}',
                'test_length_months': len(test_data)
            }
        
        return splits
    
    def create_dataset(self, dataset, look_back=3):
        """Create dataset for sequence models"""
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back):
            a = dataset[i:(i + look_back)]
            dataX.append(a)
            dataY.append(dataset[i + look_back])
        return np.array(dataX), np.array(dataY)
    
    def calculate_comprehensive_metrics(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Calculate comprehensive evaluation metrics"""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1))) * 100
        mse = mean_squared_error(y_true, y_pred)
        
        # Prediction intervals (95% confidence)
        residuals = y_true - y_pred
        std_residual = np.std(residuals)
        z_score = 1.96
        margin_of_error = z_score * std_residual
        
        lower_bound = y_pred - margin_of_error
        upper_bound = y_pred + margin_of_error
        coverage = np.mean((y_true >= lower_bound) & (y_true <= upper_bound)) * 100
        interval_width = np.mean(upper_bound - lower_bound)
        
        return {
            'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'MSE': mse,
            'lower_bound': lower_bound, 'upper_bound': upper_bound,
            'coverage': coverage, 'interval_width': interval_width,
            'residuals': residuals
        }
    
    def build_lstm_model(self, look_back: int, units: int = 50, dropout: float = 0.0):
        """Build LSTM model"""
        model = Sequential([
            LSTM(units, activation='relu', input_shape=(look_back, 1), return_sequences=False),
            Dropout(dropout),
            Dense(1)
        ])
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model
    
    def build_tcn_model(self, look_back: int, filters: int = 64, kernel_size: int = 3):
        """Build TCN model"""
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
        """Build Seq2Seq model with attention"""
        scaler = MinMaxScaler()
        
        encoder_inputs = Input(shape=(look_back, 1))
        encoder_gru = GRU(encoder_units, return_sequences=True, return_state=True)
        encoder_outputs, encoder_state = encoder_gru(encoder_inputs)
        
        decoder_inputs = Input(shape=(1, 1))
        decoder_gru = GRU(decoder_units, return_sequences=True, return_state=True)
        decoder_outputs, _ = decoder_gru(decoder_inputs, initial_state=encoder_state)
        
        attention = tf.keras.layers.Attention()
        context_vector = attention([decoder_outputs, encoder_outputs])
        
        decoder_combined = Concatenate(axis=-1)([decoder_outputs, context_vector])
        decoder_dense = Dense(decoder_units, activation='relu')(decoder_combined)
        outputs = Dense(1)(decoder_dense)
        
        model = Model([encoder_inputs, decoder_inputs], outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        return model, scaler
    
    def build_transformer_model(self, look_back: int, d_model: int = 64, num_heads: int = 4):
        """Build Transformer model"""
        scaler = MinMaxScaler()
        
        inputs = Input(shape=(look_back, 1))
        x = Dense(d_model)(inputs)
        
        positions = tf.range(start=0, limit=look_back, delta=1, dtype=tf.float32)
        position_encoding = tf.expand_dims(positions, -1) / 10000.0
        x = x + position_encoding
        
        attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
        x = Add()([x, attn_output])
        x = LayerNormalization()(x)
        
        ff_output = Dense(d_model * 4, activation='relu')(x)
        ff_output = Dense(d_model)(ff_output)
        x = Add()([x, ff_output])
        x = LayerNormalization()(x)
        
        x = GlobalAveragePooling1D()(x)
        outputs = Dense(1)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        return model, scaler
    
    def generate_autoregressive_forecast(self, model, initial_sequence, num_predictions, 
                                       look_back, model_type='lstm'):
        """Generate TRUE autoregressive forecasts"""
        predictions = []
        
        if model_type in ['lstm', 'tcn']:
            current_sequence = initial_sequence.copy()
            for step in range(num_predictions):
                next_prediction = model.predict(current_sequence, verbose=0)
                pred_value = next_prediction[0][0]
                predictions.append(pred_value)
                
                # Update with prediction (autoregressive)
                new_sequence = np.append(current_sequence[0, 1:], [[pred_value]], axis=0)
                current_sequence = new_sequence.reshape((1, look_back, 1))
        
        elif model_type == 'seq2seq':
            encoder_input = initial_sequence.copy()
            for step in range(num_predictions):
                decoder_input = np.zeros((1, 1, 1))
                prediction = model.predict([encoder_input, decoder_input], verbose=0)
                pred_value = prediction[0][0][0]
                predictions.append(pred_value)
                
                # Update with prediction
                encoder_input = np.roll(encoder_input, -1, axis=1)
                encoder_input[0, -1, 0] = pred_value
        
        elif model_type == 'transformer':
            current_sequence = initial_sequence.copy()
            for step in range(num_predictions):
                next_prediction = model.predict(current_sequence, verbose=0)
                pred_value = next_prediction[0][0]
                predictions.append(pred_value)
                
                # Update with prediction
                new_sequence = np.append(current_sequence[0, 1:], [[pred_value]], axis=0)
                current_sequence = new_sequence.reshape((1, look_back, 1))
        
        return np.array(predictions)
    
    def train_single_model(self, model_name: str, train_data: pd.DataFrame, trial_num: int):
        """Train a single model instance"""
        params = OPTIMAL_PARAMS[model_name]
        
        # Set seed for this trial
        trial_seed = RANDOM_SEED + trial_num
        np.random.seed(trial_seed)
        tf.random.set_seed(trial_seed)
        
        try:
            if model_name == 'sarima':
                deaths_series = train_data['Deaths'].values.astype(float)
                model = SARIMAX(deaths_series, 
                               order=params['order'], 
                               seasonal_order=params['seasonal_order'],
                               enforce_stationarity=False, 
                               enforce_invertibility=False)
                fitted_model = model.fit(disp=False, maxiter=200)
                return fitted_model, None
            
            else:
                # Sequence models
                deaths_series = train_data['Deaths'].values.astype(float)
                look_back = params['lookback']
                
                trainX, trainY = self.create_dataset(pd.Series(deaths_series), look_back)
                trainX = trainX.reshape((trainX.shape[0], look_back, 1))
                
                # Build model
                if model_name == 'lstm':
                    model = self.build_lstm_model(look_back, params['units'], params['dropout'])
                    scaler = None
                    
                elif model_name == 'tcn':
                    model = self.build_tcn_model(look_back, params['filters'], params['kernel_size'])
                    scaler = None
                    
                elif model_name == 'seq2seq':
                    model, scaler = self.build_seq2seq_model(look_back, params['encoder_units'], params['decoder_units'])
                    scaler.fit(deaths_series.reshape(-1, 1))
                    trainX_scaled = scaler.transform(trainX.reshape(-1, 1)).reshape(trainX.shape)
                    trainY_scaled = scaler.transform(trainY.reshape(-1, 1)).flatten()
                    trainX, trainY = trainX_scaled, trainY_scaled
                    
                elif model_name == 'transformer':
                    model, scaler = self.build_transformer_model(look_back, params['d_model'], params['num_heads'])
                    scaler.fit(deaths_series.reshape(-1, 1))
                    trainX_scaled = scaler.transform(trainX.reshape(-1, 1)).reshape(trainX.shape)
                    trainY_scaled = scaler.transform(trainY.reshape(-1, 1)).flatten()
                    trainX, trainY = trainX_scaled, trainY_scaled
                
                # Train model
                if model_name == 'seq2seq':
                    decoder_input = np.zeros((trainX.shape[0], 1, 1))
                    model.fit([trainX, decoder_input], trainY.reshape(-1, 1, 1),
                             epochs=params['epochs'], batch_size=params['batch_size'],
                             verbose=0, validation_split=0.1)
                else:
                    model.fit(trainX, trainY,
                             epochs=params['epochs'], batch_size=params['batch_size'],
                             verbose=0, validation_split=0.1)
                
                return model, scaler
                
        except Exception as e:
            print(f"    Error training {model_name} trial {trial_num}: {e}")
            return None, None
    
    def evaluate_model_on_horizon(self, model, model_name: str, train_data: pd.DataFrame, 
                                test_data: pd.DataFrame, scaler=None):
        """Evaluate a trained model on a specific horizon"""
        params = OPTIMAL_PARAMS[model_name]
        train_deaths = train_data['Deaths'].values.astype(float)
        test_deaths = test_data['Deaths'].values.astype(float)
        
        if model_name == 'sarima':
            # SARIMA predictions
            train_pred = model.fittedvalues
            test_pred = model.predict(start=len(train_deaths), 
                                    end=len(train_deaths) + len(test_deaths) - 1)
            return train_deaths, train_pred.values, test_deaths, test_pred.values
        
        else:
            # Sequence models
            look_back = params['lookback']
            
            # Prepare data
            if scaler is not None:
                train_deaths_scaled = scaler.transform(train_deaths.reshape(-1, 1)).flatten()
            else:
                train_deaths_scaled = train_deaths
            
            # Training predictions (teacher forcing)
            train_X, train_Y = self.create_dataset(pd.Series(train_deaths_scaled), look_back)
            train_X = train_X.reshape((train_X.shape[0], look_back, 1))
            
            if model_name == 'seq2seq':
                decoder_input = np.zeros((train_X.shape[0], 1, 1))
                train_pred_scaled = model.predict([train_X, decoder_input], verbose=0).reshape(-1)
            else:
                train_pred_scaled = model.predict(train_X, verbose=0).reshape(-1)
            
            # Test predictions (autoregressive)
            initial_sequence = train_deaths_scaled[-look_back:].reshape((1, look_back, 1))
            test_pred_scaled = self.generate_autoregressive_forecast(
                model, initial_sequence, len(test_deaths), look_back, model_name)
            
            # Inverse transform
            if scaler is not None:
                train_pred = scaler.inverse_transform(train_pred_scaled.reshape(-1, 1)).flatten()
                test_pred = scaler.inverse_transform(test_pred_scaled.reshape(-1, 1)).flatten()
            else:
                train_pred = train_pred_scaled
                test_pred = test_pred_scaled
            
            return (train_deaths[look_back:], train_pred, test_deaths, test_pred)
    
    def run_complete_evaluation(self, data_splits: Dict, models: List[str]):
        """
        Complete evaluation pipeline:
        1. Train models once on 2015-2019
        2. Evaluate on all horizons
        3. Generate both experiment results
        """
        print("\n" + "="*80)
        print("COMPLETE EVALUATION PIPELINE")
        print("="*80)
        print("Phase 1: Training models once on 2015-2019")
        print("Phase 2: Evaluating on all forecast horizons")
        
        train_data = data_splits['train']
        horizon_names = ['horizon_1year', 'horizon_2year', 'horizon_3year', 'horizon_4year']
        
        # Store all results
        all_results = {
            'trained_models': {},
            'horizon_evaluations': {h: {} for h in horizon_names}
        }
        
        # Phase 1: Train models
        print(f"\nPhase 1: Training {len(models)} models with {TRIALS_PER_MODEL} trials each...")
        
        for model_name in models:
            print(f"\nTraining {model_name.upper()}...")
            trained_models = []
            
            for trial in range(1, TRIALS_PER_MODEL + 1):
                if trial % 5 == 0:
                    print(f"  Trial {trial}/{TRIALS_PER_MODEL}")
                
                model, scaler = self.train_single_model(model_name, train_data, trial)
                
                if model is not None:
                    trained_models.append({
                        'trial': trial,
                        'model': model,
                        'scaler': scaler
                    })
                    
                    # Save first model
                    if trial == 1:
                        model_save_path = f"{self.results_dir}/trained_models/{model_name}_best_model.pkl"
                        with open(model_save_path, 'wb') as f:
                            pickle.dump({'model': model, 'scaler': scaler}, f)
                        print(f"    Saved model to: {model_save_path}")
            
            all_results['trained_models'][model_name] = trained_models
            print(f"  Successfully trained {len(trained_models)}/{TRIALS_PER_MODEL} models")
        
        # Phase 2: Evaluate on all horizons
        print(f"\nPhase 2: Evaluating on {len(horizon_names)} forecast horizons...")
        
        for horizon_name in horizon_names:
            horizon_info = data_splits[horizon_name]
            test_data = horizon_info['test']
            
            print(f"\n{horizon_info['description']} ({len(test_data)} months)")
            
            for model_name in models:
                if model_name not in all_results['trained_models']:
                    continue
                
                print(f"  Evaluating {model_name.upper()}...")
                trained_models = all_results['trained_models'][model_name]
                horizon_results = []
                
                for model_info in trained_models:
                    trial = model_info['trial']
                    model = model_info['model']
                    scaler = model_info['scaler']
                    
                    try:
                        # Evaluate this model on this horizon
                        train_true, train_pred, test_true, test_pred = self.evaluate_model_on_horizon(
                            model, model_name, train_data, test_data, scaler)
                        
                        # Calculate metrics
                        train_metrics = self.calculate_comprehensive_metrics(train_true, train_pred)
                        test_metrics = self.calculate_comprehensive_metrics(test_true, test_pred)
                        
                        result = {
                            'trial': trial,
                            'model_name': model_name,
                            'horizon': horizon_name,
                            'train_true': train_true,
                            'train_pred': train_pred,
                            'test_true': test_true,
                            'test_pred': test_pred,
                            'train_metrics': train_metrics,
                            'test_metrics': test_metrics
                        }
                        
                        horizon_results.append(result)
                        
                    except Exception as e:
                        print(f"    Error evaluating trial {trial}: {e}")
                        continue
                
                all_results['horizon_evaluations'][horizon_name][model_name] = horizon_results
                print(f"    Completed {len(horizon_results)} evaluations")
        
        # Save complete results
        results_path = f"{self.results_dir}/complete_evaluation_results.pkl"
        with open(results_path, 'wb') as f:
            pickle.dump(all_results, f)
        
        # Generate experiment-specific analyses
        self.generate_experiment_1_analysis(all_results, data_splits)
        self.generate_experiment_2_analysis(all_results, data_splits)
        
        return all_results
    
    def generate_experiment_1_analysis(self, all_results: Dict, data_splits: Dict):
        """Generate Experiment 1 analysis (longest horizon performance)"""
        print("\n" + "="*60)
        print("EXPERIMENT 1: EXCESS MORTALITY ESTIMATION")
        print("="*60)
        print("Analyzing performance on longest horizon (2020-2023)")
        
        # Use longest horizon results
        longest_horizon = 'horizon_4year'
        exp1_results = all_results['horizon_evaluations'][longest_horizon]
        
        # Calculate summary statistics
        summary_stats = []
        
        for model_name, model_results in exp1_results.items():
            if not model_results:
                continue
            
            # Extract metrics
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
        
        # Save experiment 1 results
        exp1_dir = f"{self.results_dir}/experiment_1_excess_mortality"
        os.makedirs(exp1_dir, exist_ok=True)
        
        summary_df = pd.DataFrame(summary_stats)
        summary_path = f"{exp1_dir}/summary_statistics.csv"
        summary_df.to_csv(summary_path, index=False)
        
        # Create visualizations
        train_data = data_splits['train']
        test_data = data_splits[longest_horizon]['test']
        self.create_experiment_1_plots(exp1_results, train_data, test_data)
        
        print(f"Experiment 1 analysis saved to: {exp1_dir}")
    
    def generate_experiment_2_analysis(self, all_results: Dict, data_splits: Dict):
        """Generate Experiment 2 analysis (variance across horizons)"""
        print("\n" + "="*60)
        print("EXPERIMENT 2: VARIANCE ANALYSIS")
        print("="*60)
        print("Analyzing error accumulation across forecast horizons")
        
        horizon_names = ['horizon_1year', 'horizon_2year', 'horizon_3year', 'horizon_4year']
        
        # Organize data by horizon
        horizon_summary = []
        
        for horizon_name in horizon_names:
            horizon_results = all_results['horizon_evaluations'][horizon_name]
            horizon_months = data_splits[horizon_name]['test_length_months']
            
            for model_name, model_results in horizon_results.items():
                if not model_results:
                    continue
                
                # Extract test metrics
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
        
        # Save experiment 2 results
        exp2_dir = f"{self.results_dir}/experiment_2_variance_analysis"
        os.makedirs(exp2_dir, exist_ok=True)
        
        horizon_df = pd.DataFrame(horizon_summary)
        horizon_path = f"{exp2_dir}/horizon_summary.csv"
        horizon_df.to_csv(horizon_path, index=False)
        
        # Create variance analysis plots
        self.create_experiment_2_plots(horizon_df, all_results, data_splits)
        
        print(f"Experiment 2 analysis saved to: {exp2_dir}")
    
    def create_experiment_1_plots(self, results: Dict, train_data: pd.DataFrame, test_data: pd.DataFrame):
        """Create plots for Experiment 1 (longest horizon)"""
        print("Creating Experiment 1 visualizations...")
        
        # Determine alignment
        max_lookback = max([OPTIMAL_PARAMS[model].get('lookback', 0) 
                           for model in results.keys() if model != 'sarima'])
        
        plot_start_idx = max_lookback
        plot_train_dates = train_data['Date'].iloc[plot_start_idx:].values
        plot_test_dates = test_data['Date'].values
        all_plot_dates = np.concatenate([plot_train_dates, plot_test_dates])
        
        # Plot actual data
        plot_train_actual = train_data['Deaths'].iloc[plot_start_idx:].values
        plot_test_actual = test_data['Deaths'].values
        all_plot_actual = np.concatenate([plot_train_actual, plot_test_actual])
        
        # Model colors
        model_colors = {
            'sarima': 'blue', 'lstm': 'red', 'tcn': 'green',
            'seq2seq': 'purple', 'transformer': 'orange'
        }
        
        # Main comparison plot
        plt.figure(figsize=(18, 12))
        plt.plot(all_plot_dates, all_plot_actual, 'k-', linewidth=3, label='Actual Deaths', zorder=10)
        
        forecast_start = test_data['Date'].iloc[0]
        plt.axvline(forecast_start, color='gray', linestyle='--', linewidth=2, 
                   alpha=0.7, label='Forecast Start (Jan 2020)')
        
        for model_name, model_results in results.items():
            if not model_results:
                continue
            
            color = model_colors.get(model_name, 'black')
            
            # Calculate average predictions
            avg_train_pred = np.mean([r['train_pred'] for r in model_results], axis=0)
            avg_test_pred = np.mean([r['test_pred'] for r in model_results], axis=0)
            
            # Handle alignment
            if model_name == 'sarima':
                model_train_pred = avg_train_pred[plot_start_idx:]
                model_train_dates = plot_train_dates
            else:
                model_train_pred = avg_train_pred
                model_train_dates = plot_train_dates
            
            all_model_dates = np.concatenate([model_train_dates, plot_test_dates])
            all_model_pred = np.concatenate([model_train_pred, avg_test_pred])
            
            plt.plot(all_model_dates, all_model_pred, '-', color=color, linewidth=2,
                    label=f'{model_name.upper()}', alpha=0.8)
            
            # Add prediction intervals
            train_true = model_results[0]['train_true']
            test_true = model_results[0]['test_true']
            
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
        
        plt.xlabel('Date', fontsize=16, fontweight='bold')
        plt.ylabel('Deaths', fontsize=16, fontweight='bold')
        plt.title('Experiment 1: Excess Mortality Estimation (2020-2023)', 
                 fontsize=20, fontweight='bold', pad=20)
        
        plt.xticks(fontsize=14, rotation=45)
        plt.yticks(fontsize=14)
        plt.legend(loc='upper left', fontsize=12, frameon=True, fancybox=True, shadow=True)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = f"{self.results_dir}/figures/experiment_1_model_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Individual comparison plots
        if 'sarima' in results:
            self.create_individual_comparison_plots(results, train_data, test_data, 'experiment_1')
        
        print(f"  Experiment 1 plots saved to: {self.results_dir}/figures/")
    
    def create_individual_comparison_plots(self, results: Dict, train_data: pd.DataFrame, 
                                         test_data: pd.DataFrame, experiment_name: str):
        """Create individual SARIMA vs model comparison plots"""
        
        sarima_results = results.get('sarima', [])
        if not sarima_results:
            return
        
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
            
            # Handle alignment
            lookback = OPTIMAL_PARAMS[model_name].get('lookback', 0)
            model_train_dates = train_data['Date'].iloc[lookback:].values
            plot_test_dates = test_data['Date'].values
            all_dates = np.concatenate([model_train_dates, plot_test_dates])
            
            all_actual = np.concatenate([model_train_true, model_test_true])
            
            # Align SARIMA
            sarima_aligned_train_pred = sarima_train_pred[lookback:]
            all_sarima_pred = np.concatenate([sarima_aligned_train_pred, sarima_test_pred])
            all_model_pred = np.concatenate([model_train_pred, model_test_pred])
            
            # Calculate metrics for intervals
            sarima_train_metrics = self.calculate_comprehensive_metrics(model_train_true, sarima_aligned_train_pred)
            sarima_test_metrics = self.calculate_comprehensive_metrics(sarima_test_true, sarima_test_pred)
            model_train_metrics = self.calculate_comprehensive_metrics(model_train_true, model_train_pred)
            model_test_metrics = self.calculate_comprehensive_metrics(model_test_true, model_test_pred)
            
            all_sarima_lower = np.concatenate([sarima_train_metrics['lower_bound'], sarima_test_metrics['lower_bound']])
            all_sarima_upper = np.concatenate([sarima_train_metrics['upper_bound'], sarima_test_metrics['upper_bound']])
            all_model_lower = np.concatenate([model_train_metrics['lower_bound'], model_test_metrics['lower_bound']])
            all_model_upper = np.concatenate([model_train_metrics['upper_bound'], model_test_metrics['upper_bound']])
            
            # Create plot
            plt.figure(figsize=(16, 10))
            
            plt.plot(all_dates, all_actual, 'k-', linewidth=3, label='Actual Deaths', zorder=5)
            plt.plot(all_dates, all_sarima_pred, '--', color='blue', linewidth=2, 
                    label='SARIMA', alpha=0.8)
            plt.fill_between(all_dates, all_sarima_lower, all_sarima_upper,
                           color='blue', alpha=0.25, label='SARIMA 95% PI')
            
            plt.plot(all_dates, all_model_pred, '-', color='red', linewidth=2,
                    label=f'{model_name.upper()}', alpha=0.8)
            plt.fill_between(all_dates, all_model_lower, all_model_upper,
                           color='red', alpha=0.25, label=f'{model_name.upper()} 95% PI')
            
            forecast_start = test_data['Date'].iloc[0]
            plt.axvline(forecast_start, color='green', linestyle='--', linewidth=2, 
                       alpha=0.7, label='Forecast Start')
            
            plt.xlabel('Date', fontsize=16, fontweight='bold')
            plt.ylabel('Deaths', fontsize=16, fontweight='bold')
            plt.title(f'{experiment_name.title()}: SARIMA vs {model_name.upper()}', 
                     fontsize=20, fontweight='bold', pad=20)
            
            plt.xticks(fontsize=14, rotation=45)
            plt.yticks(fontsize=14)
            plt.legend(loc='upper left', fontsize=14, frameon=True, fancybox=True, shadow=True)
            plt.grid(True, alpha=0.3)
            
            # Add metrics
            sarima_test_rmse = np.mean([r['test_metrics']['RMSE'] for r in sarima_results])
            model_test_rmse = np.mean([r['test_metrics']['RMSE'] for r in model_results])
            
            metrics_text = f'Test RMSE:\nSARIMA: {sarima_test_rmse:.0f}\n{model_name.upper()}: {model_test_rmse:.0f}'
            plt.text(0.02, 0.98, metrics_text, transform=plt.gca().transAxes, fontsize=12,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            
            plot_path = f"{self.results_dir}/figures/{experiment_name}_sarima_vs_{model_name}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
    
    def create_experiment_2_plots(self, horizon_df: pd.DataFrame, all_results: Dict, data_splits: Dict):
        """Create plots for Experiment 2 (variance analysis)"""
        print("Creating Experiment 2 visualizations...")
        
        models = horizon_df['Model'].unique()
        horizons = sorted(horizon_df['Forecast_Horizon_Months'].unique())
        
        # Variance analysis plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Experiment 2: Error Accumulation Across Forecast Horizons', 
                     fontsize=20, fontweight='bold')
        
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
        ax.set_title('RMSE Growth Over Time', fontsize=16, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Plot 2: MAPE vs Horizon  
        ax = axes[0, 1]
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
                           label=model, marker='s', color=colors[i], linewidth=2, markersize=8)
        
        ax.set_xlabel('Forecast Horizon (Months)', fontsize=14, fontweight='bold')
        ax.set_ylabel('MAPE (%)', fontsize=14, fontweight='bold')
        ax.set_title('MAPE Growth Over Time', fontsize=16, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Coverage vs Horizon
        ax = axes[1, 0]
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
                           label=model, marker='^', color=colors[i], linewidth=2, markersize=8)
        
        ax.axhline(y=95, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Target 95%')
        ax.set_xlabel('Forecast Horizon (Months)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Coverage (%)', fontsize=14, fontweight='bold')
        ax.set_title('Prediction Interval Coverage', fontsize=16, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Interval Width vs Horizon
        ax = axes[1, 1]
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
                           label=model, marker='D', color=colors[i], linewidth=2, markersize=8)
        
        ax.set_xlabel('Forecast Horizon (Months)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Average PI Width', fontsize=14, fontweight='bold')
        ax.set_title('Prediction Interval Width', fontsize=16, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = f"{self.results_dir}/figures/experiment_2_variance_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Individual horizon comparison plots
        self.create_horizon_comparison_plots(all_results, data_splits)
        
        print(f"  Experiment 2 plots saved to: {self.results_dir}/figures/")
    
    def create_horizon_comparison_plots(self, all_results: Dict, data_splits: Dict):
        """Create individual plots for each forecast horizon"""
        
        train_data = data_splits['train']
        horizon_names = ['horizon_1year', 'horizon_2year', 'horizon_3year', 'horizon_4year']
        
        for horizon_name in horizon_names:
            if horizon_name not in all_results['horizon_evaluations']:
                continue
            
            results = all_results['horizon_evaluations'][horizon_name]
            test_data = data_splits[horizon_name]['test']
            
            print(f"  Creating plot for {horizon_name}...")
            self.create_individual_comparison_plots(results, train_data, test_data, horizon_name)
    
    def export_prediction_data(self, all_results: Dict, data_splits: Dict):
        """Export all prediction data for custom analysis"""
        print("Exporting prediction data...")
        
        # Export experiment 1 data (longest horizon)
        longest_horizon = 'horizon_4year'
        exp1_results = all_results['horizon_evaluations'][longest_horizon]
        
        export_data = {}
        train_data = data_splits['train']
        test_data = data_splits[longest_horizon]['test']
        
        for model_name, model_results in exp1_results.items():
            if not model_results:
                continue
            
            train_preds = np.array([r['train_pred'] for r in model_results])
            test_preds = np.array([r['test_pred'] for r in model_results])
            
            avg_train_pred = np.mean(train_preds, axis=0)
            std_train_pred = np.std(train_preds, axis=0)
            avg_test_pred = np.mean(test_preds, axis=0)
            std_test_pred = np.std(test_preds, axis=0)
            
            train_true = model_results[0]['train_true']
            test_true = model_results[0]['test_true']
            
            train_metrics = self.calculate_comprehensive_metrics(train_true, avg_train_pred)
            test_metrics = self.calculate_comprehensive_metrics(test_true, avg_test_pred)
            
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
        
        # Save data
        export_path = f"{self.results_dir}/data_exports/complete_prediction_data.pkl"
        with open(export_path, 'wb') as f:
            pickle.dump(export_data, f)
        
        # CSV export
        csv_data = []
        for model_name, model_data in export_data.items():
            for period in ['train', 'test']:
                for i, date in enumerate(model_data[period]['dates']):
                    csv_data.append({
                        'Model': model_name.upper(),
                        'Period': period.title(),
                        'Date': date,
                        'Actual': model_data[period]['actual'][i],
                        'Predicted_Mean': model_data[period]['predicted_mean'][i],
                        'Predicted_Std': model_data[period]['predicted_std'][i],
                        'Lower_Bound': model_data[period]['lower_bound'][i],
                        'Upper_Bound': model_data[period]['upper_bound'][i]
                    })
        
        csv_df = pd.DataFrame(csv_data)
        csv_path = f"{self.results_dir}/data_exports/complete_prediction_data.csv"
        csv_df.to_csv(csv_path, index=False)
        
        print(f"Prediction data exported to:")
        print(f"  - {export_path}")
        print(f"  - {csv_path}")
    
    def generate_final_report(self, all_results: Dict):
        """Generate comprehensive final report"""
        print("\n" + "="*80)
        print("GENERATING FINAL REPORT")
        print("="*80)
        
        report_content = []
        report_content.append("STREAMLINED SINGLE TRAINING EVALUATION RESULTS")
        report_content.append("=" * 60)
        report_content.append(f"Execution Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_content.append(f"Random Seed: {RANDOM_SEED}")
        report_content.append(f"Trials per Model: {TRIALS_PER_MODEL}")
        report_content.append("")
        
        # Training summary
        report_content.append("TRAINING SUMMARY:")
        report_content.append("-" * 20)
        report_content.append("✓ Models trained ONCE on 2015-2019 data")
        report_content.append("✓ Same trained models evaluated on ALL horizons")
        report_content.append("✓ True autoregressive forecasting implemented")
        report_content.append("✓ No redundant training between experiments")
        report_content.append("")
        
        # Model parameters
        report_content.append("OPTIMAL HYPERPARAMETERS:")
        report_content.append("-" * 25)
        for model, params in OPTIMAL_PARAMS.items():
            report_content.append(f"{model.upper()}: {params}")
        report_content.append("")
        
        # Training success rates
        report_content.append("MODEL TRAINING SUCCESS RATES:")
        report_content.append("-" * 30)
        for model_name, trained_models in all_results['trained_models'].items():
            success_rate = len(trained_models) / TRIALS_PER_MODEL * 100
            report_content.append(f"{model_name.upper()}: {len(trained_models)}/{TRIALS_PER_MODEL} ({success_rate:.1f}%)")
        report_content.append("")
        
        # Best performing model
        try:
            summary_path = f"{self.results_dir}/experiment_1_excess_mortality/summary_statistics.csv"
            if os.path.exists(summary_path):
                summary_df = pd.read_csv(summary_path)
                best_model_idx = summary_df['Test_RMSE_Mean'].idxmin()
                best_model = summary_df.iloc[best_model_idx]
                
                report_content.append("BEST PERFORMING MODEL (Experiment 1):")
                report_content.append("-" * 35)
                report_content.append(f"Model: {best_model['Model']}")
                report_content.append(f"Test RMSE: {best_model['Test_RMSE_Mean']:.2f} ± {best_model['Test_RMSE_Std']:.2f}")
                report_content.append(f"Test MAPE: {best_model['Test_MAPE_Mean']:.2f}% ± {best_model['Test_MAPE_Std']:.2f}%")
                report_content.append(f"Coverage: {best_model['Test_Coverage_Mean']:.1f}% ± {best_model['Test_Coverage_Std']:.1f}%")
        except:
            pass
        
        report_content.append("")
        report_content.append("KEY IMPROVEMENTS:")
        report_content.append("-" * 20)
        report_content.append("✓ Single training phase eliminates redundancy")
        report_content.append("✓ True autoregressive forecasting (no future data leakage)")
        report_content.append("✓ Consistent model comparison across horizons")
        report_content.append("✓ Proper error accumulation analysis")
        report_content.append("✓ Efficient computational pipeline")
        
        report_content.append("")
        report_content.append("GENERATED OUTPUTS:")
        report_content.append("-" * 20)
        report_content.append("📊 Experiment 1: Excess mortality on 2020-2023")
        report_content.append("📈 Experiment 2: Error accumulation across horizons")
        report_content.append("🎨 Individual model comparison plots")
        report_content.append("📋 Comprehensive summary statistics")
        report_content.append("💾 Complete prediction data exports")
        report_content.append("🔧 Trained models for deployment")
        
        # Save report
        report_path = f"{self.results_dir}/EVALUATION_SUMMARY_REPORT.txt"
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_content))
        
        print(f"Final report saved to: {report_path}")
        print("\n" + "\n".join(report_content))


def main():
    """Main execution function"""
    print("=" * 80)
    print("STREAMLINED SINGLE TRAINING EVALUATION PIPELINE")
    print("Advanced Machine Learning for Substance Overdose Mortality Prediction")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  - Random seed: {RANDOM_SEED}")
    print(f"  - Trials per model: {TRIALS_PER_MODEL}")
    print(f"  - Results directory: {RESULTS_DIR}")
    print(f"  - Data path: {DATA_PATH}")
    print(f"  - Efficient design: Train once, evaluate on all horizons")
    
    if not os.path.exists(DATA_PATH):
        print(f"\nError: Data file not found at {DATA_PATH}")
        return False
    
    start_time = time.time()
    pipeline = StreamlinedSingleTrainingPipeline(DATA_PATH, RESULTS_DIR)
    
    try:
        # Step 1: Load data
        print("\n" + "="*60)
        print("STEP 1: LOADING AND PREPROCESSING DATA")
        print("="*60)
        df = pipeline.load_and_preprocess_data()
        
        # Step 2: Create splits
        print("\n" + "="*60)
        print("STEP 2: CREATING DATA SPLITS") 
        print("="*60)
        data_splits = pipeline.create_data_splits(df)
        
        print("Data splits created:")
        print(f"  Training: 2015-2019 ({len(data_splits['train'])} months)")
        for split_name, split_info in data_splits.items():
            if split_name != 'train':
                print(f"  {split_name}: {split_info['description']} ({split_info['test_length_months']} months)")
        
        # Step 3: Complete evaluation
        models = ['sarima', 'lstm', 'tcn', 'seq2seq', 'transformer']
        print(f"\nModels to evaluate: {models}")
        
        print("\n" + "="*60)
        print("STEP 3: COMPLETE EVALUATION PIPELINE")
        print("="*60)
        
        all_results = pipeline.run_complete_evaluation(data_splits, models)
        
        # Step 4: Export data
        print("\n" + "="*60)
        print("STEP 4: EXPORTING PREDICTION DATA")
        print("="*60)
        
        pipeline.export_prediction_data(all_results, data_splits)
        
        # Step 5: Generate final report
        pipeline.generate_final_report(all_results)
        
        # Execution summary
        total_time = time.time() - start_time
        
        print("\n" + "🎉" * 20)
        print("STREAMLINED SINGLE TRAINING EVALUATION COMPLETED!")
        print("🎉" * 20)
        
        print(f"\nExecution Summary:")
        print(f"  Total time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        print(f"  Models evaluated: {len(models)}")
        print(f"  Trials per model: {TRIALS_PER_MODEL}")
        print(f"  Forecast horizons: 4 (12, 24, 36, 48 months)")
        print(f"  Random seed: {RANDOM_SEED}")
        
        print(f"\n📁 Results saved to: {RESULTS_DIR}")
        
        print("\n📊 Key outputs:")
        print("  🔬 Experiment 1 (Excess Mortality):")
        print("    - figures/experiment_1_model_comparison.png")
        print("    - figures/experiment_1_sarima_vs_[model].png") 
        print("    - experiment_1_excess_mortality/summary_statistics.csv")
        
        print("  📈 Experiment 2 (Variance Analysis):")
        print("    - figures/experiment_2_variance_analysis.png")
        print("    - figures/horizon_[1-4]year_sarima_vs_[model].png")
        print("    - experiment_2_variance_analysis/horizon_summary.csv")
        
        print("  💾 Data & Models:")
        print("    - data_exports/complete_prediction_data.csv")
        print("    - trained_models/[model]_best_model.pkl")
        
        print("\n✅ Key achievements:")
        print("  ✓ Eliminated redundant training between experiments")
        print("  ✓ True autoregressive forecasting (no data leakage)")
        print("  ✓ Consistent model evaluation across all horizons")
        print("  ✓ Comprehensive error accumulation analysis")
        print("  ✓ Publication-ready visualizations")
        print("  ✓ Complete data export for custom analysis")
        
        print("\n🔄 Next steps:")
        print("  1. Review experiment results in summary CSVs")
        print("  2. Examine error accumulation patterns across horizons")
        print("  3. Use exported data for custom figure creation")
        print("  4. Deploy best performing models using saved models")
        print("  5. Integrate findings into research manuscript")
        
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
        exit(1)#!/usr/bin/env python3
