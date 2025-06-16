import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, GRU, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Add, GlobalAveragePooling1D
from statsmodels.tsa.statespace.sarimax import SARIMAX
import tensorflow as tf
from tensorflow.keras import layers
import warnings
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any
import itertools

warnings.filterwarnings('ignore')

class ModelValidationPipeline:
    """
    Comprehensive pipeline for time series model validation and comparison
    """
    
    def __init__(self, data_path: str, results_dir: str = './results'):
        self.data_path = data_path
        self.results_dir = results_dir
        self.models = {}
        self.results = []
        
        # Create results directory structure
        os.makedirs(f'{results_dir}/individual_runs', exist_ok=True)
        os.makedirs(f'{results_dir}/aggregated', exist_ok=True)
        
    def load_and_preprocess_data(self):
        """Load and preprocess the overdose dataset"""
        df = pd.read_excel(self.data_path)
        df['Deaths'] = df['Deaths'].apply(lambda x: 0 if x == 'Suppressed' else int(x))
        df['Month'] = pd.to_datetime(df['Month'])
        df.set_index('Month', inplace=True)
        df = df.reset_index()
        df['Month Code'] = pd.to_datetime(df['Month Code'])
        df = df.groupby(['Month']).agg({'Deaths': 'sum'}).reset_index()
        
        return df
    
    def create_train_val_test_split(self, df: pd.DataFrame, 
                                  train_end: str = '2018-01-01',
                                  val_end: str = '2019-01-01', 
                                  test_end: str = '2019-12-01'):
        """Create proper train/validation/test splits"""
        train = df[df['Month'] < train_end]
        validation = df[(df['Month'] >= train_end) & (df['Month'] < val_end)]
        test = df[(df['Month'] >= val_end) & (df['Month'] <= test_end)]
        
        return train, validation, test
    
    def create_dataset(self, dataset, look_back=3):
        """Create dataset for sequence models"""
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back):
            a = dataset.iloc[i:(i + look_back)]
            dataX.append(a)
            dataY.append(dataset.iloc[i + look_back])
        return np.array(dataX), np.array(dataY)
    
    def calculate_metrics(self, y_true, y_pred):
        """Calculate comprehensive evaluation metrics"""
        # Basic metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        # MAPE (handle division by zero)
        mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1))) * 100
        
        # Additional metrics
        mse = mean_squared_error(y_true, y_pred)
        
        return {
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'MSE': mse
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
        """Build Temporal Convolutional Network"""
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
    
    def build_seq2seq_model(self, look_back: int, units: int = 50):
        """Build Seq2Seq model with attention (simplified)"""
        # Encoder
        encoder_inputs = layers.Input(shape=(look_back, 1))
        encoder = layers.LSTM(units, return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        encoder_states = [state_h, state_c]
        
        # Decoder
        decoder_inputs = layers.Input(shape=(1, 1))
        decoder_lstm = layers.LSTM(units, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_dense = layers.Dense(1)
        decoder_outputs = decoder_dense(decoder_outputs)
        
        model = tf.keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def build_transformer_model(self, look_back: int, d_model: int = 64, num_heads: int = 4):
        """Build simple Transformer model"""
        inputs = layers.Input(shape=(look_back, 1))
        
        # Positional encoding (simplified)
        x = layers.Dense(d_model)(inputs)
        
        # Multi-head attention
        attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
        x = Add()([x, attention_output])
        x = LayerNormalization()(x)
        
        # Feed forward
        ff_output = layers.Dense(d_model * 4, activation='relu')(x)
        ff_output = layers.Dense(d_model)(ff_output)
        x = Add()([x, ff_output])
        x = LayerNormalization()(x)
        
        # Output
        x = GlobalAveragePooling1D()(x)
        outputs = layers.Dense(1)(x)
        
        model = tf.keras.Model(inputs, outputs)
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def generate_forecast(self, model, initial_sequence, num_predictions, look_back, model_type='lstm'):
        """Generate forecasts for different model types"""
        predictions = []
        
        if model_type in ['lstm', 'tcn', 'transformer']:
            current_sequence = initial_sequence.copy()
            for _ in range(num_predictions):
                next_prediction = model.predict(current_sequence, verbose=0)
                predictions.append(next_prediction[0][0])
                
                # Update sequence
                new_sequence = np.append(current_sequence[0, 1:], [[next_prediction[0][0]]], axis=0)
                current_sequence = new_sequence.reshape((1, look_back, 1))
                
        elif model_type == 'seq2seq':
            # Simplified seq2seq forecasting
            encoder_input = initial_sequence
            for _ in range(num_predictions):
                decoder_input = np.zeros((1, 1, 1))  # Start token
                prediction = model.predict([encoder_input, decoder_input], verbose=0)
                predictions.append(prediction[0][0][0])
                encoder_input = np.roll(encoder_input, -1, axis=1)
                encoder_input[0, -1, 0] = prediction[0][0][0]
        
        return np.array(predictions)
    
    def train_sarima_model(self, train_data, validation_data=None, 
                          order=(1,1,1), seasonal_order=(1,1,1,12)):
        """Train SARIMA model"""
        try:
            if validation_data is not None:
                # Train on train + validation for final model
                combined_data = pd.concat([train_data, validation_data])['Deaths']
            else:
                combined_data = train_data['Deaths']
                
            model = SARIMAX(combined_data, order=order, seasonal_order=seasonal_order,
                           enforce_stationarity=False, enforce_invertibility=False)
            fitted_model = model.fit(disp=False)
            return fitted_model
        except Exception as e:
            print(f"SARIMA training failed: {e}")
            return None
    
    def run_experiment(self, config: Dict[str, Any], seed: int = None):
        """Run a single experiment with given configuration"""
        if seed is not None:
            np.random.seed(seed)
            tf.random.set_seed(seed)
        
        df = self.load_and_preprocess_data()
        train, validation, test = self.create_train_val_test_split(df)
        
        results = {
            'config': config.copy(),
            'seed': seed,
            'timestamp': datetime.now().isoformat(),
            'model_results': {}
        }
        
        # SARIMA (doesn't use lookback)
        if 'SARIMA' in config['models']:
            sarima_model = self.train_sarima_model(train, validation)
            if sarima_model is not None:
                # Predict on test set
                test_start_idx = len(train) + len(validation)
                test_end_idx = test_start_idx + len(test) - 1
                sarima_predictions = sarima_model.predict(start=test_start_idx, end=test_end_idx)
                
                metrics = self.calculate_metrics(test['Deaths'].values, sarima_predictions.values)
                results['model_results']['SARIMA'] = {
                    'predictions': sarima_predictions.tolist(),
                    'metrics': metrics
                }
        
        # Sequence models (LSTM, TCN, Seq2Seq, Transformer)
        sequence_models = [m for m in config['models'] if m != 'SARIMA']
        
        for model_name in sequence_models:
            if config.get('look_back') is None:
                continue
                
            look_back = config['look_back']
            
            # Prepare data for sequence models
            train_val = pd.concat([train, validation])
            extended_test = pd.concat([train_val.iloc[-look_back:], test])
            
            trainX, trainY = self.create_dataset(train_val['Deaths'], look_back)
            testX, testY = self.create_dataset(extended_test['Deaths'], look_back)
            
            trainX = trainX.reshape((trainX.shape[0], look_back, 1))
            testX = testX.reshape((testX.shape[0], look_back, 1))
            
            # Build and train model
            try:
                if model_name == 'LSTM':
                    model = self.build_lstm_model(look_back, 
                                                config.get('lstm_units', 50),
                                                config.get('dropout', 0.0))
                elif model_name == 'TCN':
                    model = self.build_tcn_model(look_back,
                                               config.get('tcn_filters', 64),
                                               config.get('kernel_size', 3))
                elif model_name == 'Seq2Seq':
                    model = self.build_seq2seq_model(look_back, config.get('seq2seq_units', 50))
                elif model_name == 'Transformer':
                    model = self.build_transformer_model(look_back,
                                                       config.get('d_model', 64),
                                                       config.get('num_heads', 4))
                
                # Training
                if model_name == 'Seq2Seq':
                    # Special training for seq2seq
                    decoder_input = np.zeros((trainX.shape[0], 1, 1))
                    model.fit([trainX, decoder_input], trainY.reshape(-1, 1, 1),
                             epochs=config.get('epochs', 50),
                             batch_size=config.get('batch_size', 8),
                             verbose=0)
                else:
                    model.fit(trainX, trainY,
                             epochs=config.get('epochs', 50),
                             batch_size=config.get('batch_size', 8),
                             verbose=0)
                
                # Generate predictions
                initial_sequence = trainX[-1].reshape((1, look_back, 1))
                test_predictions = self.generate_forecast(model, initial_sequence, 
                                                        len(test), look_back, 
                                                        model_name.lower())
                
                metrics = self.calculate_metrics(test['Deaths'].values, test_predictions)
                results['model_results'][model_name] = {
                    'predictions': test_predictions.tolist(),
                    'metrics': metrics
                }
                
            except Exception as e:
                print(f"{model_name} training failed: {e}")
                results['model_results'][model_name] = {'error': str(e)}
        
        return results
    
    def run_hyperparameter_search(self, base_config: Dict[str, Any], 
                                param_grid: Dict[str, List], 
                                n_seeds: int = 1):
        """Run hyperparameter search across multiple seeds"""
        all_results = []
        
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        for param_combo in itertools.product(*param_values):
            config = base_config.copy()
            for name, value in zip(param_names, param_combo):
                config[name] = value
            
            print(f"Testing configuration: {dict(zip(param_names, param_combo))}")
            
            # Run multiple seeds for this configuration
            for seed in range(n_seeds):
                if n_seeds > 1:
                    actual_seed = np.random.randint(0, 2**31 - 1)
                else:
                    actual_seed = 42  # Fixed seed for single runs
                
                result = self.run_experiment(config, actual_seed)
                all_results.append(result)
                
                # Save individual result
                filename = f"result_seed{actual_seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(f"{self.results_dir}/individual_runs/{filename}", 'w') as f:
                    json.dump(result, f, indent=2)
        
        return all_results
    
    def aggregate_results(self, results: List[Dict]):
        """Aggregate results across multiple runs"""
        aggregated = {}
        
        # Group by configuration (excluding seed)
        config_groups = {}
        for result in results:
            config_key = json.dumps({k: v for k, v in result['config'].items() 
                                   if k != 'seed'}, sort_keys=True)
            if config_key not in config_groups:
                config_groups[config_key] = []
            config_groups[config_key].append(result)
        
        # Aggregate metrics for each configuration
        for config_key, group_results in config_groups.items():
            config = json.loads(config_key)
            aggregated[config_key] = {
                'config': config,
                'n_runs': len(group_results),
                'model_aggregates': {}
            }
            
            # Get all models tested
            all_models = set()
            for result in group_results:
                all_models.update(result['model_results'].keys())
            
            # Aggregate metrics for each model
            for model_name in all_models:
                model_results = []
                for result in group_results:
                    if model_name in result['model_results'] and 'metrics' in result['model_results'][model_name]:
                        model_results.append(result['model_results'][model_name]['metrics'])
                
                if model_results:
                    # Calculate mean and std for each metric
                    metrics_agg = {}
                    for metric_name in model_results[0].keys():
                        values = [r[metric_name] for r in model_results]
                        metrics_agg[metric_name] = {
                            'mean': np.mean(values),
                            'std': np.std(values),
                            'min': np.min(values),
                            'max': np.max(values)
                        }
                    
                    aggregated[config_key]['model_aggregates'][model_name] = metrics_agg
        
        return aggregated

# Example usage and configuration
def main():
    # Initialize pipeline
    pipeline = ModelValidationPipeline('data/state_month_overdose.xlsx')
    
    # Base configuration
    base_config = {
        'models': ['SARIMA', 'LSTM', 'TCN', 'Transformer'],  # Models to test
        'epochs': 100,
        'batch_size': 8
    }
    
    # Hyperparameter grid for testing
    param_grid = {
        'look_back': [3, 6, 9, 12],
        'lstm_units': [32, 50, 64],
        'dropout': [0.0, 0.1, 0.2]
    }
    
    # Run single configuration test
    print("Running single configuration test...")
    single_config = base_config.copy()
    single_config.update({'look_back': 12, 'lstm_units': 50, 'dropout': 0.1})
    
    single_result = pipeline.run_experiment(single_config, seed=42)
    print("Single test completed!")
    
    # Run hyperparameter search with multiple seeds
    print("Running hyperparameter search...")
    # For demonstration, using smaller grid and fewer seeds
    small_param_grid = {
        'look_back': [6, 12],
        'lstm_units': [50]
    }
    
    all_results = pipeline.run_hyperparameter_search(base_config, small_param_grid, n_seeds=3)
    
    # Aggregate results
    aggregated = pipeline.aggregate_results(all_results)
    
    # Save aggregated results
    with open(f"{pipeline.results_dir}/aggregated/aggregated_results.json", 'w') as f:
        json.dump(aggregated, f, indent=2)
    
    print(f"Completed {len(all_results)} experiments. Results saved to {pipeline.results_dir}")
    
    return pipeline, all_results, aggregated

if __name__ == "__main__":
    pipeline, results, aggregated = main()