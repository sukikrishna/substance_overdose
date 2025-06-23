import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, Flatten, GRU
from tensorflow.keras.layers import RepeatVector, Concatenate
from tensorflow.keras.layers import Input, Add, ReLU, Lambda, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention
from tensorflow.keras.models import Model
import tensorflow as tf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
from tcn import TCN  # pip install keras-tcn
from tensorflow.keras.layers import Attention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import math
import warnings
warnings.filterwarnings("ignore")  # Match grid_sarima.py to suppress SARIMA convergence warnings

# ------------------ CONFIGURATION ------------------ #
# Optimal hyperparameters from your table
OPTIMAL_PARAMS = {
    'sarima': {'order': (1, 0, 0), 'seasonal_order': (2, 2, 2, 12)},
    'lstm': {'lookback': 9, 'batch_size': 8, 'epochs': 100},
    'tcn': {'lookback': 5, 'batch_size': 32, 'epochs': 50},
    'seq2seq_attn': {'lookback': 11, 'batch_size': 16, 'epochs': 50, 'encoder_units': 128, 'decoder_units': 128},
    'transformer': {'lookback': 5, 'batch_size': 32, 'epochs': 100, 'd_model': 64, 'n_heads': 2}
}

# Multiple seeds for robust evaluation
SEEDS = [42, 123, 456, 789, 1000]
TRIALS_PER_SEED = 30  # Number of trials per model per seed

DATA_PATH = 'data/state_month_overdose.xlsx'
RESULTS_DIR = 'test_results'

# Create results directory
os.makedirs(RESULTS_DIR, exist_ok=True)

# ------------------ HELPER FUNCTIONS ------------------ #
def load_and_preprocess_data():
    """Load and preprocess the overdose data - matches grid_sarima.py implementation"""
    df = pd.read_excel(DATA_PATH)
    df['Deaths'] = df['Deaths'].apply(lambda x: 0 if x == 'Suppressed' else int(x))
    df['Month'] = pd.to_datetime(df['Month'])
    df = df.groupby('Month').agg({'Deaths': 'sum'}).reset_index()
    return df

def create_train_val_test_split(df, train_end='2019-01-01', val_end='2020-01-01'):
    """Create train/validation/test splits - matches grid_sarima.py implementation"""
    train = df[df['Month'] < train_end]
    validation = df[(df['Month'] >= train_end) & (df['Month'] < val_end)]
    test = df[df['Month'] >= val_end]
    return train, validation, test

def create_dataset(series, look_back):
    """Create dataset for supervised learning"""
    X, y = [], []
    for i in range(len(series) - look_back):
        X.append(series[i:i+look_back])
        y.append(series[i+look_back])
    return np.array(X), np.array(y)

def evaluate_metrics(y_true, y_pred):
    """Calculate evaluation metrics"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }

# ------------------ MODEL IMPLEMENTATIONS ------------------ #

def run_sarima(train_val_df, test_df, order=(1, 0, 0), seasonal_order=(2, 2, 2, 12), seed=42):
    """Run SARIMA model - matches grid_sarima.py implementation"""
    np.random.seed(seed)
    
    # Convert to series and reset index to match grid search implementation
    train_val_series = train_val_df['Deaths'].reset_index(drop=True).astype(float)
    test_series = test_df['Deaths'].reset_index(drop=True).astype(float)
    
    model = SARIMAX(train_val_series, 
                    order=order, 
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False, 
                    enforce_invertibility=False)
    results = model.fit(disp=False)
    
    # Forecast for test period - matches grid search prediction method
    forecast = results.predict(start=len(train_val_series), 
                              end=len(train_val_series) + len(test_series) - 1)
    
    # Safeguard to ensure forecast length matches test series length
    forecast = forecast[:len(test_series)]
    
    return test_series.values, forecast.values

def run_lstm(train_val_data, test_data, lookback, batch_size, epochs, seed):
    """Run LSTM model"""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    # Combine train and validation for training
    combined_data = np.concatenate([train_val_data, test_data])
    
    # Create training data from combined train+val
    X_train, y_train = create_dataset(train_val_data, lookback)
    X_train = X_train.reshape((X_train.shape[0], lookback, 1))
    
    # Build model
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(lookback, 1), return_sequences=True),
        LSTM(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    
    # Autoregressive prediction for test period
    current_input = train_val_data[-lookback:].reshape((1, lookback, 1))
    preds = []
    
    for _ in range(len(test_data)):
        pred = model.predict(current_input, verbose=0)[0][0]
        preds.append(pred)
        current_input = np.append(current_input[:, 1:, :], [[[pred]]], axis=1)
    
    return test_data, np.array(preds)

def run_tcn(train_val_data, test_data, lookback, batch_size, epochs, seed):
    """Run TCN model"""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    # Create training data from combined train+val
    X_train, y_train = create_dataset(train_val_data, lookback)
    X_train = X_train.reshape((X_train.shape[0], lookback, 1))
    
    # Build TCN model
    model = Sequential([
        TCN(input_shape=(lookback, 1), dilations=[1, 2, 4, 8], 
            nb_filters=64, kernel_size=3, dropout_rate=0.1),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    
    # Autoregressive prediction for test period
    current_input = train_val_data[-lookback:].reshape((1, lookback, 1))
    preds = []
    
    for _ in range(len(test_data)):
        pred = model.predict(current_input, verbose=0)[0][0]
        preds.append(pred)
        current_input = np.append(current_input[:, 1:, :], [[[pred]]], axis=1)
    
    return test_data, np.array(preds)

def build_seq2seq_model(lookback, encoder_units=128, decoder_units=128, use_attention=False):
    """Build seq2seq model with optional attention"""
    encoder_inputs = Input(shape=(lookback, 1), name='encoder_input')
    
    if use_attention:
        encoder_gru = GRU(encoder_units, return_sequences=True, return_state=True, name='encoder_gru')
        encoder_outputs, encoder_state = encoder_gru(encoder_inputs)
        
        if encoder_units != decoder_units:
            encoder_outputs_proj = Dense(decoder_units, name='encoder_proj')(encoder_outputs)
            encoder_state = Dense(decoder_units, name='state_transform')(encoder_state)
        else:
            encoder_outputs_proj = encoder_outputs
            
        decoder_inputs = Input(shape=(1, 1), name='decoder_input')
        decoder_gru = GRU(decoder_units, return_sequences=True, return_state=True, name='decoder_gru')
        decoder_outputs, _ = decoder_gru(decoder_inputs, initial_state=encoder_state)
        
        attention_layer = Attention(name='attention')
        context_vector = attention_layer([decoder_outputs, encoder_outputs_proj])
        decoder_combined = Concatenate(axis=-1)([decoder_outputs, context_vector])
        decoder_hidden = Dense(decoder_units, activation='relu', name='decoder_hidden')(decoder_combined)
        decoder_outputs = Dense(1, name='output_dense')(decoder_hidden)
    else:
        encoder_gru = GRU(encoder_units, return_state=True, name='encoder_gru')
        _, encoder_state = encoder_gru(encoder_inputs)
        
        if encoder_units != decoder_units:
            encoder_state = Dense(decoder_units, name='state_transform')(encoder_state)
            
        decoder_inputs = Input(shape=(1, 1), name='decoder_input')
        decoder_gru = GRU(decoder_units, return_sequences=True, name='decoder_gru')
        decoder_outputs = decoder_gru(decoder_inputs, initial_state=encoder_state)
        decoder_outputs = Dense(1, name='decoder_dense')(decoder_outputs)
    
    return Model([encoder_inputs, decoder_inputs], decoder_outputs)

def run_seq2seq(train_val_data, test_data, lookback, batch_size, epochs, seed, 
                encoder_units=128, decoder_units=128, use_attention=False):
    """Run seq2seq model with attention"""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    # Add scaling for better convergence
    scaler = MinMaxScaler()
    full_series = np.concatenate([train_val_data, test_data])
    scaled_full = scaler.fit_transform(full_series.reshape(-1, 1)).flatten()
    
    train_val_scaled = scaled_full[:len(train_val_data)]
    test_scaled = scaled_full[len(train_val_data):]
    
    # Prepare training data
    X_train, y_train = create_dataset(train_val_scaled, lookback)
    X_train = X_train.reshape((X_train.shape[0], lookback, 1))
    decoder_input_train = np.zeros((X_train.shape[0], 1, 1))
    y_train = y_train.reshape((-1, 1, 1))
    
    # Build and compile model
    model = build_seq2seq_model(lookback, encoder_units, decoder_units, use_attention)
    model.compile(optimizer=Adam(learning_rate=0.001, clipnorm=1.0), loss='mse', metrics=['mae'])
    
    # Train model
    early_stopping = EarlyStopping(monitor='loss', patience=15, restore_best_weights=True)
    model.fit([X_train, decoder_input_train], y_train, epochs=epochs, batch_size=batch_size, 
              verbose=0, callbacks=[early_stopping], validation_split=0.1)
    
    # Autoregressive prediction
    preds_scaled = []
    current_sequence = train_val_scaled[-lookback:].copy()
    
    for _ in range(len(test_data)):
        encoder_input = current_sequence.reshape((1, lookback, 1))
        decoder_input = np.zeros((1, 1, 1))
        pred_scaled = model.predict([encoder_input, decoder_input], verbose=0)[0, 0, 0]
        preds_scaled.append(pred_scaled)
        current_sequence = np.append(current_sequence[1:], pred_scaled)
    
    # Inverse transform predictions
    preds_original = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
    return test_data, preds_original

class PositionalEncoding(tf.keras.layers.Layer):
    """Positional encoding for transformer"""
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

def run_transformer(train_val_data, test_data, lookback, batch_size, epochs, seed, d_model=64, n_heads=2):
    """Run transformer model"""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    # Add scaling
    scaler = MinMaxScaler()
    full_series = np.concatenate([train_val_data, test_data])
    scaled_full = scaler.fit_transform(full_series.reshape(-1, 1)).flatten()
    
    train_val_scaled = scaled_full[:len(train_val_data)]
    test_scaled = scaled_full[len(train_val_data):]
    
    # Prepare data
    X_train, y_train = create_dataset(train_val_scaled, lookback)
    X_train = X_train.reshape((X_train.shape[0], lookback, 1))
    y_train = y_train.reshape((-1, 1))
    
    # Build transformer model
    inputs = Input(shape=(lookback, 1))
    x = Dense(d_model)(inputs)
    x = PositionalEncoding(d_model)(x)
    
    attn_output = MultiHeadAttention(num_heads=n_heads, key_dim=d_model)(x, x)
    x = Add()([x, attn_output])
    x = LayerNormalization()(x)
    
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.1)(x)
    outputs = Dense(1)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0)
    
    # Autoregressive forecasting
    current_seq = train_val_scaled[-lookback:].copy()
    preds_scaled = []
    
    for _ in range(len(test_data)):
        input_seq = current_seq.reshape((1, lookback, 1))
        pred_scaled = model.predict(input_seq, verbose=0)[0][0]
        preds_scaled.append(pred_scaled)
        current_seq = np.append(current_seq[1:], pred_scaled)
    
    preds_original = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
    return test_data, preds_original

# ------------------ MAIN EVALUATION LOOP ------------------ #

def main():
    """Main evaluation function"""
    print("Loading and preprocessing data...")
    data = load_and_preprocess_data()
    train_data, validation_data, test_data = create_train_val_test_split(data)
    
    # Combine train and validation for final training
    train_val_data = pd.concat([train_data, validation_data], ignore_index=True)
    
    print(f"Train+Val data shape: {train_val_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    
    # Results storage for final comparison
    all_results = {}
    
    # Models to evaluate
    models_to_evaluate = ['transformer', 'sarima', 'lstm', 'tcn', 'seq2seq_attn']
    
    for model_name in models_to_evaluate:
        print(f"\n--- Evaluating {model_name.upper()} ---")
        
        model_dir = os.path.join(RESULTS_DIR, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Get optimal parameters for this model
        params = OPTIMAL_PARAMS[model_name]
        
        seed_results = {}  # Store results by seed
        
        # Loop through each seed
        for seed in SEEDS:
            print(f"\n  Processing seed {seed}...")
            
            seed_dir = os.path.join(model_dir, f'seed_{seed}')
            os.makedirs(seed_dir, exist_ok=True)
            
            trial_results = []
            
            # Run multiple trials for this seed
            for trial in range(TRIALS_PER_SEED):
                # Create unique seed for each trial
                trial_seed = seed + trial * 1000  # Ensure different seeds for each trial
                print(f"    Trial {trial + 1}/{TRIALS_PER_SEED} (trial_seed: {trial_seed})")
                
                try:
                    if model_name == 'sarima':
                        y_true, y_pred = run_sarima(train_val_data, test_data, 
                                                  order=params['order'], 
                                                  seasonal_order=params['seasonal_order'], 
                                                  seed=trial_seed)
                    
                    elif model_name == 'lstm':
                        y_true, y_pred = run_lstm(train_val_data['Deaths'].values, 
                                                test_data['Deaths'].values,
                                                params['lookback'], params['batch_size'], 
                                                params['epochs'], trial_seed)
                    
                    elif model_name == 'tcn':
                        y_true, y_pred = run_tcn(train_val_data['Deaths'].values, 
                                               test_data['Deaths'].values,
                                               params['lookback'], params['batch_size'], 
                                               params['epochs'], trial_seed)
                    
                    elif model_name == 'seq2seq_attn':
                        y_true, y_pred = run_seq2seq(train_val_data['Deaths'].values, 
                                                   test_data['Deaths'].values,
                                                   params['lookback'], params['batch_size'], 
                                                   params['epochs'], trial_seed,
                                                   params['encoder_units'], params['decoder_units'], 
                                                   use_attention=False)
                    
                    elif model_name == 'transformer':
                        y_true, y_pred = run_transformer(train_val_data['Deaths'].values, 
                                                       test_data['Deaths'].values,
                                                       params['lookback'], params['batch_size'], 
                                                       params['epochs'], trial_seed,
                                                       params['d_model'], params['n_heads'])
                    
                    # Calculate metrics
                    metrics = evaluate_metrics(y_true, y_pred)
                    metrics['Trial'] = trial + 1
                    metrics['Seed'] = seed
                    metrics['Trial_Seed'] = trial_seed
                    trial_results.append(metrics)
                    
                    # Save individual trial predictions
                    trial_df = pd.DataFrame({
                        'True': y_true,
                        'Predicted': y_pred
                    })
                    trial_df.to_csv(os.path.join(seed_dir, f'trial_{trial + 1}_predictions.csv'), index=False)
                    
                except Exception as e:
                    print(f"      Error in trial {trial + 1}: {str(e)}")
                    continue
            
            # Process results for this seed
            if trial_results:
                # Save all trial results for this seed
                trial_results_df = pd.DataFrame(trial_results)
                trial_results_df.to_csv(os.path.join(seed_dir, 'all_trials_metrics.csv'), index=False)
                
                # Calculate and save summary statistics for this seed
                seed_summary = trial_results_df[['RMSE', 'MAE', 'MAPE']].agg(['mean', 'std', 'min', 'max'])
                seed_summary.to_csv(os.path.join(seed_dir, 'seed_summary_statistics.csv'))
                
                # Store seed results for model comparison
                seed_results[seed] = {
                    'mean_rmse': trial_results_df['RMSE'].mean(),
                    'std_rmse': trial_results_df['RMSE'].std(),
                    'mean_mae': trial_results_df['MAE'].mean(),
                    'std_mae': trial_results_df['MAE'].std(),
                    'mean_mape': trial_results_df['MAPE'].mean(),
                    'std_mape': trial_results_df['MAPE'].std(),
                    'trials_completed': len(trial_results_df)
                }
                
                print(f"    Seed {seed} completed {len(trial_results_df)} trials")
                print(f"    Mean RMSE: {trial_results_df['RMSE'].mean():.4f} ± {trial_results_df['RMSE'].std():.4f}")
        
        # Save seed comparison for this model
        if seed_results:
            seed_comparison_df = pd.DataFrame(seed_results).T
            seed_comparison_df = seed_comparison_df.round(4)
            seed_comparison_df.to_csv(os.path.join(model_dir, 'seed_comparison.csv'))
            
            # Calculate overall model statistics across seeds
            all_seed_means = [seed_results[seed]['mean_rmse'] for seed in seed_results]
            all_seed_mae_means = [seed_results[seed]['mean_mae'] for seed in seed_results]
            all_seed_mape_means = [seed_results[seed]['mean_mape'] for seed in seed_results]
            
            all_results[model_name] = {
                'mean_rmse_across_seeds': np.mean(all_seed_means),
                'std_rmse_across_seeds': np.std(all_seed_means),
                'mean_mae_across_seeds': np.mean(all_seed_mae_means),
                'std_mae_across_seeds': np.std(all_seed_mae_means),
                'mean_mape_across_seeds': np.mean(all_seed_mape_means),
                'std_mape_across_seeds': np.std(all_seed_mape_means),
                'seeds_completed': len(seed_results)
            }
            
            print(f"  Model {model_name} overall RMSE: {np.mean(all_seed_means):.4f} ± {np.std(all_seed_means):.4f}")
    
    # Create final comparison table across all models
    if all_results:
        final_comparison_df = pd.DataFrame(all_results).T
        final_comparison_df = final_comparison_df.round(4)
        final_comparison_df.to_csv(os.path.join(RESULTS_DIR, 'final_model_comparison.csv'))
        
        print("\n" + "="*60)
        print("FINAL MODEL COMPARISON (ACROSS ALL SEEDS)")
        print("="*60)
        print(final_comparison_df.to_string())
        print(f"\nResults saved to: {RESULTS_DIR}/")
        print("\nDirectory structure:")
        print("test_results/")
        print("├── final_model_comparison.csv")
        print("├── model_name/")
        print("│   ├── seed_comparison.csv")
        print("│   └── seed_X/")
        print("│       ├── all_trials_metrics.csv")
        print("│       ├── seed_summary_statistics.csv")
        print("│       └── trial_X_predictions.csv")

if __name__ == "__main__":
    main()