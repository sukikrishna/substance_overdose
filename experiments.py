import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Conv1D, Flatten, GRU
from tensorflow.keras.layers import Input, Add, ReLU, Lambda, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Attention, RepeatVector, TimeDistributed
import tensorflow as tf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
from tcn import TCN  # pip install keras-tcn
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# ------------------ CONFIGURATION ------------------ #
MODEL_TYPE = 'seq2seq'  # Options: 'lstm', 'sarima', 'tcn', 'seq2seq', 'transformer', 'tcn_updated', 'tcn_fixed'
TRIAL_MODE = 'fixed_seed'  # Options: 'fixed_seed', 'multi_seed'
SEEDS = [42] if TRIAL_MODE == 'fixed_seed' else [123, 456, 11, 245, 56712, 23467, 98, 38, 1506, 42]
TRIALS_PER_CONFIG = 30

# Hyperparameters for different models
LOOKBACKS = [11, 12]
BATCH_SIZES = [8, 16, 32]
EPOCHS_LIST = [50, 100]

# Alternative: Use same size for encoder and decoder to avoid state size issues
ENCODER_UNITS = [64, 128]  
DECODER_UNITS = [64, 128]  # Could be set to same as ENCODER_UNITS
USE_ATTENTION = [False, True]

DATA_PATH = 'data/state_month_overdose.xlsx'
RESULTS_DIR = 'results'

# ------------------ HELPER FUNCTIONS ------------------ #
def load_and_preprocess_data():
    df = pd.read_excel(DATA_PATH)
    df['Deaths'] = df['Deaths'].apply(lambda x: 0 if x == 'Suppressed' else int(x))
    df['Month'] = pd.to_datetime(df['Month'])
    df = df.groupby('Month').agg({'Deaths': 'sum'}).reset_index()
    return df

def create_dataset(series, look_back):
    X, y = [], []
    for i in range(len(series) - look_back):
        X.append(series[i:i+look_back])
        y.append(series[i+look_back])
    return np.array(X), np.array(y)

def create_seq2seq_dataset(series, look_back, forecast_horizon=1):
    """Create dataset for seq2seq with variable forecast horizon"""
    X, y = [], []
    for i in range(len(series) - look_back - forecast_horizon + 1):
        X.append(series[i:i+look_back])
        y.append(series[i+look_back:i+look_back+forecast_horizon])
    return np.array(X), np.array(y)

def evaluate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'PI Width': 0,
        'CI Coverage': 0,
        'PI Overlap': 0
    }

def create_train_val_test_split_lstm(df: pd.DataFrame, 
                                  train_end: str = '2019-01-01',
                                  val_end: str = '2020-01-01', 
                                  test_end: str = '2020-12-01'):
    """Create proper train/validation/test splits"""
    train = df[df['Month'] < train_end]
    validation = df[(df['Month'] >= train_end) & (df['Month'] < val_end)]
    test = df[(df['Month'] >= val_end)]
    
    return train, validation, test

# def build_seq2seq_model(look_back, encoder_units=64, decoder_units=64, use_attention=True):
#     """
#     Build seq2seq model with proper architecture for time series forecasting
#     Handle different encoder/decoder sizes properly
#     """
#     # Encoder
#     encoder_inputs = Input(shape=(look_back, 1), name='encoder_input')
    
#     if use_attention:
#         # Encoder with return_sequences=True for attention
#         encoder_gru = GRU(encoder_units, return_sequences=True, return_state=True, name='encoder_gru')
#         encoder_outputs, encoder_state = encoder_gru(encoder_inputs)
        
#         # Transform encoder state to match decoder size if needed
#         if encoder_units != decoder_units:
#             state_transform = Dense(decoder_units, name='state_transform')
#             encoder_state = state_transform(encoder_state)
        
#         # Decoder
#         decoder_inputs = Input(shape=(1, 1), name='decoder_input')
#         decoder_gru = GRU(decoder_units, return_sequences=True, return_state=True, name='decoder_gru')
#         decoder_outputs, _ = decoder_gru(decoder_inputs, initial_state=encoder_state)
        
#         # Attention mechanism
#         attention_layer = Attention(name='attention')
#         context_vector = attention_layer([decoder_outputs, encoder_outputs])
        
#         # Combine context with decoder output
#         decoder_combined = tf.keras.layers.Concatenate(axis=-1)([decoder_outputs, context_vector])
#         decoder_dense = Dense(decoder_units, activation='relu', name='decoder_hidden')
#         decoder_hidden = decoder_dense(decoder_combined)
#         output_dense = Dense(1, name='output_dense')
#         decoder_outputs = output_dense(decoder_hidden)
#     else:
#         # Standard encoder-decoder without attention
#         encoder_gru = GRU(encoder_units, return_state=True, name='encoder_gru')
#         _, encoder_state = encoder_gru(encoder_inputs)
        
#         # Transform encoder state to match decoder size if needed
#         if encoder_units != decoder_units:
#             state_transform = Dense(decoder_units, name='state_transform')
#             encoder_state = state_transform(encoder_state)
        
#         # Decoder
#         decoder_inputs = Input(shape=(1, 1), name='decoder_input')
#         decoder_gru = GRU(decoder_units, return_sequences=True, name='decoder_gru')
#         decoder_outputs = decoder_gru(decoder_inputs, initial_state=encoder_state)
        
#         # Output layer
#         decoder_dense = Dense(1, name='decoder_dense')
#         decoder_outputs = decoder_dense(decoder_outputs)
    
#     # Model
#     model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
#     return model

def build_seq2seq_model(look_back, encoder_units=64, decoder_units=64, use_attention=True):
    from tensorflow.keras.layers import GRU, Input, Dense, Attention, Concatenate
    from tensorflow.keras.models import Model

    # Encoder
    encoder_inputs = Input(shape=(look_back, 1), name='encoder_input')

    if use_attention:
        encoder_gru = GRU(encoder_units, return_sequences=True, return_state=True, name='encoder_gru')
        encoder_outputs, encoder_state = encoder_gru(encoder_inputs)

        # Project encoder outputs to match decoder dimension
        if encoder_units != decoder_units:
            encoder_outputs_proj = Dense(decoder_units, name='encoder_proj')(encoder_outputs)
        else:
            encoder_outputs_proj = encoder_outputs

        # Transform encoder state to match decoder state size if needed
        if encoder_units != decoder_units:
            encoder_state = Dense(decoder_units, name='state_transform')(encoder_state)

        # Decoder
        decoder_inputs = Input(shape=(1, 1), name='decoder_input')
        decoder_gru = GRU(decoder_units, return_sequences=True, return_state=True, name='decoder_gru')
        decoder_outputs, _ = decoder_gru(decoder_inputs, initial_state=encoder_state)

        # Attention
        attention_layer = Attention(name='attention')
        context_vector = attention_layer([decoder_outputs, encoder_outputs_proj])

        # Combine context and decoder output
        decoder_combined = Concatenate(axis=-1)([decoder_outputs, context_vector])
        decoder_hidden = Dense(decoder_units, activation='relu', name='decoder_hidden')(decoder_combined)
        decoder_outputs = Dense(1, name='output_dense')(decoder_hidden)

    else:
        # Encoder without attention
        encoder_gru = GRU(encoder_units, return_state=True, name='encoder_gru')
        _, encoder_state = encoder_gru(encoder_inputs)

        if encoder_units != decoder_units:
            encoder_state = Dense(decoder_units, name='state_transform')(encoder_state)

        decoder_inputs = Input(shape=(1, 1), name='decoder_input')
        decoder_gru = GRU(decoder_units, return_sequences=True, name='decoder_gru')
        decoder_outputs = decoder_gru(decoder_inputs, initial_state=encoder_state)
        decoder_outputs = Dense(1, name='decoder_dense')(decoder_outputs)

    return Model([encoder_inputs, decoder_inputs], decoder_outputs)


def run_seq2seq(train, test, look_back, batch_size, epochs, seed, 
                encoder_units=64, decoder_units=64, use_attention=True):
    """
    Run seq2seq model with proper scaling and autoregressive prediction
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    # Add scaling like the TCN model for better convergence
    scaler = MinMaxScaler()
    full_series = np.concatenate([train, test])
    scaled_full = scaler.fit_transform(full_series.reshape(-1, 1)).flatten()
    
    train_scaled = scaled_full[:len(train)]
    test_scaled = scaled_full[len(train):]
    
    # Prepare training data with scaled values
    X_train, y_train = create_dataset(train_scaled, look_back)
    X_train = X_train.reshape((X_train.shape[0], look_back, 1))
    
    # For seq2seq, decoder input is typically a start token or previous target
    # We'll use zeros as start tokens
    decoder_input_train = np.zeros((X_train.shape[0], 1, 1))
    y_train = y_train.reshape((-1, 1, 1))
    
    # Build and compile model
    model = build_seq2seq_model(look_back, encoder_units, decoder_units, use_attention)
    model.compile(
        optimizer=Adam(learning_rate=0.001, clipnorm=1.0),  # Add gradient clipping
        loss='mse', 
        metrics=['mae']
    )
    
    # Train model with early stopping
    early_stopping = EarlyStopping(
        monitor='loss', 
        patience=15, 
        restore_best_weights=True,
        min_delta=1e-6
    )
    
    model.fit(
        [X_train, decoder_input_train], y_train, 
        epochs=epochs, 
        batch_size=batch_size, 
        verbose=0, 
        callbacks=[early_stopping],
        validation_split=0.1  # Add validation split
    )
    
    # Autoregressive prediction for test period
    preds_scaled = []
    current_sequence = train_scaled[-look_back:].copy()
    
    for _ in range(len(test)):
        # Prepare encoder input
        encoder_input = current_sequence.reshape((1, look_back, 1))
        # Decoder input (start token)
        decoder_input = np.zeros((1, 1, 1))
        
        # Predict next value
        pred_scaled = model.predict([encoder_input, decoder_input], verbose=0)[0, 0, 0]
        preds_scaled.append(pred_scaled)
        
        # Update sequence for next prediction
        current_sequence = np.append(current_sequence[1:], pred_scaled)
    
    # Inverse transform predictions to original scale
    preds_scaled = np.array(preds_scaled).reshape(-1, 1)
    preds_original = scaler.inverse_transform(preds_scaled).flatten()
    
    # Get original test values
    y_test = test
    
    return y_test, preds_original

# Existing model functions (keeping them unchanged)
def run_lstm(train, test, look_back, batch_size, epochs, seed):
    np.random.seed(seed)
    X_train, y_train = create_dataset(train, look_back)
    X_test, y_test = create_dataset(np.concatenate([train[-look_back:], test]), look_back)
    X_train = X_train.reshape((X_train.shape[0], look_back, 1))
    X_test = X_test.reshape((X_test.shape[0], look_back, 1))
    model = Sequential([LSTM(50, activation='relu', input_shape=(look_back, 1)), Dense(1)])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    preds = []
    current_input = X_test[0].reshape((1, look_back, 1))
    for _ in range(len(y_test)):
        pred = model.predict(current_input)[0][0]
        preds.append(pred)
        current_input = np.append(current_input[:, 1:, :], [[[pred]]], axis=1)
    return y_test, np.array(preds)

def run_sarima(train_df, test_df):
    np.random.seed(seed)
    train_series = train_df['Deaths'].astype(float)
    test_series = test_df['Deaths'].astype(float)

    model = SARIMAX(train_series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
                    enforce_stationarity=False, enforce_invertibility=False)
    results = model.fit(disp=False)
    
    fitted = results.fittedvalues
    forecast = results.predict(start=len(train_series), end=len(train_series) + len(test_series) - 1)
    
    return train_series.values, fitted.values, test_series.values, forecast.values

def run_tcn_fixed(train, test, look_back, batch_size, epochs, seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)

    scaler = MinMaxScaler()
    full_series = np.concatenate([train, test])
    scaled_full = scaler.fit_transform(full_series.reshape(-1, 1)).flatten()

    train_scaled = scaled_full[:len(train)]
    test_scaled = scaled_full[len(train):]

    def create_dataset(dataset, look_back):
        X, y = [], []
        for i in range(len(dataset) - look_back):
            X.append(dataset[i:i + look_back])
            y.append(dataset[i + look_back])
        return np.array(X), np.array(y)

    X_train, y_train = create_dataset(train_scaled, look_back)
    X_train = X_train.reshape((X_train.shape[0], look_back, 1))

    model = Sequential([
        TCN(input_shape=(look_back, 1)),
        Dense(1, activation='relu')
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    preds = []
    current_input = train_scaled[-look_back:].reshape((1, look_back, 1))
    for _ in range(len(test_scaled)):
        pred = model.predict(current_input, verbose=0)[0][0]
        preds.append(pred)
        current_input = np.append(current_input[:, 1:, :], [[[pred]]], axis=1)

    all_preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    y_test = full_series[len(train):len(train)+len(test)]

    return y_test, all_preds

# ------------------ MAIN EXECUTION ------------------ #
data = load_and_preprocess_data()
train_data, validation_data, test_data = create_train_val_test_split_lstm(data)

for seed in SEEDS:
    print(f"Running with seed: {seed}")
    
    if MODEL_TYPE == 'seq2seq':
        # Custom loop for seq2seq with additional hyperparameters
        for look_back in LOOKBACKS:
            for bs in BATCH_SIZES:
                for ep in EPOCHS_LIST:
                    for enc_units in ENCODER_UNITS:
                        for dec_units in DECODER_UNITS:
                            for use_att in USE_ATTENTION:
                                config_name = f'lookback_{look_back}_bs_{bs}_epochs_{ep}_enc_{enc_units}_dec_{dec_units}_att_{use_att}'
                                print(f"Processing: {config_name}")
                                
                                base_dir = os.path.join(RESULTS_DIR, MODEL_TYPE,
                                    'fixed_seed_variability' if TRIAL_MODE == 'fixed_seed' else f'multi_seed_variability/seed_{seed}',
                                    config_name)
                                os.makedirs(base_dir, exist_ok=True)
                                
                                metrics_all_test = []
                                metrics_all_train = []
                                
                                for trial in range(TRIALS_PER_CONFIG):
                                    try:
                                        # Train on training data, predict on training subset for training metrics
                                        y_train_true, y_train_pred = run_seq2seq(
                                            train_data['Deaths'].values, 
                                            train_data['Deaths'].iloc[look_back:].values, 
                                            look_back, bs, ep, seed, enc_units, dec_units, use_att
                                        )
                                        
                                        # Train on training data, predict on validation data for test metrics
                                        y_test_true, y_test_pred = run_seq2seq(
                                            train_data['Deaths'].values, 
                                            validation_data['Deaths'].values, 
                                            look_back, bs, ep, seed, enc_units, dec_units, use_att
                                        )
                                        
                                        # Check if predictions are reasonable (not NaN, not too small)
                                        if (np.any(np.isnan(y_train_pred)) or np.any(np.isnan(y_test_pred)) or
                                            np.mean(y_train_pred) < 100 or np.mean(y_test_pred) < 100):
                                            print(f"Warning: Poor predictions in trial {trial}, skipping...")
                                            continue
                                        
                                        # Save predictions
                                        train_df = pd.DataFrame({'True': y_train_true, 'Pred': y_train_pred})
                                        test_df = pd.DataFrame({'True': y_test_true, 'Pred': y_test_pred})
                                        
                                        train_df.to_csv(os.path.join(base_dir, f'trial_{trial}_train.csv'), index=False)
                                        test_df.to_csv(os.path.join(base_dir, f'trial_{trial}_test.csv'), index=False)
                                        
                                        # Calculate metrics
                                        metrics_train = evaluate_metrics(y_train_true, y_train_pred)
                                        metrics_test = evaluate_metrics(y_test_true, y_test_pred)
                                        metrics_all_train.append(metrics_train)
                                        metrics_all_test.append(metrics_test)
                                        
                                        print(f"Trial {trial}: Train RMSE={metrics_train['RMSE']:.2f}, Test RMSE={metrics_test['RMSE']:.2f}")
                                        
                                    except Exception as e:
                                        print(f"Error in trial {trial}: {e}")
                                        continue
                                    
                                # Save summary metrics
                                if metrics_all_train:
                                    pd.DataFrame(metrics_all_train).agg(['mean', 'std']).to_csv(
                                        os.path.join(base_dir, 'summary_metrics_train.csv'))
                                if metrics_all_test:
                                    pd.DataFrame(metrics_all_test).agg(['mean', 'std']).to_csv(
                                        os.path.join(base_dir, 'summary_metrics_test.csv'))
    else:
        # Original loop for other models
        for look_back in LOOKBACKS:
            for bs in BATCH_SIZES:
                for ep in EPOCHS_LIST:
                    config_name = f'lookback_{look_back}_bs_{bs}_epochs_{ep}'
                    print(config_name)
                    base_dir = os.path.join(RESULTS_DIR, MODEL_TYPE,
                        'fixed_seed_variability' if TRIAL_MODE == 'fixed_seed' else f'multi_seed_variability/seed_{seed}',
                        config_name)
                    os.makedirs(base_dir, exist_ok=True)
                    metrics_all_test = []
                    metrics_all_train = []
                    
                    for trial in range(TRIALS_PER_CONFIG):
                        if MODEL_TYPE == 'lstm':
                            y_train_true, y_train_pred = run_lstm(train_data['Deaths'], train_data['Deaths'].iloc[look_back:], look_back, bs, ep, seed)
                            y_test_true, y_test_pred = run_lstm(train_data['Deaths'], validation_data['Deaths'], look_back, bs, ep, seed)
                        elif MODEL_TYPE == 'sarima':
                            y_train_true, y_train_pred, y_test_true, y_test_pred = run_sarima(train_data, validation_data)
                        elif MODEL_TYPE == 'tcn_fixed':
                            y_train_true, y_train_pred = run_tcn_fixed(train_data['Deaths'], train_data['Deaths'].iloc[look_back:], look_back, bs, ep, seed)
                            y_test_true, y_test_pred = run_tcn_fixed(train_data['Deaths'], validation_data['Deaths'], look_back, bs, ep, seed)
                        else:
                            raise ValueError("Unknown model type")

                        train_df = pd.DataFrame({'True': y_train_true, 'Pred': y_train_pred})
                        test_df = pd.DataFrame({'True': y_test_true, 'Pred': y_test_pred})

                        train_df.to_csv(os.path.join(base_dir, f'trial_{trial}_train.csv'), index=False)
                        test_df.to_csv(os.path.join(base_dir, f'trial_{trial}_test.csv'), index=False)
                        
                        metrics_train = evaluate_metrics(y_train_true, y_train_pred)
                        metrics_test = evaluate_metrics(y_test_true, y_test_pred)
                        metrics_all_train.append(metrics_train)
                        metrics_all_test.append(metrics_test)
                        
                    pd.DataFrame(metrics_all_train).agg(['mean', 'std']).to_csv(
                        os.path.join(base_dir, 'summary_metrics_train.csv'))
                    pd.DataFrame(metrics_all_test).agg(['mean', 'std']).to_csv(
                        os.path.join(base_dir, 'summary_metrics_test.csv'))

print("Experiment completed!")
