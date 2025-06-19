import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (LSTM, Dense, Conv1D, Flatten, GRU, Input, 
                                   Add, ReLU, Lambda, Dropout, BatchNormalization,
                                   LayerNormalization, MultiHeadAttention, 
                                   GlobalAveragePooling1D, Embedding)
import tensorflow as tf
from statsmodels.tsa.statespace.sarimax import SARIMAX
try:
    from tcn import TCN
    TCN_AVAILABLE = True
except ImportError:
    TCN_AVAILABLE = False
    print("Warning: TCN not available. Install with: pip install keras-tcn")
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

MODEL_TYPE = 'transformer'
TRIAL_MODE = 'fixed_seed'
SEEDS = [42] if TRIAL_MODE == 'fixed_seed' else [123, 456, 11, 245, 56712, 23467, 98, 38, 1506, 42]
TRIALS_PER_CONFIG = 30
LOOKBACKS = [3, 5, 7, 9, 11, 12]
BATCH_SIZES = [16, 32, 64]
EPOCHS_LIST = [50, 100, 150]
HIDDEN_UNITS = [32, 64, 128]
LEARNING_RATES = [0.001, 0.01]
DROPOUT_RATES = [0.1, 0.2, 0.3]

DATA_PATH = 'data/state_month_overdose.xlsx'
RESULTS_DIR = 'results_paper'

scaler = MinMaxScaler()

def load_and_preprocess_data():
    df = pd.read_excel(DATA_PATH)
    df['Deaths'] = df['Deaths'].apply(lambda x: 0 if x == 'Suppressed' else int(x))
    df['Month'] = pd.to_datetime(df['Month'])
    df = df.groupby('Month').agg({'Deaths': 'sum'}).reset_index()
    df['Deaths_scaled'] = scaler.fit_transform(df[['Deaths']])
    return df

def create_train_val_test_split_lstm(df: pd.DataFrame, 
                                  train_end: str = '2019-01-01',
                                  val_end: str = '2020-01-01', 
                                  test_end: str = '2020-12-01'):
    train = df[df['Month'] < train_end]
    validation = df[(df['Month'] >= train_end) & (df['Month'] < val_end)]
    test = df[(df['Month'] >= val_end)]
    return train, validation, test

def create_dataset(series, look_back):
    X, y = [], []
    for i in range(len(series) - look_back):
        X.append(series[i:i+look_back])
        y.append(series[i+look_back])
    return np.array(X), np.array(y)

def evaluate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'PI_Width': 0,
        'CI_Coverage': 0,
        'PI_Overlap': 0
    }

def run_lstm(train, test, look_back, batch_size, epochs, hidden_units=50, lr=0.001, dropout=0.2, seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)

    train_vals = train.values
    test_vals = test.values
    X_train, y_train = create_dataset(train_vals, look_back)
    X_test, y_test = create_dataset(np.concatenate([train_vals[-look_back:], test_vals]), look_back)
    X_train = X_train.reshape((X_train.shape[0], look_back, 1))
    X_test = X_test.reshape((X_test.shape[0], look_back, 1))

    model = Sequential([
        LSTM(hidden_units, activation='relu', input_shape=(look_back, 1), dropout=dropout),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error')
    early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[early_stop])

    preds = []
    current_input = X_test[0].reshape((1, look_back, 1))
    for _ in range(len(y_test)):
        pred = model.predict(current_input, verbose=0)[0][0]
        preds.append(pred)
        current_input = np.append(current_input[:, 1:, :], [[[pred]]], axis=1)

    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    preds_inv = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    return y_test_inv, preds_inv

def run_tcn(train, test, look_back, batch_size, epochs, hidden_units=64, lr=0.001, dropout=0.2, seed=42):
    if not TCN_AVAILABLE:
        raise ImportError("TCN not available. Install with: pip install keras-tcn")

    np.random.seed(seed)
    tf.random.set_seed(seed)

    train_vals = train.values
    test_vals = test.values
    X_train, y_train = create_dataset(train_vals, look_back)
    extended_test = np.concatenate([train_vals[-look_back:], test_vals])
    y_test = extended_test[look_back:]
    X_train = X_train.reshape((X_train.shape[0], look_back, 1))

    model = Sequential([
        TCN(nb_filters=hidden_units, kernel_size=3, dilations=[1, 2, 4, 8], dropout_rate=dropout, input_shape=(look_back, 1)),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error')
    early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[early_stop])

    current_input = np.array(train_vals[-look_back:]).reshape((1, look_back, 1))
    preds = []
    for _ in range(len(test_vals)):
        pred = model.predict(current_input, verbose=0)[0][0]
        preds.append(pred)
        current_input = np.append(current_input[:, 1:, :], [[[pred]]], axis=1)

    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    preds_inv = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    return y_test_inv, preds_inv

def run_seq2seq(train, test, look_back, batch_size, epochs, hidden_units=50, lr=0.001, dropout=0.2, seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)

    train_vals = train.values
    test_vals = test.values
    X_train, y_train = create_dataset(train_vals, look_back)
    X_test, y_test = create_dataset(np.concatenate([train_vals[-look_back:], test_vals]), look_back)
    X_train = X_train.reshape((X_train.shape[0], look_back, 1))
    X_test = X_test.reshape((X_test.shape[0], look_back, 1))

    encoder_inputs = Input(shape=(look_back, 1))
    encoder = GRU(hidden_units, return_sequences=True, return_state=True, dropout=dropout)
    encoder_outputs, encoder_state = encoder(encoder_inputs)

    decoder_inputs = Input(shape=(look_back, 1))
    decoder_gru = GRU(hidden_units, return_sequences=True, return_state=True, dropout=dropout)
    decoder_outputs, _ = decoder_gru(decoder_inputs, initial_state=encoder_state)

    encoder_proj = Dense(hidden_units)(encoder_outputs)
    attention_scores = tf.keras.layers.Dot(axes=[2, 2])([decoder_outputs, encoder_proj])
    attention_weights = tf.keras.layers.Softmax(axis=-1)(attention_scores)
    context_vector = tf.keras.layers.Dot(axes=[2, 1])([attention_weights, encoder_proj])
    decoder_combined = tf.keras.layers.Concatenate(axis=-1)([decoder_outputs, context_vector])
    output = Dense(1, activation='linear')(decoder_combined)
    final_output = Lambda(lambda x: x[:, -1, :])(output)

    model = Model([encoder_inputs, decoder_inputs], final_output)
    model.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error')
    early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    model.fit([X_train, X_train], y_train, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[early_stop])

    preds = []
    current_input = X_test[0].reshape((1, look_back, 1))
    for _ in range(len(y_test)):
        pred = model.predict([current_input, current_input], verbose=0)[0][0]
        preds.append(pred)
        current_input = np.append(current_input[:, 1:, :], [[[pred]]], axis=1)

    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    preds_inv = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    return y_test_inv, preds_inv



def run_transformer(train, test, look_back, batch_size, epochs, d_model=64, num_heads=4, 
                   dff=256, num_layers=2, lr=0.001, dropout=0.1, seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)

    train_vals = train.values
    test_vals = test.values

    X_train, y_train = create_dataset(train_vals, look_back)
    X_test, y_test = create_dataset(np.concatenate([train_vals[-look_back:], test_vals]), look_back)
    X_train = X_train.reshape((X_train.shape[0], look_back, 1))
    X_test = X_test.reshape((X_test.shape[0], look_back, 1))

    inputs = Input(shape=(look_back, 1))
    x = Dense(d_model)(inputs)
    pos_encoding = positional_encoding(look_back, d_model)
    x = x + pos_encoding

    for _ in range(num_layers):
        attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model//num_heads, dropout=dropout)(x, x)
        x = LayerNormalization(epsilon=1e-6)(x + attn_output)
        ffn_output = Sequential([
            Dense(dff, activation='relu'),
            Dropout(dropout),
            Dense(d_model)
        ])(x)
        x = LayerNormalization(epsilon=1e-6)(x + ffn_output)

    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout)(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)

    def lr_schedule(epoch): return lr * (0.95 ** (epoch // 10))
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
    early_stop = EarlyStopping(monitor='loss', patience=15, restore_best_weights=True)

    model.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[early_stop, lr_scheduler])

    preds = []
    current_input = X_test[0].reshape((1, look_back, 1))
    for _ in range(len(y_test)):
        pred = model.predict(current_input, verbose=0)[0][0]
        preds.append(pred)
        current_input = np.append(current_input[:, 1:, :], [[[pred]]], axis=1)

    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    preds_inv = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    return y_test_inv, preds_inv

def positional_encoding(length, depth):
    depth = depth / 2
    positions = np.arange(length)[:, np.newaxis]
    depths = np.arange(depth)[np.newaxis, :] / depth
    angle_rates = 1 / (10000**depths)
    angle_rads = positions * angle_rates
    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)
    return tf.cast(pos_encoding, dtype=tf.float32)

# ------------------ MAIN EXPERIMENT LOOP ------------------ #
if __name__ == "__main__":
    data = load_and_preprocess_data()
    train_data, validation_data, test_data = create_train_val_test_split_lstm(data)

    train_series = train_data['Deaths_scaled']
    val_series = validation_data['Deaths_scaled']
    
    # Define hyperparameter combinations based on model type
    if MODEL_TYPE == 'transformer':
        hyperparams = [
            (lb, bs, ep, hu, lr, dr) 
            for lb in LOOKBACKS 
            for bs in BATCH_SIZES 
            for ep in EPOCHS_LIST
            for hu in [64, 128]  # d_model for transformer
            for lr in LEARNING_RATES
            for dr in DROPOUT_RATES
        ]
    else:
        hyperparams = [
            (lb, bs, ep, hu, lr, dr) 
            for lb in LOOKBACKS 
            for bs in BATCH_SIZES 
            for ep in EPOCHS_LIST
            for hu in HIDDEN_UNITS
            for lr in LEARNING_RATES
            for dr in DROPOUT_RATES
        ]
    
    for seed in SEEDS:
        for look_back, bs, ep, hu, lr, dr in hyperparams:
            config_name = f'lookback_{look_back}_bs_{bs}_epochs_{ep}_hu_{hu}_lr_{lr}_dr_{dr}'
            print(f"Running {MODEL_TYPE}: {config_name}")
            
            base_dir = os.path.join(RESULTS_DIR, MODEL_TYPE,
                'fixed_seed_variability' if TRIAL_MODE == 'fixed_seed' else f'multi_seed_variability/seed_{seed}',
                config_name)
            os.makedirs(base_dir, exist_ok=True)
            
            metrics_all_test = []
            metrics_all_train = []
            
            for trial in range(TRIALS_PER_CONFIG):
                try:
                    if MODEL_TYPE == 'transformer':
                        y_train_true, y_train_pred = run_transformer(
                            train_series, train_series.iloc[look_back:], 
                            look_back, bs, ep, hu, 4, 256, 2, lr, dr, seed
                        )
                        y_test_true, y_test_pred = run_transformer(
                            train_series, val_series, 
                            look_back, bs, ep, hu, 4, 256, 2, lr, dr, seed
                        )
                    elif MODEL_TYPE == 'lstm':
                        y_train_true, y_train_pred = run_lstm(
                            train_series, train_series.iloc[look_back:], 
                            look_back, bs, ep, hu, lr, dr, seed
                        )
                        y_test_true, y_test_pred = run_lstm(
                            train_series, val_series, 
                            look_back, bs, ep, hu, lr, dr, seed
                        )
                    elif MODEL_TYPE == 'tcn':
                        y_train_true, y_train_pred = run_tcn(
                            train_series, train_series.iloc[look_back:], 
                            look_back, bs, ep, hu, lr, dr, seed
                        )
                        y_test_true, y_test_pred = run_tcn(
                            train_series, val_series, 
                            look_back, bs, ep, hu, lr, dr, seed
                        )
                    elif MODEL_TYPE == 'seq2seq':
                        y_train_true, y_train_pred = run_seq2seq(
                            train_series, train_series.iloc[look_back:], 
                            look_back, bs, ep, hu, lr, dr, seed
                        )
                        y_test_true, y_test_pred = run_seq2seq(
                            train_series, val_series, 
                            look_back, bs, ep, hu, lr, dr, seed
                        )
                    else:
                        raise ValueError(f"Unsupported MODEL_TYPE: {MODEL_TYPE}")

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
                    
                except Exception as e:
                    print(f"Error in trial {trial}: {str(e)}")
                    continue
            
            # Save summary metrics
            if metrics_all_train and metrics_all_test:
                pd.DataFrame(metrics_all_train).agg(['mean', 'std']).to_csv(
                    os.path.join(base_dir, 'summary_metrics_train.csv'))
                pd.DataFrame(metrics_all_test).agg(['mean', 'std']).to_csv(
                    os.path.join(base_dir, 'summary_metrics_test.csv'))