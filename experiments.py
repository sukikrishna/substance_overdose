import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, Flatten, GRU
from tensorflow.keras.layers import Input, Add, ReLU, Lambda, Dense
from tensorflow.keras.models import Model
import tensorflow as tf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
from tcn import TCN  # pip install keras-tcn
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Attention

# ------------------ CONFIGURATION ------------------ #
MODEL_TYPE = 'sarima'  # Options: 'lstm', 'sarima', 'tcn', 'seq2seq', 'transformer', 'tcn_updated', 'tcn_fixed'
TRIAL_MODE = 'fixed_seed'  # Options: 'fixed_seed', 'multi_seed'
SEEDS = [42] if TRIAL_MODE == 'fixed_seed' else [123, 456, 11, 245, 56712, 23467, 98, 38, 1506, 42]
TRIALS_PER_CONFIG = 30

LOOKBACKS = [3, 5, 7, 9, 11, 12]
BATCH_SIZES = [8, 16, 32]
EPOCHS_LIST = [50, 100]

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
    train = df[df['Month'] < train_end]#['Deaths'].values
    validation = df[(df['Month'] >= train_end) & (df['Month'] < val_end)]#['Deaths'].values
    test = df[(df['Month'] >= val_end)]# & (df['Month'] <= test_end)]#['Deaths'].values
    
    return train, validation, test




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

# def run_sarima(train_df, test_df):
#     np.random.seed(seed)
#     train_series = train_df['Deaths'].astype(float)
#     test_series = test_df['Deaths'].astype(float)
#     model = SARIMAX(train_series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
#                     enforce_stationarity=False, enforce_invertibility=False)
#     results = model.fit(disp=False)
#     forecast = results.predict(start=len(train_series), end=len(train_series) + len(test_series) - 1)
#     return test_series.values, forecast.values

def run_sarima(train_df, test_df):
    np.random.seed(seed)
    train_series = train_df['Deaths'].astype(float)
    test_series = test_df['Deaths'].astype(float)

    model = SARIMAX(train_series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
                    enforce_stationarity=False, enforce_invertibility=False)
    results = model.fit(disp=False)
    
    # In-sample fit (for training)
    fitted = results.fittedvalues

    # Forecast (for test)
    forecast = results.predict(start=len(train_series), end=len(train_series) + len(test_series) - 1)
    
    return train_series.values, fitted.values, test_series.values, forecast.values


# def run_tcn(train, test, look_back, batch_size, epochs, seed):
#     np.random.seed(seed)
#     X_train, y_train = create_dataset(train, look_back)
#     X_test, y_test = create_dataset(np.concatenate([train[-look_back:], test]), look_back)
#     X_train = X_train.reshape((X_train.shape[0], look_back, 1))
#     X_test = X_test.reshape((X_test.shape[0], look_back, 1))
#     model = Sequential([TCN(input_shape=(look_back, 1)), Dense(1)])
#     model.compile(optimizer='adam', loss='mean_squared_error')
#     model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
#     preds = model.predict(X_test).flatten()
#     return y_test, preds

def run_tcn(train, test, look_back, batch_size, epochs, seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)

    X_train, y_train = create_dataset(train, look_back)
    extended_test = np.concatenate([train[-look_back:], test])
    y_test = extended_test[look_back:]

    X_train = X_train.reshape((X_train.shape[0], look_back, 1))

    model = Sequential([TCN(input_shape=(look_back, 1)), Dense(1)])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    # convert to NumPy array before reshaping
    current_input = np.array(train[-look_back:]).reshape((1, look_back, 1))
    preds = []
    for _ in range(len(test)):
        pred = model.predict(current_input, verbose=0)[0][0]
        preds.append(pred)
        current_input = np.append(current_input[:, 1:, :], [[[pred]]], axis=1)

    return y_test, np.array(preds)



def run_tcn_fixed(train, test, look_back, batch_size, epochs, seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Reshape and scale data
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
        Dense(1, activation='relu')  # ensures non-negative output
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    # Generate predictions
    preds = []
    current_input = train_scaled[-look_back:].reshape((1, look_back, 1))
    for _ in range(len(test_scaled)):
        pred = model.predict(current_input, verbose=0)[0][0]
        preds.append(pred)
        current_input = np.append(current_input[:, 1:, :], [[[pred]]], axis=1)

    # Inverse transform predictions
    all_preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()

    # Get original unscaled test ground truth for comparison
    y_test = full_series[len(train):len(train)+len(test)]

    return y_test, all_preds
    

# UPDATED IMPLEMENTATION BELOW REFERENCING: https://pmc.ncbi.nlm.nih.gov/articles/PMC8044508/pdf/10489_2021_Article_2359.pdf


def build_tcn(input_length, num_filters=32, kernel_size=2, num_layers=4, output_size=1):
    inputs = Input(shape=(input_length, 1))
    x = inputs
    skips = []

    for i in range(num_layers):
        dilation_rate = 2 ** i
        conv = Conv1D(filters=num_filters, kernel_size=kernel_size,
                      padding='causal', dilation_rate=dilation_rate)(x)
        conv = ReLU()(conv)
        skip = Conv1D(1, 1, padding='same')(conv)  # skip connection projection
        skips.append(skip)
        x = Add()([conv, x])  # residual connection

    x = Add()(skips)
    x = ReLU()(x)
    x = Dense(output_size)(x)

    return Model(inputs, x)
    
def forecast_tcn(model, initial_seq, n_steps, look_back):
    preds = []
    current_input = initial_seq.copy().reshape((1, look_back, 1))

    for _ in range(n_steps):
        pred = model.predict(current_input, verbose=0)[0, -1]
        preds.append(pred)
        # Slide window
        current_input = np.append(current_input[:, 1:, :], [[[pred]]], axis=1)

    return np.array(preds)

def run_tcn_updated(train, test, look_back, batch_size, epochs, seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)

    X_train, y_train = create_dataset(train, look_back)
    extended_test = np.concatenate([train[-look_back:], test])
    y_test = extended_test[look_back:]

    X_train = X_train.reshape((X_train.shape[0], look_back, 1))

    model = build_tcn(input_length=look_back)
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    # Forecasting
    initial_seq = train[-look_back:]
    preds = forecast_tcn(model, initial_seq, len(test), look_back)

    return y_test, preds



def run_seq2seq(train, test, look_back, batch_size, epochs, seed):
    np.random.seed(seed)
    X_train, y_train = create_dataset(train, look_back)
    X_test, y_test = create_dataset(np.concatenate([train[-look_back:], test]), look_back)
    X_train = X_train.reshape((X_train.shape[0], look_back, 1))
    X_test = X_test.reshape((X_test.shape[0], look_back, 1))

    encoder_inputs = Input(shape=(look_back, 1))
    encoder = GRU(50, return_state=True)
    _, encoder_state = encoder(encoder_inputs)

    decoder_inputs = Input(shape=(look_back, 1))
    decoder_gru = GRU(50, return_sequences=True)
    decoder_outputs = decoder_gru(decoder_inputs, initial_state=encoder_state)
    attention = Attention()([decoder_outputs, encoder_inputs])
    decoder_dense = Dense(1)
    outputs = decoder_dense(attention)

    model = Model([encoder_inputs, decoder_inputs], outputs)
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit([X_train, X_train], y_train.reshape((-1, 1, 1)), epochs=epochs, batch_size=batch_size, verbose=0)

    preds = model.predict([X_test, X_test]).flatten()
    return y_test, preds

# Add placeholder function for transformer if needed

data = load_and_preprocess_data()
train_data, validation_data, test_data = create_train_val_test_split_lstm(data)

# train_data = data[data['Month'] < '2019-01-01']
# validation_data = data[(data['Month'] >= '2019-01-01') & (data['Month'] <= '2019-12-01')]

for seed in SEEDS:
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
                        y_train_true, y_train_pred = run_sarima(train_data, train_data.iloc[look_back:])
                        y_test_true, y_test_pred = run_sarima(train_data, validation_data)
                    elif MODEL_TYPE == 'tcn':
                        y_train_true, y_train_pred = run_tcn(train_data['Deaths'], train_data['Deaths'].iloc[look_back:], look_back, bs, ep, seed)
                        y_test_true, y_test_pred = run_tcn(train_data['Deaths'], validation_data['Deaths'], look_back, bs, ep, seed)
                    elif MODEL_TYPE == 'tcn_updated':
                        y_train_true, y_train_pred = run_tcn_updated(train_data['Deaths'], train_data['Deaths'].iloc[look_back:], look_back, bs, ep, seed)
                        y_test_true, y_test_pred = run_tcn_updated(train_data['Deaths'], validation_data['Deaths'], look_back, bs, ep, seed)
                    elif MODEL_TYPE == 'seq2seq':
                        y_train_true, y_train_pred = run_seq2seq(train_data['Deaths'], train_data['Deaths'].iloc[look_back:], look_back, bs, ep, seed)
                        y_test_true, y_test_pred = run_seq2seq(train_data['Deaths'], validation_data['Deaths'], look_back, bs, ep, seed)
                    else:
                        raise ValueError("Unknown model type")
                    pd.DataFrame({'True': y_train_true, 'Pred': y_train_pred}).to_csv(
                        os.path.join(base_dir, f'trial_{trial}_train.csv'), index=False)
                    pd.DataFrame({'True': y_test_true, 'Pred': y_test_pred}).to_csv(
                        os.path.join(base_dir, f'trial_{trial}_test.csv'), index=False)
                    metrics_train = evaluate_metrics(y_train_true, y_train_pred)
                    metrics_test = evaluate_metrics(y_test_true, y_test_pred)
                    metrics_all_train.append(metrics_train)
                    metrics_all_test.append(metrics_test)
                pd.DataFrame(metrics_all_train).agg(['mean', 'std']).to_csv(
                    os.path.join(base_dir, 'summary_metrics_train.csv'))
                pd.DataFrame(metrics_all_test).agg(['mean', 'std']).to_csv(
                    os.path.join(base_dir, 'summary_metrics_test.csv'))
