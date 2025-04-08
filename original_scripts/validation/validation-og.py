look_back = 3
batch_size = 1
loss = 'mean_squared_error'

look_backs = [3, 5, 7, 9, 11, 12]

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import tensorflow as tf

np.random.seed(42)

df = pd.read_excel('../data/state_month_overdose.xlsx')

df['Deaths'] = df['Deaths'].apply(lambda x: 0 if x == 'Suppressed' else int(x))

df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)

df = df.reset_index() #2015/01
df['Month Code'] = pd.to_datetime(df['Month Code'])#.reset_index() #2015-01-01
# df.set_index('Month', inplace=True)
df = df.groupby(['Month']).agg({'Deaths': 'sum'}).reset_index()



train = df[df['Month'] < '2019-01-01']
test = df[(df['Month'] >= '2019-01-01') & (df['Month'] <= '2019-12-01')]
testog = test
test = test.reset_index().drop(columns = ['index'])



# Modify the create_dataset function to use a lookback of 3 months
def create_dataset(dataset, look_back=3):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset.iloc[i:(i + look_back)]  # Collect the previous 'look_back' months
        dataX.append(a)
        dataY.append(dataset.iloc[i + look_back])  # The target is the subsequent month
    return np.array(dataX), np.array(dataY)



extended_test = pd.concat([train.iloc[-look_back:], test])

# Prepare LSTM datasets
trainX, trainY = create_dataset(train['Deaths'], look_back)
testX, testY = create_dataset(extended_test['Deaths'], look_back)

# Reshape inputs to match LSTM input requirements (samples, time_steps, features)
trainX = trainX.reshape((trainX.shape[0], look_back, 1))
testX = testX.reshape((testX.shape[0], look_back, 1))

# Rebuild the LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(look_back, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=1)

# model = Sequential([
#     LSTM(50, return_sequences=False, input_shape=(look_back, 1)),
#     Dense(1)
# ])
# model.compile(optimizer='adam', loss='mse')
# model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=1)

# Updated generate_forecast function to handle 3-month lookback
def generate_forecast(model, initial_sequence, num_predictions=12, look_back=3):
    predictions = []
    for _ in range(num_predictions):
        # Generate the next prediction
        next_prediction = model.predict(initial_sequence)
        predictions.append(next_prediction[0][0])
        
        # Update the sequence with the new prediction
        new_sequence = np.append(initial_sequence[0, 1:], [[next_prediction[0][0]]], axis=0)
        initial_sequence = new_sequence.reshape((1, look_back, 1))

    return np.array(predictions)

# Prepare the initial sequence for forecasting using the last `look_back` months from training
initial_sequence = trainX[-1].reshape((1, look_back, 1))
initial_sequence

# Generate test predictions with the updated lookback logic
testPredict = generate_forecast(model, initial_sequence, num_predictions=testY.shape[0], look_back=look_back)
trainPredict = model.predict(trainX)

# Flatten predictions for visualization and evaluation
testPredictlst = testPredict.flatten().tolist()
trainPredictlst = trainPredict.flatten().tolist()

val_df = df[df['Month'] <= '2019-12-01']
val_df

combined_array = [0] * look_back + trainPredictlst + testPredictlst


sarima_model = SARIMAX(train['Deaths'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
                enforce_stationarity=False,
                enforce_invertibility=False)
sarima_result = sarima_model.fit(disp=False)
sarima_predictions = sarima_result.predict(start=0, end=len(train) + len(test) - 1, dynamic=False)


val_df['LSTM Predictions'] = combined_array
val_df['SARIMA Predictions'] = sarima_predictions

val_df.to_csv(f'../tables/{look_back}month_predictionresults_batch_{batch_size}_loss_{loss}.csv')