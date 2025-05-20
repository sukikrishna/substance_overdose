# ### models/lstm_model.py

# import numpy as np
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
# from sklearn.preprocessing import MinMaxScaler
# from models.base_model import BaseModel

# class LSTMModel(BaseModel):
#     def __init__(self, look_back=3, batch_size=1, epochs=50):
#         self.look_back = look_back
#         self.batch_size = batch_size
#         self.epochs = epochs
#         self.model = None
#         self.scaler = MinMaxScaler(feature_range=(0, 1))

#     def fit(self, X, y):
#         X = X.reshape((X.shape[0], X.shape[1], 1))
#         self.model = Sequential()
#         self.model.add(LSTM(50, activation='relu', input_shape=(self.look_back, 1)))
#         self.model.add(Dense(1))
#         self.model.compile(optimizer='adam', loss='mean_squared_error')
#         self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)

#     def predict(self, X):
#         return self.model.predict(X, verbose=0).flatten()

"""LSTM model implementation for time series forecasting."""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from models.base_model import BaseModel

class LSTMModel(BaseModel):
    """
    Long Short-Term Memory (LSTM) model for time series forecasting.
    
    Parameters
    ----------
    look_back : int, default=3
        Number of previous time steps to use as input features
    batch_size : int, default=1
        Batch size for training
    epochs : int, default=50
        Number of epochs to train for
    verbose : int, default=0
        Verbosity mode (0=silent, 1=progress bar)
    """
    
    def __init__(self, look_back=3, batch_size=1, epochs=50, verbose=0):
        self.look_back = look_back
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        self.model = None
        
    def fit(self, X, y):
        """
        Fit LSTM model to training data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_timesteps)
            Training time series data, should already be shaped for LSTM input
        y : array-like of shape (n_samples,)
            Target values
            
        Returns
        -------
        self : object
            Fitted model instance
        """
        # Reshape X if it's not already in the right format
        if len(X.shape) == 2:
            X = X.reshape((X.shape[0], X.shape[1], 1))
            
        # Build model
        self.model = Sequential()
        self.model.add(LSTM(50, activation='relu', input_shape=(self.look_back, 1)))
        self.model.add(Dense(1))
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        
        # Fit model
        self.model.fit(
            X, y, 
            epochs=self.epochs, 
            batch_size=self.batch_size, 
            verbose=self.verbose
        )
        
        return self
    
    def predict(self, X):
        """
        Generate predictions for new data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_timesteps)
            Data for which to generate predictions
            
        Returns
        -------
        array-like of shape (n_samples,)
            Predictions
        """
        # Ensure X is in the right format
        if len(X.shape) == 2:
            X = X.reshape((X.shape[0], X.shape[1], 1))
            
        return self.model.predict(X, verbose=0).flatten()