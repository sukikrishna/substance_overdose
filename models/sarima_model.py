# ### models/sarima_model.py

# import numpy as np
# from statsmodels.tsa.statespace.sarimax import SARIMAX
# from models.base_model import BaseModel

# class SARIMAModel(BaseModel):
#     def __init__(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
#         self.order = order
#         self.seasonal_order = seasonal_order
#         self.model = None

#     def fit(self, X, y=None):
#         self.model = SARIMAX(X, order=self.order, seasonal_order=self.seasonal_order,
#                              enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)

#     def predict(self, steps):
#         return self.model.forecast(steps=steps)

"""SARIMA model implementation for time series forecasting."""

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from models.base_model import BaseModel

class SARIMAModel(BaseModel):
    """
    Seasonal AutoRegressive Integrated Moving Average (SARIMA) model.
    
    Parameters
    ----------
    order : tuple, default=(1, 1, 1)
        ARIMA order (p, d, q)
    seasonal_order : tuple, default=(1, 1, 1, 12)
        Seasonal ARIMA order (P, D, Q, s)
    """
    
    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.result = None
        
    def fit(self, X, y=None):
        """
        Fit SARIMA model to time series data.
        
        Parameters
        ----------
        X : array-like or pandas Series
            Time series data
        y : ignored
            Not used, present for API consistency
            
        Returns
        -------
        self : object
            Fitted model instance
        """
        # Fit SARIMA model
        self.model = SARIMAX(
            X,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        self.result = self.model.fit(disp=False)
        return self
        
    def predict(self, steps):
        """
        Generate forecasts for future time steps.
        
        Parameters
        ----------
        steps : int
            Number of steps to forecast
            
        Returns
        -------
        array-like
            Predicted values
        """
        # Generate forecasts
        try:
            forecast = self.result.forecast(steps=steps)
            return forecast
        except (KeyError, ValueError) as e:
            # Alternative approach if the standard forecast fails
            # Forecast from the end of the training data
            forecast = self.result.get_forecast(steps=steps).predicted_mean
            return forecast.values