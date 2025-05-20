"""Utility functions for time series forecasting pipeline."""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

def create_dataset(series, look_back):
    """
    Create dataset with look-back time steps as features.
    
    Parameters
    ----------
    series : pandas Series
        Time series data
    look_back : int
        Number of time steps to look back
        
    Returns
    -------
    X : numpy.ndarray
        Input features with shape (n_samples, look_back)
    y : numpy.ndarray
        Target values with shape (n_samples,)
    """
    X, y = [], []
    for i in range(len(series) - look_back):
        X.append(series.iloc[i:(i + look_back)].values)
        y.append(series.iloc[i + look_back])
    return np.array(X), np.array(y)

def generate_forecast(model, initial_sequence, num_predictions, look_back):
    """
    Generate multi-step forecasts using a recursive strategy.
    
    Parameters
    ----------
    model : object
        Fitted model with predict method
    initial_sequence : numpy.ndarray
        Initial sequence of shape (1, look_back, 1) for LSTM input
    num_predictions : int
        Number of future steps to predict
    look_back : int
        Number of time steps used as input features
        
    Returns
    -------
    numpy.ndarray
        Forecasted values
    """
    predictions = []
    curr_seq = initial_sequence.copy()
    
    for _ in range(num_predictions):
        # Generate next prediction
        next_pred = model.predict(curr_seq)[0]
        predictions.append(next_pred)
        
        # Update sequence with the new prediction
        curr_seq = np.append(curr_seq[0, 1:, 0], next_pred).reshape(1, look_back, 1)
        
    return np.array(predictions)

def calculate_prediction_intervals(actual, predictions, alpha=0.05):
    """
    Calculate prediction intervals based on the residuals.
    
    Parameters
    ----------
    actual : array-like
        Actual values
    predictions : array-like
        Predicted values
    alpha : float, default=0.05
        Significance level (1-alpha = confidence level)
        
    Returns
    -------
    lower : array-like
        Lower bounds of prediction intervals
    upper : array-like
        Upper bounds of prediction intervals
    """
    residuals = actual - predictions
    std_residual = np.std(residuals)
    
    # For 95% prediction intervals with z-score of 1.96
    z_score = 1.96
    margin_of_error = z_score * std_residual
    
    lower_bound = predictions - margin_of_error
    upper_bound = predictions + margin_of_error
    
    return lower_bound, upper_bound

def calculate_metrics(actual, predictions):
    """
    Calculate performance metrics for model evaluation.
    
    Parameters
    ----------
    actual : array-like
        Actual values
    predictions : array-like
        Predicted values
        
    Returns
    -------
    dict
        Dictionary containing metrics (RMSE, MAE, MAPE)
    """
    rmse = np.sqrt(mean_squared_error(actual, predictions))
    mae = mean_absolute_error(actual, predictions)
    mape = mean_absolute_percentage_error(actual, predictions) * 100
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }

def compute_overlap_with_ground_truth(actual, lower, upper):
    """
    Calculate the percentage of actual values that fall within prediction intervals.
    
    Parameters
    ----------
    actual : array-like
        Actual values
    lower : array-like
        Lower bounds of prediction intervals
    upper : array-like
        Upper bounds of prediction intervals
        
    Returns
    -------
    float
        Percentage of actual values within prediction intervals
    """
    actual = np.array(actual)
    inside = (actual >= lower) & (actual <= upper)
    return 100.0 * np.sum(inside) / len(actual)