"""
Time Series Pipeline for Substance Overdose Mortality Forecasting

This module implements a pipeline for training, testing, and validating different time series models
for forecasting substance overdose mortality over time, including periods of high uncertainty
like the COVID-19 pandemic.

The pipeline supports:
- LSTM and SARIMA models (extensible to other models)
- Hyperparameter grid search
- Cross-validation over different time periods
- Multiple random seeds for robustness testing
- Comprehensive metrics calculation and model comparison
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from statsmodels.tsa.statespace.sarimax import SARIMAX
import datetime
import json
from abc import ABC, abstractmethod

class TimeSeriesModel(ABC):
    """Abstract base class for time series models."""
    
    def __init__(self, **kwargs):
        """Initialize model with hyperparameters."""
        self.hyperparams = kwargs
        self.model = None
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, X_train, y_train=None):
        """Fit the model to training data."""
        pass
    
    @abstractmethod
    def predict(self, X_test, num_steps=None):
        """Generate predictions using the fitted model."""
        pass
    
    def get_params(self):
        """Return the model's hyperparameters."""
        return self.hyperparams
    
    def __str__(self):
        """Return a string representation of the model."""
        return f"{self.__class__.__name__}({', '.join([f'{k}={v}' for k, v in self.hyperparams.items()])})"


class LSTMModel(TimeSeriesModel):
    """LSTM model for time series forecasting."""
    
    def __init__(self, look_back=3, lstm_units=50, batch_size=1, epochs=50, dropout_rate=0.0, **kwargs):
        """Initialize LSTM model.
        
        Args:
            look_back (int): Number of time steps to look back.
            lstm_units (int): Number of LSTM units in the model.
            batch_size (int): Batch size for training.
            epochs (int): Number of epochs for training.
            dropout_rate (float): Dropout rate for regularization.
        """
        super().__init__(look_back=look_back, lstm_units=lstm_units, batch_size=batch_size, 
                        epochs=epochs, dropout_rate=dropout_rate, **kwargs)
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.look_back = look_back
        self.lstm_units = lstm_units
        self.batch_size = batch_size
        self.epochs = epochs
        self.dropout_rate = dropout_rate
        self.model = None
        
    def _create_dataset(self, dataset, look_back=3):
        """Create dataset with lookback window.
        
        Args:
            dataset (array-like): Time series data.
            look_back (int): Number of previous time steps to use as input features.
            
        Returns:
            tuple: (X, y) where X is the input sequences and y is the target values.
        """
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back):
            a = dataset[i:(i + look_back)]
            dataX.append(a)
            dataY.append(dataset[i + look_back])
        return np.array(dataX), np.array(dataY)
    
    def fit(self, X_train, y_train=None):
        """Fit the LSTM model to the training data.
        
        Args:
            X_train (array-like): Time series data for training.
            y_train (array-like, optional): Not used for this model as targets are created from X_train.
            
        Returns:
            self: The fitted model.
        """
        # Scale the data
        train_scaled = self.scaler.fit_transform(X_train.reshape(-1, 1))
        
        # Create dataset with lookback window
        trainX, trainY = self._create_dataset(train_scaled, self.look_back)
        
        # Reshape input to be [samples, time steps, features]
        trainX = trainX.reshape((trainX.shape[0], self.look_back, 1))
        
        # Build LSTM model
        self.model = Sequential()
        self.model.add(LSTM(self.lstm_units, input_shape=(self.look_back, 1), return_sequences=False, activation='relu'))
        if self.dropout_rate > 0:
            self.model.add(Dropout(self.dropout_rate))
        self.model.add(Dense(1))
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        
        # Train the model
        self.model.fit(trainX, trainY, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        self.is_fitted = True
        return self
    
    def predict(self, X_test, num_steps=None):
        """Generate predictions using the fitted LSTM model.
        
        Args:
            X_test (array-like): Initial time series data for prediction.
            num_steps (int, optional): Number of steps to forecast. If None, predicts only next step.
            
        Returns:
            array: Predicted values.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions.")
        
        # Scale the data using the same scaler fitted on training data
        test_scaled = self.scaler.transform(X_test.reshape(-1, 1))
        
        if num_steps is None or num_steps <= 0:
            # One-step prediction
            testX, _ = self._create_dataset(test_scaled, self.look_back)
            testX = testX.reshape((testX.shape[0], self.look_back, 1))
            predictions = self.model.predict(testX)
            predictions = self.scaler.inverse_transform(predictions)
            return predictions
        else:
            # Multi-step forecast
            predictions = []
            current_sequence = test_scaled[-self.look_back:].reshape((1, self.look_back, 1))
            
            for _ in range(num_steps):
                # Predict next value
                next_pred = self.model.predict(current_sequence)[0, 0]
                predictions.append(next_pred)
                
                # Update sequence for next prediction
                current_sequence = np.append(current_sequence[:, 1:, :], 
                                            [[next_pred]], 
                                            axis=1)
            
            # Inverse transform the predictions
            predictions = np.array(predictions).reshape(-1, 1)
            predictions = self.scaler.inverse_transform(predictions)
            return predictions.flatten()
    
    def generate_forecast(self, initial_sequence, num_predictions):
        """Generate a forecast starting from an initial sequence.
        
        Args:
            initial_sequence (array-like): Initial sequence of shape (1, look_back, 1).
            num_predictions (int): Number of future steps to predict.
            
        Returns:
            array: Predicted values.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions.")
        
        predictions = []
        current_sequence = initial_sequence.copy()
        
        for _ in range(num_predictions):
            # Generate the next prediction
            next_prediction = self.model.predict(current_sequence, verbose=0)
            predictions.append(next_prediction[0][0])
            
            # Update the sequence with the new prediction
            new_sequence = np.append(current_sequence[0, 1:], [[next_prediction[0][0]]], axis=0)
            current_sequence = new_sequence.reshape((1, self.look_back, 1))
        
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions).flatten()
        return predictions


class SARIMAModel(TimeSeriesModel):
    """SARIMA model for time series forecasting."""
    
    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), enforce_stationarity=False, 
                enforce_invertibility=False, **kwargs):
        """Initialize SARIMA model.
        
        Args:
            order (tuple): (p, d, q) order of the SARIMA model.
            seasonal_order (tuple): (P, D, Q, s) seasonal order of the SARIMA model.
            enforce_stationarity (bool): Whether to enforce stationarity.
            enforce_invertibility (bool): Whether to enforce invertibility.
        """
        super().__init__(order=order, seasonal_order=seasonal_order, 
                        enforce_stationarity=enforce_stationarity, 
                        enforce_invertibility=enforce_invertibility, **kwargs)
        self.order = order
        self.seasonal_order = seasonal_order
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        self.model = None
        self.result = None
    
    def fit(self, X_train, y_train=None):
        """Fit the SARIMA model to the training data.
        
        Args:
            X_train (array-like): Time series data for training.
            y_train (array-like, optional): Not used for this model as targets are created from X_train.
            
        Returns:
            self: The fitted model.
        """
        self.model = SARIMAX(
            X_train,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=self.enforce_stationarity,
            enforce_invertibility=self.enforce_invertibility
        )
        self.result = self.model.fit(disp=False)
        self.is_fitted = True
        return self
    
    def predict(self, X_test=None, num_steps=None):
        """Generate predictions using the fitted SARIMA model.
        
        Args:
            X_test (array-like, optional): Not used directly by SARIMA for prediction.
            num_steps (int, optional): Number of steps to forecast. If None, defaults to 12.
            
        Returns:
            array: Predicted values.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions.")
        
        # If num_steps is not provided, default to 12 (monthly data for a year)
        if num_steps is None:
            num_steps = 12
        
        # Generate forecast starting from the end of the training data
        forecast = self.result.get_forecast(steps=num_steps)
        predicted_mean = forecast.predicted_mean
        
        return predicted_mean
    
    def get_confidence_intervals(self, num_steps=12, alpha=0.05):
        """Get confidence intervals for the predictions.
        
        Args:
            num_steps (int): Number of steps to forecast.
            alpha (float): Significance level for the confidence intervals.
            
        Returns:
            tuple: (lower_bound, upper_bound) arrays for the confidence intervals.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before getting confidence intervals.")
        
        forecast = self.result.get_forecast(steps=num_steps)
        conf_int = forecast.conf_int(alpha=alpha)
        
        return conf_int.iloc[:, 0], conf_int.iloc[:, 1]


class Pipeline:
    """Pipeline for time series model training, testing, and validation."""
    
    def __init__(self, data_path, output_dir=None, config=None):
        """Initialize pipeline.
        
        Args:
            data_path (str): Path to the data file.
            output_dir (str, optional): Directory to save outputs. If None, uses current directory.
            config (dict, optional): Configuration parameters for the pipeline.
        """
        self.data_path = data_path
        self.output_dir = output_dir or os.getcwd()
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Default configuration
        default_config = {
            'random_seeds': [42],
            'validation_periods': [('2019-01-01', '2020-01-01')],
            'look_back_periods': [3],
            'final_test_period': ('2020-01-01', None)
        }
        
        self.config = config or default_config
        self.data = None
        self._load_data()
    
    def _load_data(self):
        """Load and preprocess the data."""
        # Load data based on file extension
        file_ext = os.path.splitext(self.data_path)[1].lower()
        
        if file_ext == '.xlsx' or file_ext == '.xls':
            df = pd.read_excel(self.data_path)
        elif file_ext == '.csv':
            df = pd.read_csv(self.data_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        # Handle 'Suppressed' values in Deaths column if they exist
        if 'Deaths' in df.columns and df['Deaths'].dtype == object:
            df['Deaths'] = df['Deaths'].apply(lambda x: 0 if x == 'Suppressed' else int(x))
        
        # Convert date columns if they exist
        if 'Month' in df.columns:
            df['Month'] = pd.to_datetime(df['Month'])
            
        if 'Month Code' in df.columns:
            df['Month Code'] = pd.to_datetime(df['Month Code'])
        
        # Group by month if needed
        if 'Month' in df.columns:
            df = df.groupby(['Month']).agg({'Deaths': 'sum'}).reset_index()
        
        self.data = df
    
    def _split_data(self, train_end_date, test_end_date=None):
        """Split data into training and testing sets.
        
        Args:
            train_end_date (str): End date for training data (exclusive).
            test_end_date (str, optional): End date for testing data (inclusive). If None, uses all available data.
            
        Returns:
            tuple: (train_df, test_df) DataFrames for training and testing.
        """
        train = self.data[self.data['Month'] < train_end_date].copy()
        
        if test_end_date:
            test = self.data[(self.data['Month'] >= train_end_date) & 
                             (self.data['Month'] <= test_end_date)].copy()
        else:
            test = self.data[self.data['Month'] >= train_end_date].copy()
        
        return train, test
    
    def _prepare_lstm_data(self, df, look_back):
        """Prepare data for LSTM model.
        
        Args:
            df (DataFrame): Input DataFrame.
            look_back (int): Number of time steps to look back.
            
        Returns:
            tuple: (X, y) arrays for LSTM input.
        """
        data = df['Deaths'].values
        dataX, dataY = [], []
        
        for i in range(len(data) - look_back):
            dataX.append(data[i:(i + look_back)])
            dataY.append(data[i + look_back])
        
        return np.array(dataX), np.array(dataY)
    
    def _calculate_metrics(self, actual, predicted):
        """Calculate performance metrics.
        
        Args:
            actual (array-like): Actual values.
            predicted (array-like): Predicted values.
            
        Returns:
            dict: Dictionary of metrics.
        """
        metrics = {
            'mae': mean_absolute_error(actual, predicted),
            'rmse': np.sqrt(mean_squared_error(actual, predicted)),
            'mape': mean_absolute_percentage_error(actual, predicted) * 100
        }
        return metrics
    
    def _calculate_prediction_intervals(self, actual, predictions, alpha=0.05):
        """Calculate prediction intervals.
        
        Args:
            actual (array-like): Actual values.
            predictions (array-like): Predicted values.
            alpha (float): Significance level.
            
        Returns:
            tuple: (lower_bound, upper_bound) arrays for the prediction intervals.
        """
        residuals = actual - predictions
        std_residual = np.std(residuals)
        
        # Z-score for the desired confidence level (e.g., 95% PI -> z = 1.96)
        z_score = 1.96  # For 95% confidence
        
        # Calculate margin of error
        margin_of_error = z_score * std_residual
        
        lower_bound = predictions - margin_of_error
        upper_bound = predictions + margin_of_error
        
        return lower_bound, upper_bound
    
    def _calculate_overlap(self, lower1, upper1, lower2, upper2):
        """Calculate the percentage of overlapping prediction intervals.
        
        Args:
            lower1, upper1 (array-like): Lower and upper bounds for first model.
            lower2, upper2 (array-like): Lower and upper bounds for second model.
            
        Returns:
            float: Percentage of overlap.
        """
        overlap_count = sum(
            (u1 >= l2) & (l1 <= u2) 
            for l1, u1, l2, u2 in zip(lower1, upper1, lower2, upper2)
        )
        
        percent_overlap = (overlap_count / len(lower1)) * 100
        return percent_overlap
    
    def run_single_trial(self, model_class, hyperparams, validation_period, random_seed):
        """Run a single trial for a model with specific hyperparameters.
        
        Args:
            model_class (class): Model class to instantiate.
            hyperparams (dict): Hyperparameters for the model.
            validation_period (tuple): (train_end_date, test_end_date) for validation.
            random_seed (int): Random seed for reproducibility.
            
        Returns:
            dict: Results of the trial.
        """
        # Set random seeds
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)
        
        # Split data
        train_end_date, test_end_date = validation_period
        train_df, test_df = self._split_data(train_end_date, test_end_date)
        
        # Instantiate model
        model = model_class(**hyperparams)
        
        # Train model
        if isinstance(model, LSTMModel):
            look_back = hyperparams.get('look_back', 3)
            
            # Prepare initial sequence for LSTM forecast
            extended_test = pd.concat([train_df.iloc[-look_back:], test_df])
            
            # Fit the model on training data
            model.fit(train_df['Deaths'].values)
            
            # Prepare the initial sequence for forecasting
            trainX, _ = model._create_dataset(
                model.scaler.transform(train_df['Deaths'].values.reshape(-1, 1)),
                look_back
            )
            initial_sequence = trainX[-1].reshape((1, look_back, 1))
            
            # Generate predictions
            test_predictions = model.generate_forecast(
                initial_sequence, 
                num_predictions=len(test_df)
            )
            train_predictions = model.predict(train_df['Deaths'].values)
            
            # Flatten predictions
            train_preds = train_predictions.flatten()
            test_preds = test_predictions.flatten()
            
        elif isinstance(model, SARIMAModel):
            # Fit SARIMA model
            model.fit(train_df['Deaths'].values)
            
            # Generate predictions
            train_preds = model.result.predict(
                start=0, 
                end=len(train_df) - 1, 
                dynamic=False
            )
            
            test_preds = model.result.forecast(steps=len(test_df))
        
        else:
            raise ValueError(f"Unsupported model class: {model_class.__name__}")
        
        # Calculate metrics
        train_metrics = self._calculate_metrics(
            train_df['Deaths'].values[len(train_df) - len(train_preds):], 
            train_preds
        )
        test_metrics = self._calculate_metrics(test_df['Deaths'].values, test_preds)
        
        # Calculate prediction intervals
        lower_train, upper_train = self._calculate_prediction_intervals(
            train_df['Deaths'].values[len(train_df) - len(train_preds):], 
            train_preds
        )
        lower_test, upper_test = self._calculate_prediction_intervals(
            test_df['Deaths'].values, 
            test_preds
        )
        
        # Prepare results
        trial_result = {
            'model_class': model_class.__name__,
            'hyperparams': hyperparams,
            'random_seed': random_seed,
            'validation_period': validation_period,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'train_predictions': train_preds.tolist(),
            'test_predictions': test_preds.tolist(),
            'train_actual': train_df['Deaths'].values[len(train_df) - len(train_preds):].tolist(),
            'test_actual': test_df['Deaths'].values.tolist(),
            'train_pi': {
                'lower': lower_train.tolist(),
                'upper': upper_train.tolist()
            },
            'test_pi': {
                'lower': lower_test.tolist(),
                'upper': upper_test.tolist()
            }
        }
        
        return trial_result
    
    def run_experiment(self, model_class, hyperparameter_grid):
        """Run an experiment with grid search over hyperparameters.
        
        Args:
            model_class (class): Model class to instantiate.
            hyperparameter_grid (dict): Grid of hyperparameters to search.
            
        Returns:
            dict: Results of the experiment.
        """
        # Generate all hyperparameter combinations
        keys = hyperparameter_grid.keys()
        values = hyperparameter_grid.values()
        hyperparameter_combinations = [
            dict(zip(keys, combination)) 
            for combination in itertools.product(*values)
        ]
        
        results = {
            'model_class': model_class.__name__,
            'hyperparameter_grid': hyperparameter_grid,
            'validation_periods': self.config['validation_periods'],
            'random_seeds': self.config['random_seeds'],
            'trials': []
        }
        
        # Run trials for each combination of hyperparameters, validation period, and random seed
        total_trials = (
            len(hyperparameter_combinations) * 
            len(self.config['validation_periods']) * 
            len(self.config['random_seeds'])
        )
        
        print(f"Running {total_trials} trials for {model_class.__name__}...")
        trial_count = 0
        
        for hyperparams in hyperparameter_combinations:
            for validation_period in self.config['validation_periods']:
                for seed in self.config['random_seeds']:
                    trial_count += 1
                    print(f"  Trial {trial_count}/{total_trials}: {hyperparams}, {validation_period}, seed={seed}")
                    
                    # Run single trial
                    trial_result = self.run_single_trial(
                        model_class, 
                        hyperparams, 
                        validation_period, 
                        seed
                    )
                    
                    results['trials'].append(trial_result)
                    
                    # Save interim results
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    interim_filename = os.path.join(
                        self.output_dir, 
                        f"interim_{model_class.__name__}_{timestamp}.json"
                    )
                    with open(interim_filename, 'w') as f:
                        json.dump(results, f, indent=2)
        
        # Find best hyperparameters based on average test RMSE
        best_hyperparam_performance = {}
        
        for hyperparams in hyperparameter_combinations:
            hyperparams_str = str(hyperparams)
            rmse_values = []
            
            for trial in results['trials']:
                if trial['hyperparams'] == hyperparams:
                    rmse_values.append(trial['test_metrics']['rmse'])
            
            avg_rmse = sum(rmse_values) / len(rmse_values)
            best_hyperparam_performance[hyperparams_str] = avg_rmse
        
        best_hyperparams_str = min(best_hyperparam_performance, key=best_hyperparam_performance.get)
        best_hyperparams = next(
            (h for h in hyperparameter_combinations if str(h) == best_hyperparams_str),
            None
        )
        
        results['best_hyperparams'] = best_hyperparams
        results['best_avg_rmse'] = best_hyperparam_performance[best_hyperparams_str]
        
        # Save final results
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        final_filename = os.path.join(
            self.output_dir, 
            f"final_{model_class.__name__}_{timestamp}.json"
        )
        with open(final_filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def run_final_model(self, model_class, hyperparams, random_seed=42):
        """Run the final model on the full training data and generate predictions for test period.
        
        Args:
            model_class (class): Model class to instantiate.
            hyperparams (dict): Hyperparameters for the model.
            random_seed (int): Random seed for reproducibility.
            
        Returns:
            dict: Results of the final model.
        """
        # Set random seeds
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)
        
        # Split data for final test
        train_end_date, test_end_date = self.config['final_test_period']
        train_df, test_df = self._split_data(train_end_date, test_end_date)
        
        # Instantiate model
        model = model_class(**hyperparams)
        
        # Train and predict
        if isinstance(model, LSTMModel):
            look_back = hyperparams.get('look_back', 3)
            
            # Prepare initial sequence for LSTM forecast
            extended_test = pd.concat([train_df.iloc[-look_back:], test_df])
            
            # Fit the model on training data
            model.fit(train_df['Deaths'].values)
            
            # Prepare the initial sequence for forecasting
            trainX, _ = model._create_dataset(
                model.scaler.transform(train_df['Deaths'].values.reshape(-1, 1)),
                look_back
            )
            initial_sequence = trainX[-1].reshape((1, look_back, 1))
            
            # Generate predictions
            test_predictions = model.generate_forecast(
                initial_sequence, 
                num_predictions=len(test_df)
            )
            train_predictions = model.predict(train_df['Deaths'].values)
            
            # Flatten predictions
            train_preds = train_predictions.flatten()
            test_preds = test_predictions.flatten()
            
        elif isinstance(model, SARIMAModel):
            # Fit SARIMA model
            model.fit(train_df['Deaths'].values)
            
            # Generate predictions
            train_preds = model.result.predict(
                start=0, 
                end=len(train_df) - 1, 
                dynamic=False
            )
            
            test_preds = model.result.forecast(steps=len(test_df))
        
        else:
            raise ValueError(f"Unsupported model class: {model_class.__name__}")
        
        # Calculate metrics
        train_metrics = self._calculate_metrics(
            train_df['Deaths'].values[len(train_df) - len(train_preds):], 
            train_preds
        )
        test_metrics = self._calculate_metrics(test_df['Deaths'].values, test_preds)
        
        # Calculate prediction intervals
        lower_train, upper_train = self._calculate_prediction_intervals(
            train_df['Deaths'].values[len(train_df) - len(train_preds):], 
            train_preds
        )
        lower_test, upper_test = self._calculate_prediction_intervals(
            test_df['Deaths'].values, 
            test_preds
        )
        
        # Prepare results
        final_results = {
            'model_class': model_class.__name__,
            'hyperparams': hyperparams,
            'random_seed': random_seed,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'train_predictions': train_preds.tolist(),
            'test_predictions': test_preds.tolist(),
            'train_actual': train_df['Deaths'].values[len(train_df) - len(train_preds):].tolist(),
            'test_actual': test_df['Deaths'].values.tolist(),
            'train_pi': {
                'lower': lower_train.tolist(),
                'upper': upper_train.tolist()
            },
            'test_pi': {
                'lower': lower_test.tolist(),
                'upper': upper_test.tolist()
            },
            'dates': {
                'train': train_df['Month'].astype(str).tolist(),
                'test': test_df['Month'].astype(str).tolist()
            }
        }
        
        # Save final model results
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        final_filename = os.path.join(
            self.output_dir, 
            f"final_model_{model_class.__name__}_{timestamp}.json"
        )
        with open(final_filename, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        # Create and save visualization
        self.visualize_predictions(
            train_df['Month'].values,
            test_df['Month'].values,
            train_df['Deaths'].values[len(train_df) - len(train_preds):],
            test_df['Deaths'].values,
            train_preds,
            test_preds,
            lower_train,
            upper_train,
            lower_test,
            upper_test,
            f"Final {model_class.__name__} Model",
            os.path.join(self.output_dir, f"final_model_{model_class.__name__}_{timestamp}.png")
        )
        
        return final_results
    
    def visualize_predictions(self, train_dates, test_dates, train_actual, test_actual, 
                            train_pred, test_pred, train_lower, train_upper,
                            test_lower, test_upper, title, save_path):
        """Visualize model predictions with prediction intervals.
        
        Args:
            train_dates (array-like): Dates for training data.
            test_dates (array-like): Dates for testing data.
            train_actual (array-like): Actual values for training data.
            test_actual (array-like): Actual values for testing data.
            train_pred (array-like): Predicted values for training data.
            test_pred (array-like): Predicted values for testing data.
            train_lower (array-like): Lower bound for training prediction intervals.
            train_upper (array-like): Upper bound for training prediction intervals.
            test_lower (array-like): Lower bound for testing prediction intervals.
            test_upper (array-like): Upper bound for testing prediction intervals.
            title (str): Title for the plot.
            save_path (str): Path to save the plot.
        """
        plt.figure(figsize=(12, 8))
        
        # Plot training data and predictions
        plt.plot(train_dates, train_actual, 'b.-', label='Training Data')
        plt.plot(train_dates, train_pred, 'r-', label='Training Predictions')
        plt.fill_between(train_dates, train_lower, train_upper, color='r', alpha=0.2, label='Training 95% PI')
        
        # Plot test data and predictions
        plt.plot(test_dates, test_actual, 'g.-', label='Test Data')
        plt.plot(test_dates, test_pred, 'm-', label='Test Predictions')
        plt.fill_between(test_dates, test_lower, test_upper, color='m', alpha=0.2, label='Test 95% PI')
        
        # Add vertical line to separate training and testing
        plt.axvline(x=test_dates[0], color='k', linestyle='--', label='Train/Test Split')
        
        # Customize plot
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Deaths')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        # Format x-axis to show dates properly
        plt.gcf().autofmt_xdate()
        
        # Save plot
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
    def compare_models(self, experiment_results):
        """Compare the results of multiple model experiments.
        
        Args:
            experiment_results (list): List of experiment result dictionaries.
            
        Returns:
            dict: Comparison results.
        """
        if not experiment_results:
            raise ValueError("No experiment results provided for comparison")
        
        comparison = {
            'models': [],
            'best_hyperparams': {},
            'test_metrics': {},
            'pi_overlap': {}
        }
        
        # Extract best model configurations from each experiment
        for exp_result in experiment_results:
            model_name = exp_result['model_class']
            comparison['models'].append(model_name)
            comparison['best_hyperparams'][model_name] = exp_result['best_hyperparams']
            
            # Calculate average metrics across trials with best hyperparameters
            best_hyperparams = exp_result['best_hyperparams']
            best_trials = [
                trial for trial in exp_result['trials'] 
                if trial['hyperparams'] == best_hyperparams
            ]
            
            # Calculate average metrics
            mae_values = [trial['test_metrics']['mae'] for trial in best_trials]
            rmse_values = [trial['test_metrics']['rmse'] for trial in best_trials]
            mape_values = [trial['test_metrics']['mape'] for trial in best_trials]
            
            comparison['test_metrics'][model_name] = {
                'mae': {
                    'mean': np.mean(mae_values),
                    'std': np.std(mae_values)
                },
                'rmse': {
                    'mean': np.mean(rmse_values),
                    'std': np.std(rmse_values)
                },
                'mape': {
                    'mean': np.mean(mape_values),
                    'std': np.std(mape_values)
                }
            }
        
        # If we have exactly two models, calculate prediction interval overlap
        if len(comparison['models']) == 2:
            model1, model2 = comparison['models']
            
            # Run final models with best hyperparameters
            model1_class = globals()[model1]
            model2_class = globals()[model2]
            
            model1_results = self.run_final_model(
                model1_class, 
                comparison['best_hyperparams'][model1]
            )
            
            model2_results = self.run_final_model(
                model2_class, 
                comparison['best_hyperparams'][model2]
            )
            
            # Calculate PI overlap on test data
            overlap_percentage = self._calculate_overlap(
                model1_results['test_pi']['lower'],
                model1_results['test_pi']['upper'],
                model2_results['test_pi']['lower'],
                model2_results['test_pi']['upper']
            )
            
            comparison['pi_overlap'][f"{model1}_vs_{model2}"] = overlap_percentage
            
            # Visualize comparison
            self._visualize_model_comparison(model1_results, model2_results)
        
        # Save comparison results
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_filename = os.path.join(
            self.output_dir, 
            f"model_comparison_{timestamp}.json"
        )
        with open(comparison_filename, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        return comparison
    
    def _visualize_model_comparison(self, model1_results, model2_results):
        """Visualize comparison between two models.
        
        Args:
            model1_results (dict): Results from first model.
            model2_results (dict): Results from second model.
        """
        model1_name = model1_results['model_class']
        model2_name = model2_results['model_class']
        
        # Convert date strings back to datetime
        train_dates = pd.to_datetime(model1_results['dates']['train'])
        test_dates = pd.to_datetime(model1_results['dates']['test'])
        
        # Extract data
        train_actual = np.array(model1_results['train_actual'])
        test_actual = np.array(model1_results['test_actual'])
        
        model1_train_pred = np.array(model1_results['train_predictions'])
        model1_test_pred = np.array(model1_results['test_predictions'])
        model1_train_lower = np.array(model1_results['train_pi']['lower'])
        model1_train_upper = np.array(model1_results['train_pi']['upper'])
        model1_test_lower = np.array(model1_results['test_pi']['lower'])
        model1_test_upper = np.array(model1_results['test_pi']['upper'])
        
        model2_train_pred = np.array(model2_results['train_predictions'])
        model2_test_pred = np.array(model2_results['test_predictions'])
        model2_train_lower = np.array(model2_results['train_pi']['lower'])
        model2_train_upper = np.array(model2_results['train_pi']['upper'])
        model2_test_lower = np.array(model2_results['test_pi']['lower'])
        model2_test_upper = np.array(model2_results['test_pi']['upper'])
        
        # Create comparison plot
        plt.figure(figsize=(14, 10))
        
        # Plot actual data
        plt.plot(train_dates, train_actual, 'k.-', label='Training Data', linewidth=2)
        plt.plot(test_dates, test_actual, 'k.-', label='Test Data', linewidth=2)
        
        # Plot model 1 predictions
        plt.plot(train_dates, model1_train_pred, 'b-', label=f'{model1_name} Train', alpha=0.7)
        plt.plot(test_dates, model1_test_pred, 'b-', label=f'{model1_name} Test', linewidth=2)
        plt.fill_between(test_dates, model1_test_lower, model1_test_upper, color='b', alpha=0.2, 
                        label=f'{model1_name} 95% PI')
        
        # Plot model 2 predictions
        plt.plot(train_dates, model2_train_pred, 'r-', label=f'{model2_name} Train', alpha=0.7)
        plt.plot(test_dates, model2_test_pred, 'r-', label=f'{model2_name} Test', linewidth=2)
        plt.fill_between(test_dates, model2_test_lower, model2_test_upper, color='r', alpha=0.2, 
                        label=f'{model2_name} 95% PI')
        
        # Add vertical line to separate training and testing
        plt.axvline(x=test_dates[0], color='k', linestyle='--', label='Train/Test Split')
        
        # Calculate metrics for display
        model1_rmse = np.sqrt(mean_squared_error(test_actual, model1_test_pred))
        model2_rmse = np.sqrt(mean_squared_error(test_actual, model2_test_pred))
        model1_mape = mean_absolute_percentage_error(test_actual, model1_test_pred) * 100
        model2_mape = mean_absolute_percentage_error(test_actual, model2_test_pred) * 100
        
        # Find PI overlap
        overlap_percentage = self._calculate_overlap(
            model1_test_lower, model1_test_upper,
            model2_test_lower, model2_test_upper
        )
        
        # Customize plot
        plt.title(f'Model Comparison: {model1_name} vs {model2_name}', fontsize=16)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Deaths', fontsize=14)
        
        # Add metrics to legend
        legend_text = [
            'Training Data', 
            'Test Data',
            f'{model1_name} Train',
            f'{model1_name} Test (RMSE={model1_rmse:.2f}, MAPE={model1_mape:.2f}%)',
            f'{model1_name} 95% PI',
            f'{model2_name} Train',
            f'{model2_name} Test (RMSE={model2_rmse:.2f}, MAPE={model2_mape:.2f}%)',
            f'{model2_name} 95% PI',
            'Train/Test Split'
        ]
        
        plt.legend(legend_text, loc='best', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.figtext(0.5, 0.01, f'Prediction Interval Overlap: {overlap_percentage:.2f}%', 
                   ha='center', fontsize=12)
        
        # Format x-axis
        plt.gcf().autofmt_xdate()
        
        # Save plot
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_plot_path = os.path.join(
            self.output_dir, 
            f"model_comparison_{model1_name}_vs_{model2_name}_{timestamp}.png"
        )
        plt.tight_layout()
        plt.savefig(comparison_plot_path)
        plt.close()


# For potential future extension with Temporal Fusion Transformer
class TFTModel(TimeSeriesModel):
    """Temporal Fusion Transformer model for time series forecasting.
    
    Note: This is a placeholder for future implementation.
    """
    
    def __init__(self, look_back=3, hidden_size=64, num_attention_heads=4, dropout_rate=0.1, 
                batch_size=16, epochs=100, **kwargs):
        """Initialize TFT model.
        
        Args:
            look_back (int): Number of time steps to look back.
            hidden_size (int): Hidden layer size.
            num_attention_heads (int): Number of attention heads.
            dropout_rate (float): Dropout rate.
            batch_size (int): Batch size for training.
            epochs (int): Number of epochs for training.
        """
        super().__init__(look_back=look_back, hidden_size=hidden_size, 
                        num_attention_heads=num_attention_heads, 
                        dropout_rate=dropout_rate, batch_size=batch_size, 
                        epochs=epochs, **kwargs)
        self.look_back = look_back
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None
        
    def fit(self, X_train, y_train=None):
        """Fit the TFT model to the training data.
        
        This is a placeholder for future implementation.
        
        Args:
            X_train (array-like): Time series data for training.
            y_train (array-like, optional): Not used for this model.
            
        Returns:
            self: The fitted model.
        """
        raise NotImplementedError("TFT model implementation is not yet available")
    
    def predict(self, X_test, num_steps=None):
        """Generate predictions using the fitted TFT model.
        
        This is a placeholder for future implementation.
        
        Args:
            X_test (array-like): Initial time series data for prediction.
            num_steps (int, optional): Number of steps to forecast.
            
        Returns:
            array: Predicted values.
        """
        raise NotImplementedError("TFT model implementation is not yet available")