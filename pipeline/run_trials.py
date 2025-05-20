"""Module for running multiple trials of time series forecasting models."""

import os
import numpy as np
import pandas as pd
from utils import create_dataset, generate_forecast, calculate_prediction_intervals

def load_data(data_path):
    """
    Load and preprocess overdose data.
    
    Parameters
    ----------
    data_path : str
        Path to Excel file containing overdose data
        
    Returns
    -------
    pd.DataFrame
        Preprocessed dataframe with 'Month' and 'Deaths' columns
    """
    df = pd.read_excel(data_path)
    
    # Convert 'Suppressed' to 0
    df['Deaths'] = df['Deaths'].apply(lambda x: 0 if x == 'Suppressed' else int(x))
    
    # Convert month to datetime and ensure it's properly formatted
    df['Month'] = pd.to_datetime(df['Month'])
    
    # Aggregate by month if there are multiple entries per month
    df = df.groupby('Month').agg({'Deaths': 'sum'}).reset_index()
    
    return df

def run_trials(model_name, model_class, data_path, output_dir, lookbacks, 
               batch_sizes, epochs, seeds, train_end='2020-01-01'):
    """
    Run multiple trials for a given model across different hyperparameters.
    
    Parameters
    ----------
    model_name : str
        Name of the model (used for directory creation)
    model_class : class
        Model class to instantiate
    data_path : str
        Path to data file
    output_dir : str
        Directory to save results
    lookbacks : list
        List of lookback values to test
    batch_sizes : list
        List of batch sizes to test
    epochs : int
        Number of training epochs
    seeds : list
        List of random seeds for reproducibility
    train_end : str, default='2020-01-01'
        End date for training data (format: YYYY-MM-DD)
    """
    # Load data
    df = load_data(data_path)
    
    # Split data into train and test
    train = df[df['Month'] < train_end]
    test = df[df['Month'] >= train_end].reset_index(drop=True)
    
    # Create main directory for model
    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    for lookback in lookbacks:
        for batch in batch_sizes:
            for seed in seeds:
                # Set random seed for reproducibility
                np.random.seed(seed)
                if 'tensorflow' in str(type(model_class)):
                    import tensorflow as tf
                    tf.random.set_seed(seed)
                
                # Create directory for this parameter combination
                combo_id = f"lookback{lookback}_batch{batch}_seed{seed}"
                combo_dir = os.path.join(model_dir, combo_id)
                os.makedirs(combo_dir, exist_ok=True)
                
                for trial in range(10):  # Run 10 trials for each parameter combination
                    # Create directory for this trial
                    trial_dir = os.path.join(combo_dir, f"trial_{trial}")
                    os.makedirs(trial_dir, exist_ok=True)
                    
                    # SARIMA model has a different workflow
                    if model_name.lower() == 'sarima':
                        # Fit SARIMA model on training data
                        model = model_class()
                        model.fit(train['Deaths'])
                        
                        # Generate predictions
                        test_preds = model.predict(len(test))
                        
                        # Calculate prediction intervals using training residuals
                        train_preds = model.result.fittedvalues
                        # Handle potential length mismatch with training data
                        if len(train_preds) < len(train):
                            # Fill missing values at the beginning with NaN
                            diff = len(train) - len(train_preds)
                            train_preds = np.concatenate([np.full(diff, np.nan), train_preds])
                        
                        # Calculate residuals on non-NaN values
                        valid_indices = ~np.isnan(train_preds)
                        residuals = train['Deaths'].values[valid_indices] - train_preds[valid_indices]
                        std_residual = np.std(residuals)
                        
                        # 95% prediction intervals (z=1.96)
                        z_score = 1.96
                        margin_of_error = z_score * std_residual
                        lower_pi = test_preds - margin_of_error
                        upper_pi = test_preds + margin_of_error
                        
                        # Save predictions
                        out_df = test.copy()
                        out_df['Predictions'] = test_preds
                        out_df['Lower PI'] = lower_pi
                        out_df['Upper PI'] = upper_pi
                        out_df.to_csv(os.path.join(trial_dir, "predictions.csv"), index=False)
                    else:
                        # For LSTM and other models
                        trainX, trainY = create_dataset(train['Deaths'], lookback)
                        
                        # Prepare test data by including necessary lookback from training
                        extended_test = pd.concat([train.iloc[-lookback:], test])
                        testX, testY = create_dataset(extended_test['Deaths'], lookback)
                        
                        # Reshape inputs for LSTM
                        trainX = trainX.reshape((trainX.shape[0], lookback, 1))
                        testX = testX.reshape((testX.shape[0], lookback, 1))
                        
                        # Create and fit model
                        model = model_class(look_back=lookback, batch_size=batch, epochs=epochs)
                        model.fit(trainX, trainY)
                        
                        # Generate predictions for test data
                        initial_sequence = trainX[-1].reshape((1, lookback, 1))
                        test_preds = generate_forecast(model, initial_sequence, len(testY), lookback)
                        
                        # Calculate prediction intervals
                        train_preds = model.predict(trainX)
                        residuals = trainY - train_preds
                        std_residual = np.std(residuals)
                        
                        # 95% prediction intervals (z=1.96)
                        z_score = 1.96
                        margin_of_error = z_score * std_residual
                        lower_pi = test_preds - margin_of_error
                        upper_pi = test_preds + margin_of_error
                        
                        # Save predictions
                        out_df = test.copy()
                        out_df['Predictions'] = test_preds
                        out_df['Lower PI'] = lower_pi
                        out_df['Upper PI'] = upper_pi
                        out_df.to_csv(os.path.join(trial_dir, "predictions.csv"), index=False)
                
                print(f"Completed {model_name} - lookback:{lookback}, batch:{batch}, seed:{seed}")