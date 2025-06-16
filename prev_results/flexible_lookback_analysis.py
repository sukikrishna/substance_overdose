"""
Flexible Lookback Analysis for Substance Overdose Time Series

This script evaluates the effect of different lookback periods on LSTM and SARIMA models
for forecasting substance overdose mortality. It implements the approach demonstrated in
the FlexibleLookback.ipynb notebook, but in a more structured and reusable way.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import tensorflow as tf
import os

class FlexibleLookbackAnalysis:
    """Analyzes the effect of different lookback periods on forecasting models."""
    
    def __init__(self, data_path, output_dir='results/flexible_lookback'):
        """Initialize the analysis.
        
        Args:
            data_path (str): Path to the data file.
            output_dir (str): Directory to save results.
        """
        self.data_path = data_path
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Load and preprocess data
        self.df = None
        self._load_data()
        
        # Default parameters
        self.validation_periods = [
            ('2019-11-01', '2020-01-01'),
            ('2019-09-01', '2020-01-01'),
            ('2019-07-01', '2020-01-01'),
            ('2019-01-01', '2020-01-01'),
            ('2018-07-01', '2020-01-01'),
            ('2018-01-01', '2020-01-01')
        ]
        self.look_back_periods = range(3, 12, 2)  # 3, 5, 7, 9, 11
        
    def _load_data(self):
        """Load and preprocess the data."""
        df = pd.read_excel(self.data_path)
        df['Deaths'] = df['Deaths'].apply(lambda x: 0 if x == 'Suppressed' else int(x))
        df['Month'] = pd.to_datetime(df['Month'])
        df = df.groupby(['Month']).agg({'Deaths': 'sum'}).reset_index()
        self.df = df
        
    def create_dataset(self, dataset, look_back=1):
        """Create dataset with lookback window.
        
        Args:
            dataset (DataFrame): Time series data.
            look_back (int): Number of previous time steps to use as input features.
            
        Returns:
            tuple: (X, y) where X is the input sequences and y is the target values.
        """
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back):
            a = dataset.iloc[i:(i + look_back)].values
            dataX.append(a)
            dataY.append(dataset.iloc[i + look_back])
        return np.array(dataX), np.array(dataY)
    
    def calculate_confidence_intervals(self, predictions, alpha=0.05):
        """Calculate confidence intervals for predictions.
        
        Args:
            predictions (array-like): Predicted values.
            alpha (float): Significance level.
            
        Returns:
            tuple: (lower_bound, upper_bound) arrays for the confidence intervals.
        """
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        
        # Z-score for the desired confidence level (95% CI -> z = 1.96)
        z_score = 1.96
        margin_of_error = z_score * (std_pred / np.sqrt(len(predictions)))
        
        lower_bound = predictions - margin_of_error
        upper_bound = predictions + margin_of_error
        
        return lower_bound, upper_bound
    
    def calculate_overlap(self, lower1, upper1, lower2, upper2):
        """Calculate the percentage of overlapping confidence intervals.
        
        Args:
            lower1, upper1 (array-like): Lower and upper bounds for first model.
            lower2, upper2 (array-like): Lower and upper bounds for second model.
            
        Returns:
            float: Percentage of overlap.
        """
        overlap_count = 0
        
        for l1, u1, l2, u2 in zip(lower1, upper1, lower2, upper2):
            if u1 >= l2 and l1 <= u2:
                overlap_count += 1
                
        percent_overlap = (overlap_count / len(lower1)) * 100
        return percent_overlap
    
    def generate_forecast(self, model, initial_sequence, look_back, num_predictions=12):
        """Generate a multi-step forecast for an LSTM model.
        
        Args:
            model (Keras model): Trained LSTM model.
            initial_sequence (array): Initial sequence for prediction.
            look_back (int): Lookback period.
            num_predictions (int): Number of steps to predict.
            
        Returns:
            array: Predicted values.
        """
        predictions = []
        current_sequence = initial_sequence.copy()
        
        for _ in range(num_predictions):
            # Generate the next prediction
            next_pred = model.predict(current_sequence, verbose=0)
            predictions.append(next_pred[0][0])
            
            # Update the sequence for the next prediction
            new_sequence = np.append(current_sequence[:, 1:], [[next_pred[0][0]]], axis=1)
            current_sequence = new_sequence.reshape((1, look_back, 1))
            
        return np.array(predictions)
    
    def run_analysis(self):
        """Run the flexible lookback analysis for all validation periods and lookback values."""
        results = []
        
        for val_start, val_end in self.validation_periods:
            for look_back in self.look_back_periods:
                print(f"Analyzing validation period {val_start} to {val_end} with lookback {look_back}")
                
                # Create a copy of the dataframe
                df = self.df.copy()
                
                # Split data
                train = df[df['Month'] <= val_start]
                val = df[(df['Month'] >= val_start) & (df['Month'] <= val_end)]
                test = df[df['Month'] >= '2020-01-01']
                
                # Create datasets for LSTM
                trainX, trainY = self.create_dataset(train['Deaths'], look_back)
                valX, valY = self.create_dataset(val['Deaths'], look_back)
                testX, testY = self.create_dataset(test['Deaths'], look_back)
                
                # Reshape inputs for LSTM
                trainX = trainX.reshape(trainX.shape[0], trainX.shape[1], 1)
                valX = valX.reshape(valX.shape[0], valX.shape[1], 1)
                testX = testX.reshape(testX.shape[0], testX.shape[1], 1)
                
                # Build and train initial LSTM model on the training data
                model = Sequential()
                model.add(LSTM(50, activation='relu', input_shape=(look_back, 1)))
                model.add(Dense(1))
                model.compile(loss='mean_squared_error', optimizer='adam')
                model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=0)
                
                # Prepare the initial sequence for validation forecast
                val_initial_sequence = np.array(train['Deaths'][-look_back:]).reshape((look_back, 1))
                val_initial_sequence = np.array([val_initial_sequence])
                
                # Generate validation predictions
                valPredict = self.generate_forecast(model, val_initial_sequence, look_back, num_predictions=len(valX))
                
                # Retrain the LSTM model on both training and validation data
                combined_train_val = pd.concat([train, val], axis=0)
                combinedX, combinedY = self.create_dataset(combined_train_val['Deaths'], look_back)
                combinedX = combinedX.reshape(combinedX.shape[0], combinedX.shape[1], 1)
                model.fit(combinedX, combinedY, epochs=100, batch_size=1, verbose=0)
                
                # Prepare the initial sequence for test forecast
                test_initial_sequence = np.array([[valPredict[-1]]])
                test_initial_sequence = np.array([test_initial_sequence])
                
                # Generate test predictions
                testPredict = self.generate_forecast(model, test_initial_sequence, look_back, num_predictions=len(testX))
                
                # Generate train predictions
                trainPredict = model.predict(trainX, verbose=0)
                
                # LSTM metrics
                lstm_mape = mean_absolute_percentage_error(testY, testPredict)
                lstm_mse = mean_squared_error(testY, testPredict)
                lstm_rmse = np.sqrt(lstm_mse)
                
                # Combine predictions for visualization
                combined_array = [0] * look_back + trainPredict.flatten().tolist() + valPredict.flatten().tolist() + testPredict.flatten().tolist()
                
                # Add LSTM predictions to dataframe
                df['LSTM Predictions'] = combined_array[:len(df)]
                
                # SARIMA model training on combined training + validation data
                sarima_model = SARIMAX(
                    combined_train_val['Deaths'], 
                    order=(1, 1, 1), 
                    seasonal_order=(1, 1, 1, 12),
                    enforce_stationarity=False, 
                    enforce_invertibility=False
                )
                sarima_result = sarima_model.fit(disp=False)
                
                # Generate SARIMA predictions
                sarima_predictions = sarima_result.predict(start=0, end=df.shape[0]-1, dynamic=False)
                df['SARIMA Predictions'] = sarima_predictions
                
                # Extract SARIMA test predictions
                sarimaTestPredict = df[df['Month'] > '2020-01-01']['SARIMA Predictions']
                
                # SARIMA metrics
                sarima_mape = mean_absolute_percentage_error(testY, sarimaTestPredict)
                sarima_mse = mean_squared_error(testY, sarimaTestPredict)
                sarima_rmse = np.sqrt(sarima_mse)
                
                # Calculate confidence intervals
                lower_bound_test, upper_bound_test = self.calculate_confidence_intervals(testPredict)
                lower_bound_sarima, upper_bound_sarima = self.calculate_confidence_intervals(sarimaTestPredict)
                
                # Calculate CI overlap
                ci_overlap = self.calculate_overlap(
                    lower_bound_test, upper_bound_test, 
                    lower_bound_sarima, upper_bound_sarima
                )
                
                # Store results
                result = {
                    'Validation Period': f"{val_start} to {val_end}",
                    'Look-back': look_back,
                    'LSTM MAPE': lstm_mape,
                    'LSTM MSE': lstm_mse,
                    'LSTM RMSE': lstm_rmse,
                    'SARIMA MAPE': sarima_mape,
                    'SARIMA MSE': sarima_mse,
                    'SARIMA RMSE': sarima_rmse,
                    'CI Overlap %': ci_overlap
                }
                results.append(result)
                
                # Save the dataframe with predictions
                df.to_csv(os.path.join(
                    self.output_dir, 
                    f"lookback_{look_back}_valperiod_{val_start.replace('-', '')}_to_{val_end.replace('-', '')}.csv"
                ))
                
                # Create visualization
                self._create_plot(
                    df, 
                    val_start, 
                    val_end, 
                    look_back, 
                    lower_bound_test, 
                    upper_bound_test, 
                    lower_bound_sarima, 
                    upper_bound_sarima,
                    ci_overlap
                )
        
        # Save results to CSV
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(self.output_dir, "flexible_lookback_results.csv"), index=False)
        
        return results_df
    
    def _create_plot(self, df, val_start, val_end, look_back, 
                    lstm_lower, lstm_upper, sarima_lower, sarima_upper, overlap_pct):
        """Create visualization comparing LSTM and SARIMA predictions.
        
        Args:
            df (DataFrame): DataFrame with actual and predicted values.
            val_start (str): Start date of validation period.
            val_end (str): End date of validation period.
            look_back (int): Lookback period used.
            lstm_lower (array): Lower bound of LSTM confidence interval.
            lstm_upper (array): Upper bound of LSTM confidence interval.
            sarima_lower (array): Lower bound of SARIMA confidence interval.
            sarima_upper (array): Upper bound of SARIMA confidence interval.
            overlap_pct (float): Percentage of CI overlap.
        """
        plt.figure(figsize=(14, 8))
        
        # Extract test period data
        test_df = df[df['Month'] >= '2020-01-01']
        
        # Plot actual data
        plt.plot(df['Month'], df['Deaths'], 'k-', label='Actual Deaths', linewidth=2)
        
        # Plot predictions
        plt.plot(df['Month'], df['LSTM Predictions'], 'b-', label='LSTM Predictions', alpha=0.7)
        plt.plot(df['Month'], df['SARIMA Predictions'], 'r-', label='SARIMA Predictions', alpha=0.7)
        
        # Plot confidence intervals for test period
        test_months = test_df['Month'].values
        plt.fill_between(
            test_months, 
            np.append(lstm_lower, [np.nan] * (len(test_months) - len(lstm_lower))), 
            np.append(lstm_upper, [np.nan] * (len(test_months) - len(lstm_upper))), 
            color='blue', alpha=0.2, label='LSTM 95% CI'
        )
        plt.fill_between(
            test_months, 
            np.append(sarima_lower, [np.nan] * (len(test_months) - len(sarima_lower))), 
            np.append(sarima_upper, [np.nan] * (len(test_months) - len(sarima_upper))), 
            color='red', alpha=0.2, label='SARIMA 95% CI'
        )
        
        # Add vertical lines for train/validation/test splits
        plt.axvline(x=pd.to_datetime(val_start), color='g', linestyle='--', 
                   label=f'Validation Start ({val_start})')
        plt.axvline(x=pd.to_datetime(val_end), color='m', linestyle='--', 
                   label=f'Test Start ({val_end})')
        
        # Customize plot
        plt.title(f'Comparison of LSTM vs SARIMA Models with {look_back}-Month Lookback', fontsize=16)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Deaths', fontsize=14)
        plt.legend(loc='best', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add CI overlap info
        plt.figtext(0.5, 0.01, f'Test Period CI Overlap: {overlap_pct:.2f}%', ha='center', fontsize=12)
        
        # Add RMSE info
        lstm_rmse = np.sqrt(mean_squared_error(
            test_df['Deaths'].values[:len(lstm_lower)], 
            test_df['LSTM Predictions'].values[:len(lstm_lower)]
        ))
        sarima_rmse = np.sqrt(mean_squared_error(
            test_df['Deaths'].values[:len(sarima_lower)], 
            test_df['SARIMA Predictions'].values[:len(sarima_lower)]
        ))
        
        plt.figtext(0.5, 0.04, 
                   f'LSTM RMSE: {lstm_rmse:.2f}, SARIMA RMSE: {sarima_rmse:.2f}', 
                   ha='center', fontsize=12)
        
        # Save plot
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Adjust for figtext
        plt.savefig(os.path.join(
            self.output_dir, 
            f"plot_lookback_{look_back}_valperiod_{val_start.replace('-', '')}_to_{val_end.replace('-', '')}.png"
        ))
        plt.close()
        
    def plot_optimal_lookback(self, results_df=None):
        """Create a visualization of model performance across different lookback periods.
        
        Args:
            results_df (DataFrame, optional): Results DataFrame. If None, will load from file.
            
        Returns:
            None
        """
        if results_df is None:
            results_file = os.path.join(self.output_dir, "flexible_lookback_results.csv")
            if not os.path.exists(results_file):
                raise FileNotFoundError(f"Results file not found: {results_file}")
            results_df = pd.read_csv(results_file)
        
        # Create a plot for each validation period
        for val_period in results_df['Validation Period'].unique():
            period_results = results_df[results_df['Validation Period'] == val_period]
            
            plt.figure(figsize=(12, 10))
            
            # Create subplots
            plt.subplot(2, 1, 1)
            plt.plot(period_results['Look-back'], period_results['LSTM RMSE'], 'bo-', label='LSTM RMSE')
            plt.plot(period_results['Look-back'], period_results['SARIMA RMSE'], 'ro-', label='SARIMA RMSE')
            plt.title(f'RMSE by Lookback Period for Validation: {val_period}', fontsize=14)
            plt.xlabel('Lookback Period (Months)', fontsize=12)
            plt.ylabel('RMSE', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            plt.subplot(2, 1, 2)
            plt.plot(period_results['Look-back'], period_results['LSTM MAPE'], 'bo-', label='LSTM MAPE (%)')
            plt.plot(period_results['Look-back'], period_results['SARIMA MAPE'], 'ro-', label='SARIMA MAPE (%)')
            plt.title(f'MAPE by Lookback Period for Validation: {val_period}', fontsize=14)
            plt.xlabel('Lookback Period (Months)', fontsize=12)
            plt.ylabel('MAPE (%)', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Add CI overlap to the plot
            for i, row in period_results.iterrows():
                plt.annotate(
                    f"CI Overlap: {row['CI Overlap %']:.1f}%",
                    (row['Look-back'], row['LSTM MAPE']),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha='center'
                )
            
            plt.tight_layout()
            plt.savefig(os.path.join(
                self.output_dir, 
                f"optimal_lookback_plot_{val_period.replace(' ', '_').replace('-', '')}.png"
            ))
            plt.close()
        
        # Create a comprehensive plot with all validation periods
        plt.figure(figsize=(15, 12))
        
        plt.subplot(2, 1, 1)
        for val_period in results_df['Validation Period'].unique():
            period_results = results_df[results_df['Validation Period'] == val_period]
            plt.plot(period_results['Look-back'], period_results['LSTM RMSE'], 'o-', 
                    label=f'LSTM: {val_period}')
            plt.plot(period_results['Look-back'], period_results['SARIMA RMSE'], 'o--', 
                    label=f'SARIMA: {val_period}')
        
        plt.title('RMSE by Lookback Period for All Validation Periods', fontsize=16)
        plt.xlabel('Lookback Period (Months)', fontsize=14)
        plt.ylabel('RMSE', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        
        plt.subplot(2, 1, 2)
        for val_period in results_df['Validation Period'].unique():
            period_results = results_df[results_df['Validation Period'] == val_period]
            plt.plot(period_results['Look-back'], period_results['CI Overlap %'], 'o-', 
                    label=f'CI Overlap: {val_period}')
        
        plt.title('Confidence Interval Overlap by Lookback Period', fontsize=16)
        plt.xlabel('Lookback Period (Months)', fontsize=14)
        plt.ylabel('CI Overlap (%)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "comprehensive_lookback_analysis.png"))
        plt.close()
        
    def find_optimal_parameters(self, results_df=None):
        """Find the optimal lookback period and validation period.
        
        Args:
            results_df (DataFrame, optional): Results DataFrame. If None, will load from file.
            
        Returns:
            dict: Optimal parameters.
        """
        if results_df is None:
            results_file = os.path.join(self.output_dir, "flexible_lookback_results.csv")
            if not os.path.exists(results_file):
                raise FileNotFoundError(f"Results file not found: {results_file}")
            results_df = pd.read_csv(results_file)
        
        # Find optimal parameters based on LSTM RMSE
        lstm_best_idx = results_df['LSTM RMSE'].idxmin()
        lstm_best = results_df.loc[lstm_best_idx]
        
        # Find optimal parameters based on SARIMA RMSE
        sarima_best_idx = results_df['SARIMA RMSE'].idxmin()
        sarima_best = results_df.loc[sarima_best_idx]
        
        # Find optimal parameters based on CI Overlap (highest overlap)
        overlap_best_idx = results_df['CI Overlap %'].idxmax()
        overlap_best = results_df.loc[overlap_best_idx]
        
        # Create summary
        optimal_params = {
            'lstm_best': {
                'validation_period': lstm_best['Validation Period'],
                'look_back': int(lstm_best['Look-back']),
                'rmse': lstm_best['LSTM RMSE'],
                'mape': lstm_best['LSTM MAPE'],
                'ci_overlap': lstm_best['CI Overlap %']
            },
            'sarima_best': {
                'validation_period': sarima_best['Validation Period'],
                'look_back': int(sarima_best['Look-back']),
                'rmse': sarima_best['SARIMA RMSE'],
                'mape': sarima_best['SARIMA MAPE'],
                'ci_overlap': sarima_best['CI Overlap %']
            },
            'overlap_best': {
                'validation_period': overlap_best['Validation Period'],
                'look_back': int(overlap_best['Look-back']),
                'lstm_rmse': overlap_best['LSTM RMSE'],
                'sarima_rmse': overlap_best['SARIMA RMSE'],
                'ci_overlap': overlap_best['CI Overlap %']
            }
        }
        
        # Save optimal parameters
        with open(os.path.join(self.output_dir, "optimal_parameters.json"), 'w') as f:
            import json
            json.dump(optimal_params, f, indent=2)
        
        return optimal_params


if __name__ == "__main__":
    """Run the flexible lookback analysis."""
    # Set random seed for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Configure the path to the data
    data_path = 'data/state_month_overdose.xlsx'
    output_dir = 'results/flexible_lookback'
    
    # Create and run the analysis
    analysis = FlexibleLookbackAnalysis(data_path, output_dir)
    
    # Set custom validation periods and lookback periods if needed
    # analysis.validation_periods = [('2019-01-01', '2020-01-01')]
    # analysis.look_back_periods = [3, 5, 7]
    
    # Run the analysis
    results_df = analysis.run_analysis()
    
    # Create plots
    analysis.plot_optimal_lookback(results_df)
    
    # Find optimal parameters
    optimal_params = analysis.find_optimal_parameters(results_df)
    
    print("Flexible lookback analysis completed.")
    print(f"Results saved to {output_dir}")
    print("\nOptimal Parameters:")
    print(f"LSTM Best: Validation Period = {optimal_params['lstm_best']['validation_period']}, "
          f"Look-back = {optimal_params['lstm_best']['look_back']}, "
          f"RMSE = {optimal_params['lstm_best']['rmse']:.2f}")
    print(f"SARIMA Best: Validation Period = {optimal_params['sarima_best']['validation_period']}, "
          f"Look-back = {optimal_params['sarima_best']['look_back']}, "
          f"RMSE = {optimal_params['sarima_best']['rmse']:.2f}")
    print(f"Best CI Overlap: Validation Period = {optimal_params['overlap_best']['validation_period']}, "
          f"Look-back = {optimal_params['overlap_best']['look_back']}, "
          f"Overlap = {optimal_params['overlap_best']['ci_overlap']:.2f}%")