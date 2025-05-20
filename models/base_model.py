"""Base model class that all time series models will inherit from."""

class BaseModel:
    """Abstract base class for time series forecasting models."""
    
    def fit(self, X, y=None):
        """
        Fit the model to training data.
        
        Parameters
        ----------
        X : array-like
            Training data (features or time series)
        y : array-like, optional
            Target values if applicable
            
        Returns
        -------
        self : object
            Fitted model instance
        """
        raise NotImplementedError("Subclasses must implement fit()")

    def predict(self, X):
        """
        Generate predictions for new data.
        
        Parameters
        ----------
        X : array-like
            Data for which to generate predictions
            
        Returns
        -------
        array-like
            Predictions
        """
        raise NotImplementedError("Subclasses must implement predict()")