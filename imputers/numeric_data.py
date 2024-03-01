import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class NumericDataImputer(BaseEstimator, TransformerMixin):
    def __init__(self, group_column):
        """Custom imputer for numerical data that imputes missing values based on the median of groups defined by a categorical column. """
        self.group_column = group_column
        self.medians_ = None

    def fit(self, X, y=None):
        # Ensure that X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input is not a pandas DataFrame")
        
        # Group by the specified column and calculate medians for each group
        self.medians_ = X.groupby(self.group_column).median()
        return self

    def transform(self, X):
        """Impute missing values in the numerical columns based on the median of their group."""
        # Check if fit has been called
        if self.medians_ is None:
            raise RuntimeError("Fit method must be called before transform.")
        
        # Ensure that X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input is not a pandas DataFrame")
        
        # Make a copy of the DataFrame to avoid changing the original data
        X_transformed = X.copy()
        
        # Iterate over each group and column to impute missing values
        for name, group in X.groupby(self.group_column):
            for column in group.select_dtypes(include=['float']).columns:
                if column in self.medians_.columns:
                    # Impute missing values using the median of the group
                    median_value = self.medians_.loc[name, column]
                    X_transformed.loc[X_transformed[self.group_column] == name, column] = group[column].fillna(median_value)
        
        return X_transformed