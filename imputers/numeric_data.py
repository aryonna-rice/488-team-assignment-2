import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class NumericDataImputer(BaseEstimator, TransformerMixin):
    def __init__(self, group_column):
        """Custom imputer for numerical data that imputes missing values based on the median of groups defined by a categorical column."""
        self.group_column_ = group_column

    def fit(self, X, y=None):
        # Calculate the median for each numerical column within each group defined by group_column_
        self.medians_ = X.groupby(self.group_column_).transform('median')
        return self

    def transform(self, X):
        """Impute missing values in the numerical columns based on the median of their group."""
        # Make a copy of the DataFrame to avoid changing the original data
        X_transformed = X.copy()
        
        # Impute missing values for each numerical column based on the group's median
        numerical_columns = X.select_dtypes(include=['number']).columns
        for column in numerical_columns:
            if column != self.group_column_:  # Ensure we don't try to impute the grouping column itself
                # Use medians calculated during fit to fill missing values
                X_transformed[column].fillna(self.medians_[column], inplace=True)
        return X_transformed
