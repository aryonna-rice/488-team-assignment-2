from sklearn.base import BaseEstimator, TransformerMixin

class EarliestCRLineDateImputer(BaseEstimator, TransformerMixin):
    def __init__(self, issue_d_column='issue_d'):
        self.issue_d_column = issue_d_column

    def fit(self, X, y=None):
        # Calculate the median earliest_cr_line for each FICO score condition
        self.median_high_fico_ = X.loc[X['fico_range_high'].notna(), 'earliest_cr_line'].median()
        self.median_low_fico_ = X.loc[X['fico_range_low'].notna(), 'earliest_cr_line'].median()
        return self

    def transform(self, X):
        X = X.copy()
        
        # Condition 1: If both FICO scores are NaN, impute earliest_cr_line with issue_d
        condition_no_fico = X['fico_range_high'].isna() & X['fico_range_low'].isna()
        X.loc[condition_no_fico, 'earliest_cr_line'] = X.loc[condition_no_fico, self.issue_d_column]
        
        # Condition 2: Impute based on FICO score presence
        # For rows with a high FICO score
        condition_high_fico = X['fico_range_high'].notna() & X['earliest_cr_line'].isna()
        X.loc[condition_high_fico, 'earliest_cr_line'] = self.median_high_fico_
        
        # For rows with a low FICO score (if distinct logic is needed)
        condition_low_fico = X['fico_range_low'].notna() & X['earliest_cr_line'].isna()
        X.loc[condition_low_fico, 'earliest_cr_line'] = self.median_low_fico_

        return X
