from sklearn.base import BaseEstimator, TransformerMixin

class EarliestCRLineDateImputer(BaseEstimator, TransformerMixin):
    def __init__(self, issue_d_column='issue_d'):
        self.issue_d_column = issue_d_column

    def fit(self, X, y=None):
        # Group by fico_descriptor and calculate the median earliest_cr_line for each group
        self.medians_by_fico_descriptor_ = X.groupby('fico_descriptor')['earliest_cr_line'].median()
        return self

    def transform(self, X):
        X = X.copy()
        # Impute earliest_cr_line based on fico_descriptor group's median
        for fico_descriptor, median in self.medians_by_fico_descriptor_.items():
            condition = (X['fico_descriptor'] == fico_descriptor) & X['earliest_cr_line'].isna()
            X.loc[condition, 'earliest_cr_line'] = median
        
        # For rows with a null fico_descriptor, assign earliest_cr_line the value of issue_d
        condition_null_fico = X['fico_descriptor'].isna() & X['earliest_cr_line'].isna()
        X.loc[condition_null_fico, 'earliest_cr_line'] = X.loc[condition_null_fico, self.issue_d_column]
        
        return X
