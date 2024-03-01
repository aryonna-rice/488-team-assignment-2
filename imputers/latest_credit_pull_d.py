from sklearn.base import BaseEstimator, TransformerMixin

class LatestCreditPullDateImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        # Calculate the median earliest_cr_line for each FICO score condition
        self.loan_status_groups_ = X.groupby('loan_status')
        self.median_differences_ = self.loan_status_groups_.apply(lambda x: ((x['last_credit_pull_d'][x['last_credit_pull_d'].notna()] - x['issue_d'])).median())
        return self
    
    def transform(self, X):
        X = X.copy()
        for status, median_diff in self.median_differences_.items():
            indices = X[(X['loan_status'] == status) & X['last_credit_pull_d'].isna()].index
            X.loc[indices, 'last_credit_pull_d'] = X.loc[indices, 'issue_d'] + median_diff
        return X