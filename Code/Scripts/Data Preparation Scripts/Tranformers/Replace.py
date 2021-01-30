"""
@author: Jinal Shah

This file will contain code for
the replace_missing transformer

This transformer replaces missing
values in a dataframe

"""
from sklearn.base import BaseEstimator, TransformerMixin


class ReplaceMissing(BaseEstimator, TransformerMixin):
    # Constructor
    def __init__(self, missing_dict):
        self.missing_dict = missing_dict

    # Fit method
    def fit(self, X, y=None):
        return self

    # Transform method
    def transform(self, X, y=None):
        for i in self.missing_dict.keys():
            try:
                X[i] = X[i].fillna(self.missing_dict[i])
            except KeyError:
                print(f'Key Error for {i}  in Replace!!!')
                return KeyError
        return X

