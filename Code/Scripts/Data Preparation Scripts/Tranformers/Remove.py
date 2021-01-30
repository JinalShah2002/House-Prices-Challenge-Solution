"""
@author: Jinal Shah

This file will contain code for
the removal transformer

This transformer removes a given
set of features from a pandas
dataframe & returns the editted
dataframe.

"""
# Importing libraries
from sklearn.base import BaseEstimator, TransformerMixin


class Remove(BaseEstimator, TransformerMixin):
    # Constructor
    def __init__(self, features):
        self.features = features

    # Fit method
    def fit(self, X, y=None):
        return self

    # Transform method
    def transform(self, X, y=None):
        try:
            return X.drop(self.features, axis=1)
        except AttributeError:
            print('Attribute Error in Remove!!!')
            return AttributeError

