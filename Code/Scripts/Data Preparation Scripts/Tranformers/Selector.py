"""
@author: Jinal Shah

This file will contain code for
the selector transformer

This transformer returns the selected
features from a pandas dataframe &
returns them as a numpy array.

"""
# Importing needed libraries
from sklearn.base import BaseEstimator, TransformerMixin


class Selector(BaseEstimator, TransformerMixin):
    # Constructor
    def __init__(self, features):
        self.features = features

    # Fit method
    def fit(self, X, y=None):
        return self

    # Transform method
    def transform(self, X, y=None):
        try:
            return X[self.features].values
        except KeyError:
            return KeyError
