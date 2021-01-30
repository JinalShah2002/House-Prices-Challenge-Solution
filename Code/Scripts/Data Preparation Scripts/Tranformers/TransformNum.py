"""
@author: Jinal Shah

This file will contain code for
the transform_num transformer

This transformer transforms given
numerical features by either
doing a log(x+1), x ** .5, or
log(x) on the given feature.
This is needed to put the features
into a normal distribution.

"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class TransformNum(BaseEstimator, TransformerMixin):
    # Constructor
    def __init__(self, transform_dict):
        self.transform_dict = transform_dict

    # Fit method
    def fit(self, X, y=None):
        return self

    # Transform method
    def transform(self, X, y=None):
        for i in self.transform_dict.keys():
            try:
                if self.transform_dict[i] == 'log(x+1)':
                    X[i] = np.log(X[i] + 1)
                elif self.transform_dict[i] == 'log(x)':
                    X[i] = np.log(X[i])
                elif self.transform_dict[i] == 'x ** .5':
                    X[i] = X[i] ** .5
            except KeyError:
                print(f'Key Error for {i}  in TransformNum!!!')
                return KeyError
        return X

