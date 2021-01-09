"""

@author: Jinal Shah

This script will house all custom transformations

"""

# Importing some libraries
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

"""
Selector():
This transformer is a feature selector. It allows me to select certain features.
This will allow me to select numerical + categorical features for specific transformations.

"""


class Selector(BaseEstimator, TransformerMixin):
    # Constructor
    def __init__(self, features):
        self.features = features

    # Fit method
    def fit(self, X, y=None):
        return self

    # Transform method
    def transform(self, X, y=None):
        return X[self.features].values


"""
Remove():
This transformer allows me to remove a given set of features.

"""


class Remove(BaseEstimator, TransformerMixin):
    # Constructor
    def __init__(self, features):
        self.features = features

    # Fit method
    def fit(self, X, y=None):
        return self

    # Transform method
    def transform(self, X, y=None):
        return X.drop(self.features, axis=1)


"""
ReplaceMissing():
This transformer allows me to replace the missing values in 
the relevant features. It takes in a dictionary. The keys of
the dictionary are the features and the values are the values
to replace the missing values with.

"""


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
            X[i] = X[i].fillna(self.missing_dict[i])

        return X


""" 
TransformNum():
This transformer allows me to apply numerical transformations to 
the numerical features (log(x+1),etc). The constructor takes a dictionary, 
whose keys are the features to transform and the values are specify the transformation.
The choices of transformations are log(x+1), log(x) ,or x ** .5.

"""


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
            if self.transform_dict[i] == 'log(x+1)':
                X[i] = np.log(X[i] + 1)
            elif self.transform_dict[i] == 'log(x)':
                X[i] = np.log(X[i])
            elif self.transform_dict[i] == 'x ** .5':
                X[i] = X[i] ** .5
        return X


""" 
GarageDec():
This transformer will create a new feature called GarageBltDec.
This feature will categorize GarageYrBlt into Decades. I did this
because a) I would reduce the number of categories greatly and b)
a clearer correlation was found

"""


class GarageDec(BaseEstimator, TransformerMixin):
    # Constructor
    def __init__(self):
        pass

    # Fit method
    def fit(self, X, y=None):
        return self

    # Transform method
    def transform(self, X, y=None):
        X['GarageBltDec'] = int(X['GarageYrBlt'] / 10) * 100
        X = X.drop('GarageYrBlt', axis=1)
        return X


"""
CatYear():
This transformer will categorize a given set of features by year.
For example, YearRemodAdd's values will be categorized into Yes or No
if the remodel year is greater than or equal to 2000. This transformer
is created to reduce the categories + build a stronger correlation.

"""


class CatYear(BaseEstimator, TransformerMixin):
    # Constructor
    def __init__(self, features_dict):
        self.features_dict = features_dict

    # Fit method
    def fit(self, X, y=None):
        return self

    # Transform method
    def transform(self, X, y=None):
        for i in self.features_dict.keys():
            temp = i + '2'
            X[temp] = 'N'

            for j in range(len(X.shape[0])):
                if X[i][j] >= self.features_dict[i]:
                    X[temp] = 'Y'

        # Dropping the originals
        X = X.drop(self.features_dict.keys(), axis=1)
        return X

