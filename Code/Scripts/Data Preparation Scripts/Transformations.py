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
        try:
            return X[self.features].values
        except KeyError:
            return KeyError


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
        try:
            return X.drop(self.features, axis=1)
        except AttributeError:
            return AttributeError


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
        temp = X.copy()  # Created a temp to preserve original
        for i in self.missing_dict.keys():
            try:
                temp[i] = X[i].fillna(self.missing_dict[i])
            except KeyError:
                return KeyError
        return temp


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
        temp = X.copy()
        for i in self.transform_dict.keys():
            try:
                if self.transform_dict[i] == 'log(x+1)':
                    temp[i] = np.log(X[i] + 1)
                elif self.transform_dict[i] == 'log(x)':
                    temp[i] = np.log(X[i])
                elif self.transform_dict[i] == 'x ** .5':
                    temp[i] = X[i] ** .5
            except KeyError:
                return KeyError
        return temp


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
        # Creating necessary information
        copy = X.copy()
        copy['GarageBltDec'] = copy['GarageYrBlt'] // 10 * 10  # Reminder: // performs int division
        copy = copy.drop('GarageYrBlt', axis=1)
        return copy


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
        copy = X.copy()
        for i in self.features_dict.keys():
            temp = i + '2'
            copy[temp] = 'N'

            for j in range(copy.shape[0]):
                if copy.loc[j, i] >= self.features_dict[i]:
                    copy.loc[j, temp] = 'Y'

        # Dropping the originals
        copy = copy.drop(self.features_dict.keys(), axis=1)
        return copy