"""

@author: Jinal Shah

This script will house my custom data preprocessing pipeline

"""
# Basic Setup
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from Tranformers import Remove, TransformNum


class Preparation:
    # Constructor
    def __init__(self, X, y=None, train=False):
        # Storing the feature matrix & the dependent vector
        self.X = X
        self.y = y

        # Marking whether the data is a test file or a training file
        self.train = train

        # a list of features to remove
        self.remove_features = ['BsmtFinType2', 'GarageQual', 'GarageCond', 'Electrical', '3SsnPorch', 'BsmtFinSF2',
                                'MiscVal', 'LowQualFinSF', 'PoolArea', 'ScreenPorch', 'EnclosedPorch', 'Alley',
                                'PoolQC', 'Fence', 'MiscFeature', 'Street', 'LandContour', 'Utilities',
                                'LotConfig', 'LandSlope', 'Condition1', 'Condition2', 'BldgType', 'RoofStyle',
                                'RoofMatl', 'ExterCond', 'BsmtCond', 'Heating', 'CentralAir', 'Functional',
                                'PavedDrive', 'SaleType', 'BsmtHalfBath', 'KitchenAbvGr', 'YrSold']
        
        # Removing ID if it not a test set
        if not self.train:
            self.id = X['Id']
            self.remove_features.append('Id')
        else:
            self.remove_features.append('Id')

        # lists of numerical + categorical features
        self.num_features = ['LotArea', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'GarageArea', 'OpenPorchSF',
                             'BsmtFinSF1', 'BsmtUnfSF', 'WoodDeckSF', 'MasVnrArea', 'LotFrontage']

        self.cat_features = ['MSZoning', 'LotShape', 'Neighborhood', 'HouseStyle', 'Exterior1st', 'Exterior2nd',
                             'MasVnrType', 'ExterQual', 'Foundation', 'BsmtQual', 'BsmtExposure', 'BsmtFinType1',
                             'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageType', 'GarageFinish', 'SaleCondition',
                             'MSSubClass', 'OverallQual', 'OverallCond', 'BsmtFullBath', 'FullBath',
                             'HalfBath', 'BedroomAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'MoSold',
                              'CatRemod', 'CatYearBuilt']

        # Dictionary to store specific missing value replacements --> for categorical features
        self.missing_dict = {
            'BsmtFinSF1': np.nanmedian(self.X['BsmtFinSF1']),
            'BsmtUnfSF': np.nanmedian(self.X['BsmtUnfSF']),
            'TotalBsmtSF': np.nanmedian(self.X['TotalBsmtSF']),
            'LotFrontage': np.nanmedian(self.X['LotFrontage']),
            'MasVnrArea': np.nanmedian(self.X['MasVnrArea']),
            'MSZoning': 'RL',
            'MasVnrType': 'None',
            'BsmtQual': 'None',
            'BsmtExposure': 'None',
            'BsmtFinType1': 'None',
            'FireplaceQu': 'None',
            'GarageFinish': 'None',
            'GarageType': 'None',
            'Exterior1st': 'VinylSd',
            'Exterior2nd': 'VinylSd',
            'BsmtFullBath': 0.0,
            'KitchenQual': 'TA',
            'GarageCars': 2.0,
        }

        # Dictionary to store the transformations for the numerical features
        self.num_transform = {
            'LotArea': 'log(x+1)',
            'TotalBsmtSF': 'log(x+1)',
            '1stFlrSF': 'log(x+1)',
            '2ndFlrSF': 'log(x+1)',
            'GrLivArea': 'log(x+1)',
            'GarageArea': 'log(x+1)',
            'OpenPorchSF': 'log(x+1)',
            'LotFrontage': 'log(x)',
            'BsmtFinSF1': 'x ** .5',
            'BsmtUnfSF': 'x ** .5',
            'WoodDeckSF': 'x ** .5',
            'MasVnrArea': 'x ** .5',
        }

        # Initializing the Transformers
        self.custom_transform = Pipeline([
            ('Removal', Remove.Remove(self.remove_features)),
            ('Transform', TransformNum.TransformNum(self.num_transform)),
        ])
        self.scaler = StandardScaler()

    # This function works with the missing values in the data
    def replace_missing(self):
        temp = ''
        for key in self.missing_dict.keys():
            temp = self.X[key].fillna(self.missing_dict[key])
            self.X[key] = temp
    """ 
    Feature Engineering
    
    This function will conduct all custom feature 
    engineering for this project
    
    """
    def feature_eng(self):
        # Cat Year Feature
        self.X['CatRemod'] = 'N'
        self.X['CatYearBuilt'] = 'N'

        # Iterating through the cat year features to correct the values
        for index in self.X.index:
            if self.X.loc[index, 'YearRemodAdd'] >= 2000:
                self.X.loc[index, 'CatRemod'] = 'Y'
            if self.X.loc[index, 'YearBuilt'] >= 1975:
                self.X.loc[index, 'CatYearBuilt'] = 'Y'

        # Removing the original features
        self.X = self.X.drop('GarageYrBlt', axis=1)
        self.X = self.X.drop('YearRemodAdd', axis=1)
        self.X = self.X.drop('YearBuilt', axis=1)

    # Performing Transformations
    def transform(self):
        # Replacing the missing values
        self.replace_missing()

        # Calling feature engineering beforehand
        self.feature_eng()

        # Preliminary Transformations
        self.X[self.cat_features] = self.X[self.cat_features].astype(str)
        self.X[self.num_features] = self.X[self.num_features].astype(float)

        if self.train:
            self.X = self.custom_transform.fit_transform(self.X, self.y)
            self.X[self.num_features] = self.scaler.fit_transform(self.X[self.num_features], self.y)
            self.X = pd.get_dummies(self.X)
            self.y = np.log(self.y + 1)
        else:
            self.X = self.custom_transform.transform(self.X)
            self.X[self.num_features] = self.scaler.transform(self.X[self.num_features])
            self.X = pd.get_dummies(self.X, dtype=int)

        return self.X, self.y
    """
    
    Creating Setters for X, y ,and train
    
    """
    def set_X(self, X):
        self.X = X

    def set_y(self, y):
        self.y = y

    def set_train(self, train):
        self.train = train

