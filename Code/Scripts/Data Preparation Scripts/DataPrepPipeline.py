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
    def __init__(self, data, train=True, submission=False):
        # Storing the feature matrix & the dependent vector
        self.y = data['SalePrice']
        self.X = data.drop('SalePrice', axis=1)

        # Marking whether the data is a test file or a training file
        self.train = train
        self.submission = submission

        # a list of features to remove
        self.remove_features = ['Id','BsmtFinType2', 'GarageQual', 'GarageCond', 'Electrical', '3SsnPorch', 'BsmtFinSF2',
                                'MiscVal', 'LowQualFinSF', 'PoolArea', 'ScreenPorch', 'EnclosedPorch', 'Alley',
                                'PoolQC', 'Fence', 'MiscFeature', 'Street', 'LandContour', 'Utilities',
                                'LotConfig', 'LandSlope', 'Condition1', 'Condition2', 'BldgType', 'RoofStyle',
                                'RoofMatl', 'ExterCond', 'BsmtCond', 'Heating', 'CentralAir', 'Functional',
                                'PavedDrive', 'SaleType', 'BsmtHalfBath', 'KitchenAbvGr', 'YrSold']

        # lists of numerical + categorical features
        self.num_features = ['LotArea', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'GarageArea', 'OpenPorchSF',
                             'BsmtFinSF1', 'BsmtUnfSF', 'WoodDeckSF', 'MasVnrArea', 'LotFrontage', 'OverallCond'
                             , 'BsmtFullBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'TotRmsAbvGrd', 'Fireplaces',
                             'GarageCars']

        self.cat_features = ['MSZoning', 'LotShape', 'Neighborhood', 'HouseStyle', 'Exterior1st', 'Exterior2nd',
                             'MasVnrType', 'ExterQual', 'Foundation', 'BsmtQual', 'BsmtExposure', 'BsmtFinType1',
                             'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageType', 'GarageFinish', 'SaleCondition',
                             'MSSubClass', 'OverallQual', 'MoSold', 'CatRemod', 'CatYearBuilt']

        # Dictionary to store specific missing value replacements --> for categorical features
        self.missing_dict = {
            'BsmtFinSF1': np.nanmedian(self.X['BsmtFinSF1']),
            'BsmtUnfSF': np.nanmedian(self.X['BsmtUnfSF']),
            'TotalBsmtSF': np.nanmedian(self.X['TotalBsmtSF']),
            'LotFrontage': np.nanmedian(self.X['LotFrontage']),
            'MasVnrArea': np.nanmedian(self.X['MasVnrArea']),
            'GarageArea': np.nanmedian(self.X['GarageArea']),
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
            'BsmtFullBath': np.nanmedian(self.X['BsmtFullBath']),
            'KitchenQual': 'TA',
            'GarageCars': np.nanmedian(self.X['GarageCars']),
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
            self.X['MSSubClass_150'] = 0
            self.X['SalePrice'] = self.y
        elif self.submission:
            self.X = self.custom_transform.transform(self.X)
            self.X[self.num_features] = self.scaler.transform(self.X[self.num_features])
            self.X = pd.get_dummies(self.X, dtype=int)
            self.X['HouseStyle_2.5Fin'] = 0
            self.X['Exterior1st_ImStucc'] = 0
            self.X['Exterior1st_Stone'] = 0
            self.X['Exterior2nd_Other'] = 0
        else:
            self.X = self.custom_transform.transform(self.X)
            self.X[self.num_features] = self.scaler.transform(self.X[self.num_features])
            self.X = pd.get_dummies(self.X, dtype=int)
            self.y = np.log(self.y + 1)
            self.X['SalePrice'] = self.y
            self.X['Neighborhood_Blueste'] = 0
            self.X['OverallQual_1'] = 0
            self.X['OverallQual_2'] = 0
            self.X['Exterior1st_AsphShn'] = 0
            self.X['Exterior1st_CBlock'] = 0
            self.X['Exterior1st_ImStucc'] = 0
            self.X['Exterior1st_Stone'] = 0
            self.X['Exterior2nd_CBlock'] = 0
            self.X['Exterior2nd_Other'] = 0
            self.X['Foundation_Stone'] = 0
            self.X['HeatingQC_Po'] = 0
            self.X['GarageType_2Types'] = 0
            self.X['MSSubClass_150'] = 0

        return self.X

    # Method to change data to testing
    def change_data_test(self, data):
        self.y = data['SalePrice']
        self.X = data.drop('SalePrice', axis=1)
        self.train = False
        self.submission = False

    # Method to change data to submission
    def change_data_sub(self, data):
        self.train = False
        self.submission = True
        self.X = data
