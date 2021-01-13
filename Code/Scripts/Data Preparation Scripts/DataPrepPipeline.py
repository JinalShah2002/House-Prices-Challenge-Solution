"""

@author: Jinal Shah

This script will house my custom data preprocessing pipeline

"""
# Importing some libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline, FeatureUnion
from Transformations import CatYear, GarageDec, Selector, TransformNum, ReplaceMissing, Remove


class Preparation:
    # Constructor
    def __init__(self, X,y,train):
        # Storing the feature matrix & the dependent vector
        self.X = X
        self.y = y

        # Marking whether the data is a test file or a training file
        self.train = train

        # a list of features to remove
        self.remove_features = ['BsmtFinType2', 'GarageQual', 'GarageCond', 'Electrical', '3SsnPorch', 'BsmtFinSF2',
                                'MiscVal', 'LowQualFinSF', 'PoolArea', 'ScreenPorch', 'EnclosedPorch', 'Alley',
                                'PoolQC',
                                'Fence', 'MiscFeature', 'Street', 'LandContour', 'Utilites', 'LotConfig', 'LandSlope',
                                'Condition1', 'Condition2', 'BldgType', 'RoofStyle', 'RoofMatl', 'ExterCond',
                                'BsmtCond',
                                'Heating', 'CentralAir', 'Functional', 'PavedDrive', 'SaleType', 'BsmtHalfBath',
                                'KitchenAbvGr', 'YrSold']
        
        # Removing ID if it not a test set
        if self.train:
            self.remove_features.append('Id')

        # lists of numerical + categorical features
        self.num_features = ['LotArea', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'GarageArea', 'OpenPorchSF',
                             'BsmtFinSF1', 'BsmtUnfSF', 'WoodDeckSF', 'MasVnrArea', 'LotFrontage']
        self.cat_features = ['MSZoning', 'LotShape', 'Neighborhood', 'HouseStyle', 'Exterior1st', 'Exterior2nd',
                             'MasVnrType', 'ExterQual', 'Foundation', 'BsmtQual', 'BsmtExposure', 'BsmtFinType1',
                             'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageType', 'GarageFinish', 'SaleCondition',
                             'GarageBltDec', 'MSSubClass', 'OverallQual', 'OverallCond', 'BsmtFullBath', 'FullBath',
                             'HalfBath', 'BedroomAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'MoSold',
                             'RemodAdd2', 'YearBuilt2']

        # Dictionary to store specific missing value replacements --> for categorical features
        self.missing_dict = {
            'FireplaceQu': 'NF',
            'BsmtQual': 'N/A',
            'BsmtCond': 'N/A',
            'BsmtExposure': 'N/A',
            'BsmtFinType1': 'N/A',
            'GarageType': 'N/A',
            'GarageFinish': 'N/A',
            'GarageBltDec': 'N/A',
            'MasVnrType': 'None',
            'LotFrontage': np.nanmedian(self.X['LotFrontage']),
            'MasVnrArea': np.nanmedian(self.X['MasVnrArea']),
        }

        # Dictionary to store the transformations for the numerical features
        self.transform = {
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

        # Dictionary to categorize some features by year
        self.cat_year = {
            'YearRemodAdd': 2000,
            'YearBuilt': 1975,
        }

    # Defining the Pipeline
    def pipeline(self):
        # Feature Engineering Pipeline
        feature_eng = Pipeline([
            ('GarageBltDec', GarageDec()),
            ('CatByYear', CatYear(self.cat_year)),
        ])

        # Removal Pipeline
        removal = Pipeline([
            ('Removal', Remove(self.remove_features)),
        ])

        # Dealing with missing features
        missing = Pipeline([
            ('Missing', ReplaceMissing(self.missing_dict)),
        ])

        # Transforming the numerical features (log,etc)
        transform_num = Pipeline([
            ('Transform', TransformNum(self.transform)),
        ])

        # Transforming the numerical features
        num_pipeline = Pipeline([
            ('Selector', Selector(self.num_features)),
            ('Feature Scale', StandardScaler())
        ])

        # Categorical Pipeline
        cat_pipeline = Pipeline([
            ('Selector', Selector(self.cat_features)),
            ('Encoding', OneHotEncoder()),
        ])

        # Full Pipeline
        pipeline = FeatureUnion([
            ('feature_eng', feature_eng),
            ('removal', removal),
            ('missing', missing),
            ('transform_num', transform_num),
            ('num_pipeline', num_pipeline),
            ('cat_pipeline', cat_pipeline),
        ])

        return pipeline

    # Transforming the data
    def transform(self):
        pipeline = self.pipeline()
        
        # Adjusting the return statement based on dataset type
        if self.train:
            return pipeline.fit_transform(self.X,self.y)
        else:
            return pipeline.transform(self.X)
