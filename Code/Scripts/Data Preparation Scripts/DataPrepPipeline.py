"""

@author: Jinal Shah

This script will house my custom data preprocessing pipeline

"""
# Basic Setup
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline, FeatureUnion
from Tranformers import Remove, Replace, Selector, TransformNum


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
                             'GarageDec', 'CatRemod', 'CatYearBuilt']

        # Dictionary to store specific missing value replacements --> for categorical features
        self.missing_dict = {
            'FireplaceQu': 'NF',
            'BsmtQual': 'N/A',
            'BsmtExposure': 'N/A',
            'BsmtFinType1': 'N/A',
            'GarageType': 'N/A',
            'GarageFinish': 'N/A',
            'GarageDec': 'N/A',
            'MasVnrType': 'None',
            'LotFrontage': np.nanmedian(self.X['LotFrontage']),
            'MasVnrArea': np.nanmedian(self.X['MasVnrArea']),
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

    """ 
    Feature Engineering
    
    This function will conduct all custom feature 
    engineering for this project
    
    """
    def feature_eng(self):
        # Garage Dec Feature
        self.X['GarageDec'] = self.X['GarageYrBlt'] // 10 * 10

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

    # Building the Pipeline
    def pipeline(self):
        # Calling feature engineering beforehand
        self.feature_eng()
        
        # Building the Pipeline with all Custom Transformations, except Selector()
        custom_transform = Pipeline([
            ('Removal', Remove.Remove(self.remove_features)),
            ('Missing', Replace.ReplaceMissing(self.missing_dict)),
            ('Transform', TransformNum.TransformNum(self.num_transform)),
        ])

        # Transforming the numerical features
        num_pipeline = Pipeline([
            ('Selector', Selector.Selector(self.num_features)),
            ('Feature Scale', StandardScaler())
        ])

        # Categorical Pipeline
        cat_pipeline = Pipeline([
            ('Selector', Selector.Selector(self.cat_features)),
            ('Encoding', OneHotEncoder()),
        ])

        # Full Pipeline
        pipeline = FeatureUnion([
            ('custom_transform', custom_transform),
            ('num_pipeline', num_pipeline),
    #        ('cat_pipeline', cat_pipeline),
        ])

        return pipeline

    # Transforming the data
    def transform(self):
        pipeline = self.pipeline()
        # Adjusting the return statement based on dataset type
        if self.train:
            return pipeline.fit_transform(self.X, self.y)
        else:
            return pipeline.transform(self.X)
