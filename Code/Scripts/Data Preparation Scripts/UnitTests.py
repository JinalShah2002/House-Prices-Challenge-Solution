"""

@author Jinal Shah

This file will conduct all unit tests on the custom built transformers.
It is important to conduct these unit tests because I have built a lot of
custom transformers ,and I would like to make sure that I have implemented
them correctly. Running these tests will allow me to make sure that the
final transformed training data that I feed to my models is correct.

"""
# Importing all necessary libraries
import unittest
import pandas as pd
import numpy as np
from Transformations import Selector, Remove, ReplaceMissing, TransformNum,GarageDec, CatYear


# Creating the Class
class TestTransformers(unittest.TestCase):

    def test_selector(self):
        # Initializing basic things
        test_data = pd.DataFrame({
            'Name': ['Jinal', 'Juan', 'Joe', 'John', 'Sarah', 'Mike', 'Rachel'],
            'Age': [18, 21, 24, 25, 30, 19, 25],
            'Height': [6.5, 5.5, 5.25, 6, 5.9, 5, 6.8],
            'Weight': [120, 130, 140, 150, 125, 133, 145],
            'Grade': [11, 11, 11, 12, 10, 9, 11],
            'Math Level': [1, 2, 3, 4, 5, 6, 7],
            'Reading Level': [1, 2, 3, 4, 5, 6, 7],
            'Programming Ability': [1, 2, 3, 4, 5, 6, 7],
        })

        """ 
        Test One -> getting features of same data type
        Status: Passed
        """
        features = ['Age','Height','Reading Level']
        nump_data = test_data[features].values
        selector = Selector(features)
        result = selector.fit_transform(test_data)
        self.assertEqual(result.all(), nump_data.all())
        self.assertEqual(type(result), np.ndarray)
        
        """
        Test Two -> getting features of different data types
        Status: Passed
        """
        features = ['Name','Age','Height','Math Level','Programming Ability']
        nump_data = test_data[features].values
        selector = Selector(features)
        result = selector.fit_transform(test_data)
        self.assertEqual(result.all(), nump_data.all())
        self.assertEqual(type(result), np.ndarray)

        """ 
        Test Three -> throws exception when trying to get a feature not in data 
        Status: Passed
        """
        features = ['Name','Age','Job','Math Level']
        selector = Selector(features)
        results = selector.fit_transform(test_data)
        self.assertEqual(results, KeyError)

        """
        Test Four -> getting a single feature
        Status: Not Tested 
        """
        features = ['Name']
        nump_data = test_data[features].values
        selector = Selector(features)
        result = selector.fit_transform(test_data)
        self.assertEqual(result.all(), nump_data.all())
        self.assertEqual(type(result), np.ndarray)

    def test_remove(self):
        # Initializing Basic Information
        test_data = pd.DataFrame({
            'Name': ['Jinal', 'Juan', 'Joe', 'John', 'Sarah', 'Mike', 'Rachel'],
            'Age': [18, 21, 24, 25, 30, 19, 25],
            'Height': [6.5, 5.5, 5.25, 6, 5.9, 5, 6.8],
            'Weight': [120, 130, 140, 150, 125, 133, 145],
            'Grade': [11, 11, 11, 12, 10, 9, 11],
            'Math Level': [1, 2, 3, 4, 5, 6, 7],
            'Reading Level': [1, 2, 3, 4, 5, 6, 7],
            'Programming Ability': [1, 2, 3, 4, 5, 6, 7],
        })

        """
        Test One -> removing only 1 feature
        Status: Passed
        """
        features = ['Grade']
        temp = test_data.drop('Grade',axis=1)
        remove = Remove(features)
        test = remove.fit_transform(test_data)
        self.assertEqual(temp.keys().all(), test.keys().all())
        self.assertEqual(type(test), pd.DataFrame)

        """
        Test Two -> removing multiple features
        Status: Passed
        """
        features = ['Grade','Programming Ability']
        temp = test_data.drop(features,axis=1)
        remove = Remove(features)
        test = remove.fit_transform(test_data)
        self.assertEqual(temp.keys().all(), test.keys().all())
        self.assertEqual(type(test), pd.DataFrame)

        """
        Test Three -> removing a feature that doesn't exist
         Status: Passed
        """
        features = ['Job']
        remove = Remove(features)
        test = remove.fit_transform(remove)
        self.assertEqual(test, AttributeError)

        """ 
        Test Four -> removing no features
        Status: Passed
        """
        features = []
        remove = Remove(features)
        test = remove.fit_transform(test_data)
        self.assertEqual(test_data.keys().all(), test.keys().all())
        self.assertEqual(type(test), pd.DataFrame)



    def test_replaceMissing(self):
        test_data = pd.DataFrame({
            'Name': ['Jinal', 'Juan', 'Joe', np.nan, 'Sarah', 'Mike', 'Rachel'],
            'Age': [18, 21, 24, 25, np.nan, 19, 25],
            'Height': [6.5, 5.5, np.nan, 6, np.nan, 5, 6.8],
            'Weight': [120, 130, 140, 150, 125, 133, 145],
            'Grade': [11, 11, 11, 12, np.nan, 9, 11],
            'Math Level': [1, 2, 3, 4, 5, 6, 7],
            'Reading Level': [1, 2, 3, 4, np.NaN, 6, 7],
            'Programming Ability': [1, 2, 3, 4, 5, 6, 7],
        })

        """
        Test One -> Replacing all missing values
        Test Status: Passed 
        """
        missing_dict = {
            'Name':'N/A',
            'Age':np.nanmedian(test_data['Age']),
            'Height':np.nanmedian(test_data['Height']),
            'Grade':'N/A',
            'Reading Level': 'N/A'
        }
        temp = ReplaceMissing(missing_dict)
        result = temp.fit_transform(test_data)
        self.assertEqual(type(result), pd.DataFrame)
        self.assertEqual(test_data['Name'].isnull().sum(), 1)
        self.assertEqual(test_data['Age'].isnull().sum(), 1)
        self.assertEqual(result['Name'].isnull().sum(), 0)
        self.assertEqual(result['Age'].isnull().sum(), 0)
        self.assertEqual(result['Age'][4], np.nanmedian(test_data['Age']))
        self.assertEqual(result['Reading Level'][4], 'N/A')

        """  
        Test Two -> trying to replace values of keys that don't exist
        Test Status: Passed
        """
        missing_dict = {
            'Job':'N/A'
        }
        temp = ReplaceMissing(missing_dict)
        result = temp.fit_transform(test_data)
        self.assertEqual(result,KeyError)


    def test_transformNum(self):
        test_data = pd.DataFrame({
            'Name': ['Jinal', 'Juan', 'Joe', 'John', 'Sarah', 'Mike', 'Rachel'],
            'Age': [18, 21, 24, 25, 30, 19, 25],
            'Height': [6.5, 5.5, 5.25, 6, 5.9, 5, 6.8],
            'Weight': [120, 130, 140, 150, 125, 133, 145],
            'Grade': [11, 11, 11, 12, 10, 9, 11],
            'Math Level': [1, 2, 3, 4, 5, 6, 7],
            'Reading Level': [1, 2, 3, 4, 5, 6, 7],
            'Programming Ability': [1, 2, 3, 4, 5, 6, 7],
        })

        """
        Test One -> Transforming each numerical feature
        Test Status: Passed
        """
        transform = {
            'Age': 'log(x+1)',
            'Height': 'log(x)',
            'Weight': 'x ** .5'
        }
        temp = TransformNum(transform)
        result = temp.fit_transform(test_data)
        self.assertEqual(type(result), pd.DataFrame)
        self.assertEqual(result['Age'].all(), np.log(test_data['Age']+1).all())
        self.assertEqual(result['Height'].all(), np.log(test_data['Height']).all())
        self.assertEqual(result['Weight'].all(), (test_data['Weight'] ** .5).all())

        """   
        Test Two -> trying to transform a key that doesn't exist
        Test Status: Passed
        """
        transform = {
            'Job': 'log(x+1)'
        }
        temp = TransformNum(transform)
        result = temp.fit_transform(test_data)
        self.assertEqual(result, KeyError)

        """ 
        Test Three -> trying to make a transformation that isn't log(x+1), log(x), or x ** .5
        Test Status: Passed
        """
        transform = {
            'Age': 'x ** 1/3',
        }
        temp = TransformNum(transform)
        result = temp.fit_transform(test_data)
        self.assertEqual(type(result), pd.DataFrame)
        self.assertEqual(result['Age'].all(), test_data['Age'].all())


    def test_GarageDec(self):
        test_data = pd.DataFrame({
            'Name': ['Jinal', 'Juan', 'Joe', 'John', 'Sarah', 'Mike', 'Rachel'],
            'Age': [18, 21, 24, 25, 30, 19, 25],
            'Height': [6.5, 5.5, 5.25, 6, 5.9, 5, 6.8],
            'Weight': [120, 130, 140, 150, 125, 133, 145],
            'Grade': [11, 11, 11, 12, 10, 9, 11],
            'Math Level': [1, 2, 3, 4, 5, 6, 7],
            'Reading Level': [1, 2, 3, 4, 5, 6, 7],
            'Programming Ability': [1, 2, 3, 4, 5, 6, 7],
        })

        temp = GarageDec()
        return temp

    def test_CatYear(self):
        test_data = pd.DataFrame({
            'Name': ['Jinal', 'Juan', 'Joe', 'John', 'Sarah', 'Mike', 'Rachel'],
            'Age': [18, 21, 24, 25, 30, 19, 25],
            'Height': [6.5, 5.5, 5.25, 6, 5.9, 5, 6.8],
            'Weight': [120, 130, 140, 150, 125, 133, 145],
            'Grade': [11, 11, 11, 12, 10, 9, 11],
            'Math Level': [1, 2, 3, 4, 5, 6, 7],
            'Reading Level': [1, 2, 3, 4, 5, 6, 7],
            'Programming Ability': [1, 2, 3, 4, 5, 6, 7],
        })

        temp = CatYear({})
        return temp

    def test_CustomPipeline(self):
        return 0


# Runner
if __name__ == '__main__':
    unittest.main()
