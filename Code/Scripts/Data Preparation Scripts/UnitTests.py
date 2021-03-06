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
from Tranformers import Remove, TransformNum


# Creating the Class
class TestTransformers(unittest.TestCase):

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
        remove = Remove.Remove(features)
        test = remove.fit_transform(test_data)
        self.assertEqual(temp.keys().all(), test.keys().all())
        self.assertEqual(type(test), pd.DataFrame)

        """
        Test Two -> removing multiple features
        Status: Passed
        """
        features = ['Grade','Programming Ability']
        temp = test_data.drop(features,axis=1)
        remove = Remove.Remove(features)
        test = remove.fit_transform(test_data)
        self.assertEqual(temp.keys().all(), test.keys().all())
        self.assertEqual(type(test), pd.DataFrame)

        """
        Test Three -> removing a feature that doesn't exist
         Status: Passed
        """
        features = ['Job']
        remove = Remove.Remove(features)
        test = remove.fit_transform(remove)
        self.assertEqual(test, AttributeError)

        """ 
        Test Four -> removing no features
        Status: Passed
        """
        features = []
        remove = Remove.Remove(features)
        test = remove.fit_transform(test_data)
        self.assertEqual(test_data.keys().all(), test.keys().all())
        self.assertEqual(type(test), pd.DataFrame)

    def test_transformNum(self):
        test_data = pd.DataFrame({
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
        temp2 = test_data
        temp = TransformNum.TransformNum(transform)
        result = temp.fit_transform(test_data)
        self.assertEqual(type(result), pd.DataFrame)
        self.assertEqual(result['Age'].all(), np.log(temp2['Age']+1).all())
        self.assertEqual(test_data['Height'].all(), np.log(temp2['Height']).all())
        self.assertEqual(result['Weight'].all(), (temp2['Weight'] ** .5).all())

        """   
        Test Two -> trying to transform a key that doesn't exist
        Test Status: Passed
        """
        transform = {
            'Job': 'log(x+1)'
        }
        temp = TransformNum.TransformNum(transform)
        result = temp.fit_transform(test_data)
        self.assertEqual(result, KeyError)

        """ 
        Test Three -> trying to make a transformation that isn't log(x+1), log(x), or x ** .5
        Test Status: Passed
        """
        transform = {
            'Age': 'x ** 1/3',
        }
        temp = TransformNum.TransformNum(transform)
        result = temp.fit_transform(test_data)
        self.assertEqual(type(result), pd.DataFrame)
        self.assertEqual(result['Age'].all(), test_data['Age'].all())


# Runner
if __name__ == '__main__':
    unittest.main()

