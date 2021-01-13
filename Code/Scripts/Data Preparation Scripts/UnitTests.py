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
        Test One
        Status: Passed
        """
        features = ['Age','Height','Reading Level']
        nump_data = test_data[features].values
        selector = Selector(features)
        result = selector.fit_transform(test_data)
        self.assertEqual(result.all(),nump_data.all())
        self.assertEqual(type(result),type(nump_data))
        self.assertEqual(type(result),np.ndarray)
        
        """
        Test Two
        Status: Passed
        """
        features = ['Name','Age','Height','Math Level','Programming Ability']
        nump_data = test_data[features].values
        selector = Selector(features)
        result = selector.fit_transform(test_data)
        self.assertEqual(result.all(),nump_data.all())
        self.assertEqual(type(result), type(nump_data))
        self.assertEqual(type(result), np.ndarray)


    def test_remove(self):
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

        remove = Remove([])
        return remove

    def test_replaceMissing(self):
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

        temp = ReplaceMissing({})
        return temp

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

        temp = TransformNum({})
        return temp

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


# Runner
if __name__ == '__main__':
    unittest.main()

