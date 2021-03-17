"""
@author: Jinal Shah

This script will be my
submission script
"""
# Importing some libraries
import pandas as pd
from catboost import CatBoostRegressor
import numpy as np

# Getting the submission set
raw_sub = pd.read_csv('/Users/jinalshah/Jinal/Github Repos/House-Prices-Challenge-Solution'+
                      '/Data/Raw-Data/test.csv')
ids = raw_sub['Id']
sub_prep = pd.read_csv('/Users/jinalshah/Jinal/Github Repos/House-Prices-Challenge-Solution'+
                      '/Data/Prepared Data/prepared-submission-data.csv')
# Loading the Model
model = CatBoostRegressor()
model.load_model('/Users/jinalshah/Jinal/Github Repos/House-Prices-Challenge-Solution'+
                      '/Models/tuned_catboost1.cbm')

# Making predictions
pred = model.predict(sub_prep)

# Building a dataframe
final_sub = pd.DataFrame()
final_sub['Id'] = ids
final_sub['SalePrice'] = np.expm1(pred) # I need to raise the predictions to e bc I performed log on them

# Putting Submission into a CSV file
final_sub.to_csv(path_or_buf='/Users/jinalshah/Jinal/Github Repos/House-Prices-Challenge-Solution'+
                      '/Submissions/submission10.csv', index=False)