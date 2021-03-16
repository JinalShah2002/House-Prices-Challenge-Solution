"""
@author: Jinal Shah

This script will prepare the
training data and save it as
a csv file. This will allow
me to continously use it
without having to repeatedly
prepared it

"""
# Importing needed libraries
import pandas as pd
from DataPrepPipeline import Preparation
from sklearn.model_selection import train_test_split

# Getting the raw data & submission data
raw_data = pd.read_csv('/Users/jinalshah/Jinal/Github Repos/House-Prices-Challenge-Solution/Data/' +
                       'Raw-Data/train.csv')
sub_raw = pd.read_csv('/Users/jinalshah/Jinal/Github Repos/House-Prices-Challenge-Solution/Data/' +
                       'Raw-Data/test.csv')

# Splitting data into training & testing
training, testing = train_test_split(raw_data, test_size=0.2, shuffle=True, random_state=0)

# Putting the training data through the pipeline
pipeline = Preparation(data=training ,train=True, submission=False)
train_prep = pipeline.transform()

# Putting the testing data through the pipeline
pipeline.change_data_test(data=testing)
test_prep = pipeline.transform()

# Putting the submission data through the pipeline
pipeline.change_data_sub(data=sub_raw)
sub_prep = pipeline.transform()

# Saving files to csv files
train_prep.to_csv('/Users/jinalshah/Jinal/Github Repos/House-Prices-Challenge-Solution/Data/' +
               'Prepared Data/prepared-training-data.csv')
test_prep.to_csv('/Users/jinalshah/Jinal/Github Repos/House-Prices-Challenge-Solution/Data/' +
               'Prepared Data/prepared-testing-data.csv')
sub_prep.to_csv('/Users/jinalshah/Jinal/Github Repos/House-Prices-Challenge-Solution/Data/' +
               'Prepared Data/prepared-submission-data.csv')