import pandas as pd
import numpy as np

# Importing the datasets
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
train_data.head()
test_data.head()

print('\nInfo of numeric columns: ')
print(train_data.describe())

print('\nCount of null values in each column:')
print(train_data.isnull().sum())

