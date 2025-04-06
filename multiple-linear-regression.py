import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X= dataset.iloc[:, :-1].values
y= dataset.iloc[:, -1].values

# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
columntransformer=ColumnTransformer(transformers=[('encoder', OneHotEncoder(),[3])], remainder='passthrough')
X=np.array(columntransformer.fit_transform(X))

# Avoiding the Dummy Variable Trap
X=X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
linearregression=LinearRegression()
linearregression.fit(X_train, y_train)

# Predicting the Test set results
y_pred=linearregression.predict(X_test)
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)

X = np.append(arr=np.ones((50, 1)).astype(int), values=X, axis=1)

#Backward Elimination with OLS
import statsmodels.api as sm
X_opt = np.array(X[:, [0, 1,2, 3, 4, 5]],dtype=float)
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = np.array(X[:, [0,3, 4, 5]],dtype=float)
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = np.array(X[:, [0,3, 5]],dtype=float)
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

# Training a New Model with Optimized Features
X_train2, X_test2, y_train, y_test = train_test_split(X_opt, y, test_size=0.2, random_state=0)
regressor = LinearRegression()
regressor.fit(X_train2, y_train)

# Predicting the Test set results
y_pred2 = regressor.predict(X_test2)
from sklearn.metrics import r2_score
print(round(r2_score(y_test, y_pred),4))
print(round(r2_score(y_test, y_pred2),4))