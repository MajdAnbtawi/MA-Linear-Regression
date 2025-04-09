import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the datasets
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

X= np.concatenate((train_data.iloc[:, :-1].values, test_data.iloc[:,:].values), axis=0)
X = X[:, 1:]
y= train_data.iloc[:, -1].values

"""""
X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:,:].values
"""


# Find null values in the dataset
print(train_data.isnull().sum())
print(test_data.isnull().sum())

# Filling missing values with most_frequent / mean

from sklearn.impute import SimpleImputer
simpleimputer=SimpleImputer(missing_values=np.nan, strategy='most_frequent')
simpleimputer.fit(X[:, 0:5])
X[:, 0:5] = simpleimputer.transform(X[:, 0:5])

from sklearn.impute import SimpleImputer
simpleimputer=SimpleImputer(missing_values=np.nan, strategy='most_frequent')
simpleimputer.fit(X[:, 10:11])
X[:, 10:11] = simpleimputer.transform(X[:, 10:11])

from sklearn.impute import SimpleImputer
simpleimputer=SimpleImputer(missing_values=np.nan, strategy='mean')
simpleimputer.fit(X[:, 5:10])
X[:, 5:10] = (simpleimputer.transform(X[:, 5:10])).round(0)


# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
columntransformer=ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0, 1, 2, 3, 4, 9,10])], remainder='passthrough')
X=np.array(columntransformer.fit_transform(X))

# Encoding the Dependent Variable
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X[:len(y)], y, test_size=0.3, random_state=0)

# Logistic Regression
# Feature Scaling
from sklearn.preprocessing import StandardScaler
standardscaler=StandardScaler()
X_train = standardscaler.fit_transform(X_train)
X_test = standardscaler.transform(X_test)

# Training the Logistic Regression model on the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# calculating accuracy
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Visualizing the results using a confusion matrix heatmap

import seaborn as sns
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labelencoder_y.classes_, yticklabels=labelencoder_y.classes_)
plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

