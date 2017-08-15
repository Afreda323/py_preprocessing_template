import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Indexes depend on data!

# Import the dataset
dataset = pd.read_csv('Data.csv')

# Independent Vars - First and Second Columns
X = dataset.iloc[:,:-1].values
# Dependent Var - Third Column
y = dataset.iloc[:,3].values 

# Handle missing data - Using mean for this template
from sklearn.preprocessing import Imputer
# Replace missing data with mean
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

# Handle categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# Encode the Country with dummy vars
labelEncoder_X = LabelEncoder()
X[:, 0] = labelEncoder_X.fit_transform(X[:,0])
oneHot = OneHotEncoder(categorical_features = [0])
X = oneHot.fit_transform(X).toarray()

# Encode Yes/No vars with 0/1
labelEncoder_y = LabelEncoder()
y = labelEncoder_y.fit_transform(y)

# Splitting into training and test sets
from sklearn.model_selection import train_test_split
# 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
# Scale by distance formula
sc_X = StandardScaler()
# Set newly scaled data
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)