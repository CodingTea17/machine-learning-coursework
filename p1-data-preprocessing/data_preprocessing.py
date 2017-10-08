# Data Preprocessing Template

# Importing the libraries
import numpy as np # A library of mathmetical tools!
import matplotlib.pyplot as plt # A library for plotting data!
import pandas as pd # A library for importing and managing datasets!

# Import a dataset (DON'T FORGET TO SET THE W.D.)
dataset = pd.read_csv('Data.csv')

# Set X to the matrix of values from the data
x = dataset.iloc[:, :-1].values 
y = dataset.iloc[:, -1].values # OR index of 3

# Taking care of missing data by filling missing data with the mean values of the columns
from sklearn.preprocessing import Imputer
# Command + i to see things about Class/Method/etc?
# Defaults: Imputer(missing_values="NaN", strategy="mean", axis=0, verbose=0, copy=True)
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

# Encode the categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
# Fits the labelencoder_X to the first column "country" of our matrix 1 and it returns the first column encoded
x[:, 0] = labelencoder_X.fit_transform(x[:, 0]) # 3 different countries = [0,1,2]

# Defaults: OneHotEncoder(n_values="auto", categorical_features="all", dtype=np.float64, sparse=True, handle_unknown='error')
onehotencoder = OneHotEncoder(categorical_features = [0])
x = onehotencoder.fit_transform(x).toarray()

labelencoder_Y = LabelEncoder()
y = labelencoder_Y.fit_transform(y)

# Splits the datasets into the Training and Testing Data
from sklearn.cross_validation import train_test_split
# Options: train_test_split(*arrays, **options) **options = test_size, train_size, random_state, stratify
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)