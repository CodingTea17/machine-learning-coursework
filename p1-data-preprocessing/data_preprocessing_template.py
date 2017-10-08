# Data Preprocessing Template

# Importing the libraries
import numpy as np # A library of mathmetical tools!
import matplotlib.pyplot as plt # A library for plotting data!
import pandas as pd # A library for importing and managing datasets!

# Import a dataset (DON'T FORGET TO SET THE W.D.)
dataset = pd.read_csv('Data.csv')
# Set X to the matrix of values from the data
x = dataset.iloc[:, :-1].values 
y = dataset.iloc[:, 3].values # OR index of -1

# Splits the datasets into the Training and Testing Data
from sklearn.cross_validation import train_test_split
# Options: train_test_split(*arrays, **options) **options = test_size, train_size, random_state, stratify
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
"""