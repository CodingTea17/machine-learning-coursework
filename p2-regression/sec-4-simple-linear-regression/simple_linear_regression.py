# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
df_salary = pd.read_csv('Salary_Data.csv')
# 'x' is a matrix of features
x = df_salary.iloc[:, :-1].values #removes last coulm
y = df_salary.iloc[:, 1].values #index of dependent var

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
my_regressor = LinearRegression() # An instance of the LinearRegression class we imported
# QUESTION: Why do we fit?
# ANSWER: It "learns" the linear fit of the dataframe...
my_regressor.fit(x_train, y_train)

# "Predicting" the Test set results
y_prediction = my_regressor.predict(x_test) # Uses the test set to "predict" the results based on the model the 'my_regressor' was fit to

# Visualize the Training set results
plt.scatter(x_train, y_train, color="yellow") # Uses the matplotlib to make a scatter plot of our training data
plt.plot(x_train, my_regressor.predict(x_train), color="green") # Plots a line using the values our model returns from it's prediction
plt.title("Salary vs Experience (Training set)")
plt.xlabel("Yrs of Exp.")
plt.ylabel("Salary")
plt.show()

# Visualize the Training set results
plt.scatter(x_test, y_test, color="black") # Uses the matplotlib to make a scatter plot of our training data
plt.plot(x_train, my_regressor.predict(x_train), color="green") # Plots a line using the values our model returns from it's prediction
plt.title("Salary vs Experience (Test set)")
plt.xlabel("Yrs of Exp.")
plt.ylabel("Salary")
plt.show()
## Importing the dataset
#dataset = pd.read_csv('Salary_Data.csv')
#X = dataset.iloc[:, :-1].values
#y = dataset.iloc[:, 1].values
#
## Splitting the dataset into the Training set and Test set
#from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
#
## Feature Scaling
#"""from sklearn.preprocessing import StandardScaler
#sc_X = StandardScaler()
#X_train = sc_X.fit_transform(X_train)
#X_test = sc_X.transform(X_test)
#sc_y = StandardScaler()
#y_train = sc_y.fit_transform(y_train)"""
#
## Fitting Simple Linear Regression to the Training set
#from sklearn.linear_model import LinearRegression
#regressor = LinearRegression()
#regressor.fit(X_train, y_train)
#
## Predicting the Test set results
#y_pred = regressor.predict(X_test)
#
## Visualising the Training set results
#plt.scatter(X_train, y_train, color = 'red')
#plt.plot(X_train, regressor.predict(X_train), color = 'blue')
#plt.title('Salary vs Experience (Training set)')
#plt.xlabel('Years of Experience')
#plt.ylabel('Salary')
#plt.show()
#
## Visualising the Test set results
#plt.scatter(X_test, y_test, color = 'red')
#plt.plot(X_train, regressor.predict(X_train), color = 'blue')
#plt.title('Salary vs Experience (Test set)')
#plt.xlabel('Years of Experience')
#plt.ylabel('Salary')
#plt.show()