# Multiple Linear Regression by Dawson Mortenson

# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
x[:, 3] = labelencoder_x.fit_transform(x[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
x = onehotencoder.fit_transform(x).toarray()

# Avoiding th Dummy Variable Trap
x = x[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predicts the results using the regressor model
y_pred = regressor.predict(x_test)

# Building the optimal model using Backward Elimination

#import statsmodels.formula.api as sm
# Appends a column of 1s to the matrix to account for the X(B0) value. Otherwise the model doesn't know it exists
# To add to the beginning of the matrix we put the array of ones as the parameter "arr"
x = np.append(arr = np.ones((50, 1)).astype(int), values = x, axis = 1)

#################### OLD WAY WITHOUT AWESOME METHOD ##########################
#x_opt = x[:, [i for i in range(0, 6)]] # nifty range to list codebit
## Endog = y and Exog = x. Note: exog doesn't assume a y-intercept. We had to explicity add it above
#regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
## P-values are the measure of importance.
#regressor_OLS.summary()
#x_opt = x[:, [0,1,3,4,5]]
#regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
#regressor_OLS.summary()
#x_opt = x[:, [0,3,5]]
#regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
#regressor_OLS.summary()
#x_opt = x[:, [0,3]]
#regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
#regressor_OLS.summary()
##############################################################################

# I wrote a backwards elimination method and put it in a custom module!
# Runs a while loop remove unsignificant values
from eliminations import backwards_elimination

# Takes the x/y columns, the width of the feature set, and the p signifigance value
regressor_OLS = backwards_elimination(x,y,5)

regressor_OLS.summary()