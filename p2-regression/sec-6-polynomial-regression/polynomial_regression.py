# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
# The dataset is too small to train/test

# Feature Scaling
# The LinearRegression lib does it for us..? The dependent variable was already broken up into 10 levels within the company.


# We are fitting a linreg to compare the results
# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression as LR # Ha it worked
lin_reg = LR()
lin_reg.fit(x, y) # Tada! A super simple linear regression

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures as PF
poly_reg = PF(degree=4)
# poly_reg.degree # Outputs the degree of the poly_reg object
x_poly = poly_reg.fit_transform(x)
# Now to fit poly_reg to the linear regression model
lin_reg2 = LR() 
lin_reg2.fit(x_poly, y)

# Visualising the Linear Regression results
plt.scatter(x, y, color='blue') # Plotting the real observations
plt.plot(x, lin_reg.predict(x), color='black') # Plotting the lin_reg predicitions
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Ulgy plot 
# Visualising the Polynomial Regression results
#plt.scatter(x, y, color='blue')
#plt.plot(x, lin_reg2.predict(poly_reg.fit_transform(x)), color='black')
#plt.title('Truth or Bluff (Polynomial Regresson)')
#plt.xlabel('Position Level')
#plt.ylabel('Salary')
#plt.show()

# Pretty plot
# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
x_grid = np.arange(min(x), max(x), 0.1) #  Returns a vector
x_grid = x_grid.reshape((len(x_grid), 1)) # Returns a matrix
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, lin_reg2.predict(poly_reg.fit_transform(x_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
#
# Predicting a new result with Linear Regression
print("Salary Prediction for a 6.5 level employee (Using a Linear Regression): $" + "{0:.2f}".format(lin_reg.predict(6.5)[0]))

# Predicting a new result with Polynomial Regression. The final conclusion!
print("Salary Prediction for a 6.5 level employee (Using a Linear Regression): $" + "{0:.2f}".format(lin_reg2.predict(poly_reg.fit_transform(6.5))[0]))