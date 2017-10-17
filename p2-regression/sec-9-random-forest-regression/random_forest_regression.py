# Random Forest Regression

###### Import libraries ######
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

###### Import the dataset ######
dataset = pd.read_csv("Position_Salaries.csv")
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

###### Fitting Random Forest Regression to the dataset ######
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state = 0) # n_estimators = the number of trees
regressor.fit(x,y);

###### Predicting a new result with Polynomial Regression ######
y_pred = regressor.predict(6.5) # Takes a x value to predict y at

###### Visualizing the Regression results (high res, smooth curve) ######
x_grid = np.arange(min(x), max(x), 0.01) # (graph low end, graph high end, step)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x,y,color = "blue")
plt.plot(x_grid, regressor.predict(x_grid), color = "black") # Here we predict values using the x_grid we created above
plt.title('Salary Predictions (Random Forest Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary ($)')
plt.show()