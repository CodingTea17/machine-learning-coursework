###### Decision Tree Regression ######

###### Import libraries ######
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

###### Import the dataset ######
dataset = pd.read_csv("Position_Salaries.csv")
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

###### Split the data into train/test ######
'''
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)
'''

###### Fitting Decision Tree Regression to the dataset ######
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(x,y)

###### Predicting a new result ######
y_pred = regressor.predict(6.5)

###### Visualizing the Regression results (high res, smooth curve) ######
x_grid = np.arange(min(x), max(x), 0.001) # (graph low end, graph high end, step)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x,y,color = "blue")
plt.plot(x_grid, regressor.predict(x_grid), color = "black") # Here we predict values using the x_grid we created above
plt.title('Salary Predictions (Decision Tree Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary ($)')
plt.show()
