###### Regression Template ######

###### Preprocessing Phase ######
# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

###### Import the dataset ######
dataset = pd.read_csv("DATASET_FILE.csv")
x = dataset.iloc[:, :].values
y = dataset.iloc[:, :].values

###### Split the data into train/test ######
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)

###### Feature Scaling #######
'''
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
sc_y = StandardScaler
y_train = sc_y.fit_transform(y_train)
'''

###### Fitting the Regression Model to the dataset ######
# Create the regressor here

###### Predicting a new result with Polynomial Regression ######
y_pred = regressor.predict() # Takes a x value to predict y at

###### Visualizing the Regression results ######
plt.scatter(x,y,color = "blue")
plt.plot(x, regressor.predict(x), color = "black")
plt.title('I am the title')
plt.xlabel('I am the x axis')
plt.ylabel('I axm the y axis')
plt.show()

###### Visualizing the Regression results (high res, smooth curve) ######
x_grid = np.arange(min(x), max(x), 0.1) # (graph low end, graph high end, step)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x,y,color = "blue")
plt.plot(x_grid, regressor.predict(x_grid), color = "black") # Here we predict values using the x_grid we created above
plt.title('I am the title')
plt.xlabel('I am the x axis')
plt.ylabel('I axm the y axis')
plt.show()
