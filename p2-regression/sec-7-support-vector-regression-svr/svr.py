# SVR

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the dataset
dataset = pd.read_csv("Position_Salaries.csv")
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)

# Fitting the SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(x,y)

# Predicting a new result. First we had to inverse transform it back to its unscaled self
y_pred = sc_y.inverse_transform(regressor.predict(sc_x.transform(np.array([[6.5]]))))

# Visualizing the Regression results (high res, smooth curve)
x_grid = np.arange(min(x), max(x), 0.1) # (graph low end, graph high end, step)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x,y,color = "blue")
plt.plot(x_grid, regressor.predict(x_grid), color = "black") # Here we predict values using the x_grid we created above
plt.title('Salary Truths')
plt.xlabel('Position Level')
plt.ylabel('Salary ($)')
plt.show()
