# -*- coding: utf-8 -*-
"""random forest regression.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Uf4Lai7ZIoYBSZqXEmQdjN51VolZtnem

# Random Forest Regression

## Importing the libraries
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

"""## Importing the dataset"""

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y= dataset.iloc[:, -1].values

"""## Training the Random Forest Regression model on the whole dataset"""

from sklearn.ensemble import RandomForestRegressor
regressor =  RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(x,y)

"""## Predicting a new result"""

regressor.predict([[6.5]])

"""### Visualising the Random Forest Regression results (higher resolution)"""

x_grid = np.arange(min(x), max(x), 0.01)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, regressor.predict(x_grid), color = 'blue')
plt.title('Truth or Bluff (Support Vector Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()