# -*- coding: utf-8 -*-
"""Smple linear regression.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1qX3AFrl1HskLuLX8nqmfZl8vX9anDvod

##Importing libs
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

"""##importing data sets"""

dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

print(x)

print(y)

"""## Splitting the dataset into the Training set and Test set"""

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=.2,random_state=0)

print(x_train)

print(x_test)

print(y_train)

print(y_test)

"""## Training the Simple Linear Regression model on the Training set"""

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

"""## Predicting the Test set results"""

y_pred = regressor.predict(x_test)

print(y_pred)

print(y_test)

"""## Visualising the Training set results

"""

plt.scatter(x_train,y_train, Color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience(Training Set)')
plt.xlabel('Years of Experiance')
plt.ylabel('Years of Experiance')
plt.show()

"""## Visualising the Test set results

"""

plt.scatter(x_test,y_test, Color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience(Test Set)')
plt.xlabel('Years of Experiance')
plt.ylabel('Years of Experiance')
plt.show()

"""## Making a single prediction (for example the salary of an employee with 12 years of experience)"""

print(regressor.predict([[12]]))

"""Therefore, our model predicts that the salary of an employee with 12 years of experience is $ 138967,5.

**Important note:** Notice that the value of the feature (12 years) was input in a double pair of square brackets. That's because the "predict" method always expects a 2D array as the format of its inputs. And putting 12 into a double pair of square brackets makes the input exactly a 2D array. Simply put:

$12 \rightarrow \textrm{scalar}$

$[12] \rightarrow \textrm{1D array}$

$[[12]] \rightarrow \textrm{2D array}$

## Getting the final linear regression equation with the values of the coefficients
"""

print(regressor.coef_)
print(regressor.intercept_)

"""Therefore, the equation of our simple linear regression model is:

$$\textrm{Salary} = 9345.94 \times \textrm{YearsExperience} + 26816.19$$

**Important Note:** To get these coefficients we called the "coef_" and "intercept_" attributes from our regressor object. Attributes in Python are different than methods and usually return a simple value or an array of values.
"""

xn=regressor.coef_
yn=regressor.intercept_
print(xn)
print(yn)