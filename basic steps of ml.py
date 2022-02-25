# -*- coding: utf-8 -*-
"""Basic steps of ML.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1u9YheTHgQd8FUFMtfLrtVF3fI9uRzr4p

##Step 1 Import Libs
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

"""##Step 2 Import Dataset"""

dataset = pd.read_csv('Data.csv')
x = dataset.iloc[ :, :-1].values
y = dataset.iloc[:, -1].values

"""## Encoding the State(Independent)// Dummy variables

"""

from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[3])], remainder='passthrough')
x = np.array(ct.fit_transform(x))
print(x)

"""## Step 3 Splitting the dataset into the Training set and Test set

"""

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

"""### Encoding the Dependent Variable (For binary Info)

"""

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
print(y)