#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 16:56:52 2018

@author: somesh
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('../dataset/Salary_Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values

# splitting dataset : training set , test set
from sklearn.cross_validation import train_test_split  
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state = 0)

"""
#feature scaling 
from sklearn.preprocessing import StandardScaler
scale_X = StandardScaler()
X_train = scale_X.fit_transform(X_train)
X_test = scale_X.transform(X_test)
"""

# fitting simple linear regression to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

# visualize the hypothesis : regressor
plt.scatter(X_train,Y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs Experience')
plt.xlabel('yrs of experience')
plt.ylabel('salary')
plt.show()

# prediction on test set
y_pred = regressor.predict(X_test)

# visualize the test set
plt.scatter(X_test,Y_test,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.plot(X_test,regressor.predict(X_test),color='black')
plt.title('Salary vs Experience')
plt.xlabel('yrs of experience')
plt.ylabel('salary')
plt.show()
