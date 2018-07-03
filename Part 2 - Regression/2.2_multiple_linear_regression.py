#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 22:30:22 2018

@author: somesh
"""

# multiple linear regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('../dataset/50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# encoding catgorial data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
lblEncX = LabelEncoder()
X[:,3] = lblEncX.fit_transform(X[:,3])
ohEnc = OneHotEncoder(categorical_features = [3])   
X = ohEnc.fit_transform(X).toarray()

# avoiding dummy variable trap
X = X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) 

# fitting multiple linear regression to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

# prediction on test set
y_pred = regressor.predict(X_test)

# building optimal model using backward feature elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones(shape = (50,1)).astype(int), values = X, axis = 1)
X_opt = X[:,[0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

# assumed significance level, i.e. sl_value to stay for a feautre to be 0.05
# if p_value for a feature < sl_value, it stays
# feature with highest p_value is chosen for stay check first

# remove x2 : index of x2 = 2 in X
X_opt = X[:,[0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

# remove x1 : index of x1 = 1 in X
X_opt = X[:,[0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

# remove x2 : index of x2 = 4 in X
X_opt = X[:,[0, 3, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

# remove x2 : index of x2 = 5 in X
X_opt = X[:,[0, 3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

# conclusion of backward elimination
# feature X[:,3] is a statistically most significant feature 
# to determine dependent variable


 
