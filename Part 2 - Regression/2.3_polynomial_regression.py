# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('../dataset/Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:, 2].values

# here, I don't want to split data

# fitting Linear Regression to dataset
from sklearn.linear_model import LinearRegression
lregressor = LinearRegression()
lregressor.fit(X,y)

# fitting Polynomial Regression to dataset
from sklearn.preprocessing import PolynomialFeatures
poly_feat = PolynomialFeatures(degree=2)
X_poly = poly_feat.fit_transform(X)

pregressor = LinearRegression()
pregressor.fit(X_poly,y)

# visualize the linear regression model
plt.scatter(X,y,color='red')
plt.plot(X, lregressor.predict(X), color='blue')
plt.title('salary vs level of job')
plt.xlabel('level')
plt.ylabel('salary')
    # prediction for level = 6.5
plt.scatter(6.5, lregressor.predict(6.5), color='green')
plt.show()

# visualize the polynomial regression model
    #getting more fine curve
X_grid = np.arange(min(X),max(X),step = 0.1)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X, y, color='red')
plt.plot(X_grid, pregressor.predict(poly_feat.fit_transform(X_grid)), color='blue')
plt.title('salary vs level of job')
plt.xlabel('level')
plt.ylabel('salary')
    # prediction for level = 6.5
plt.scatter(6.5,pregressor.predict(poly_feat.fit_transform(6.5)),color='green')
plt.show()
