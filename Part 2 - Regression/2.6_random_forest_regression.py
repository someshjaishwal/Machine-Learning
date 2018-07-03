# -*- coding: utf-8 -*-

# Random forest regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('../dataset/Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:, 2].values

# taking care of missing data

# dummy vars for categorial data

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) """

# feature scaling 
"""from sklearn.preprocessing import StandardScaler
scale_X = StandardScaler()
X_train = scale_X.fit_transform(X_train)
X_test = scale_X.transform(X_test)
"""

# fitting Regression Model to dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor.fit(X,y)

# predicting resutls for new data points
ex = 6.5
y_pred = regressor.predict(ex)

# visualize the regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X),max(X),step = 0.01)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('salary vs level of job')
plt.xlabel('level')
plt.ylabel('salary')
plt.show()


