# model selection via hyperparam tuning in Kernel SVM Classification 

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Fitting classifier to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train,y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Applying k-Fold Cross Validation
from sklearn.cross_validation import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
final_accuracry  = accuracies.mean()
accuracies.std()


# parameter tuning via grid search to get best model for our problem
from sklearn.grid_search import GridSearchCV  # from sklearn.model_selection import GridSearchCV
params = [ {'C' : [1, 10, 100, 1000] , 'kernel' : ['rbf', 'sigmoid'] , 'gamma' : [3.8,3.9,4,4.1,4.2,4.3] },
           {'C' : [1, 10, 100, 1000] , 'kernel' : ['linear']}
          ]
grid = GridSearchCV(estimator = classifier,
                    param_grid = params,
                    scoring = 'accuracy',
                    cv = 10,
                    n_jobs = -1)
grid = grid.fit(X_train, y_train)
best_accuracy = grid.best_score_
best_params = grid.best_params_

# see also grid.best_estimator_
