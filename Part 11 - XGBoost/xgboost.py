# -*- coding: utf-8 -*-
# extreme gradient boosting 

# basic libararies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:,3:13].values
y = dataset.iloc[:,13].values

### PART 1 - Preprocessing Dataset

# encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_x_1 = LabelEncoder()
X[:,1]= label_x_1.fit_transform(X[:,1])
label_x_2 = LabelEncoder()
X[:,2] = label_x_2.fit_transform(X[:,2])

ohen = OneHotEncoder(categorical_features = [1])
X = ohen.fit_transform(X).toarray()
X = X[:,1:]

# splitting training and test set
from sklearn.cross_validation import train_test_split
X_train, X_text, y_train, y_test = train_test_split(X, y, test_size = 0.2,
                                                        random_state = 42)
"""
# no need of feature scaling in xtreme gradient boosting
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_text = sc.transform(X_text)
"""
### PART 2 - fitting xgboost on training set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train,y_train)

### PART 3 - Making predictions and Evaluating the model

# predicting testset results
y_pred = classifier.predict(X_text)

# making confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# evaluating model
accuracy = (cm[0,0]+cm[1,1])*100/(cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1])
print ("accuracy :",accuracy,"%")

# evaluation using k-fold cross validation
from sklearn.cross_validation import cross_val_score
accuracy_vec = cross_val_score(estimator = classifier,
                               X = X_train, y = y_train, cv = 10)
final_accurcay = accuracy_vec.mean()
std_deviation = accuracy_vec.std()
