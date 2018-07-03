#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 22:40:03 2018

@author: somesh
"""
# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('../dataset/Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values

# taking care of missing data
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values="NaN", strategy="most_frequent" , axis=0)
    #fit to calc that val by strategy, stores in imp
imp.fit(X[:,1:3]) 
    #to apply those values at NaN
X[:,1:3] = imp.transform(X[:,1:3])
     
'''
X[:,-2] = "NaN"
imp.fit(X[:,1:3]) 
X[:,1:3] = imp.transform(X[:,1:3])

'''

# encoding catgorial data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
lblEncX = LabelEncoder()
X[:,0] = lblEncX.fit_transform(X[:,0])
    #encode categorial integer feature ie dummy feature
ohEnc = OneHotEncoder(categorical_features = [0])   
X = ohEnc.fit_transform(X).toarray()
    #encode dependent var
lblEncY = LabelEncoder() 
Y = lblEncY.fit_transform(Y)


# splitting dataset : training set , test set
    # okay.. test set ?? for cross-validation of model ??
from sklearn.cross_validation import train_test_split  
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state = 0)


#feature scaling 
from sklearn.preprocessing import StandardScaler
scale_X = StandardScaler()
X_train = scale_X.fit_transform(X_train)
X_test = scale_X.transform(X_test)
