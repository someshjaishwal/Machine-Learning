# -*- coding: utf-8 -*-
# artificial neural network

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

# feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_text = sc.transform(X_text)


### PART 2 - Create ANN and training it

import keras
from keras.models import Sequential
from keras.layers import Dense

# initialize ANN ( creating skeleton)
classifier = Sequential()

# adding input layer and first hidden layer to skeleton
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11 ))

# adding one more hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# adding output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# compile the ANN that is, add optimizer(SGD) and loss function
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# now, the ANN is ready for training
# Training/fitting ANN to training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)


### PART 3 - Making predictions and Evaluating the model

# predicting testset results
y_pred = classifier.predict(X_text)
y_pred = (y_pred > 0.5)

# making confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# evaluating model
accuracy = (cm[0,0]+cm[1,1])*100/(cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1])
print ("accuracy :",accuracy,"%")
