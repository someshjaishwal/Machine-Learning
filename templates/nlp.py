######################### natural language processing ########################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# loading dataset
dataset = pd.read_csv('../dataset/Restaurant_Reviews.tsv',
                      delimiter = '\t' ,quoting = 3)

# idea : create reviews vs words table or bags vs words model (feature matrix)

# requirements and prerequisites 
# clearing text (removing punctuations, numbers, lowering case,etc.)
# get the corpus

import re # re : support for regular expression
import nltk
nltk.download('stopwords') # contains list of insignificant words 
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer # for stemming purpose

corpus = []
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]', ' ',dataset['Review'][i]) # filtering letters
    review = review.lower() # lower casing
    review = review.split() # split the string into list, join signif words later
    # filtering signif words and apply stemming to 'em
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review 
              if word not in set(stopwords.words('english'))]
    # stemmed review may contain duplicate words
    # ex. dataset['Review][0]                      
    review = ' '.join(review)
    corpus.append(review)

# creating bag of words model
# reviews or bags vs words feature matrix
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values


# classification using Naive bayes

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
							 random_state = 0)

# Fitting classifier to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

