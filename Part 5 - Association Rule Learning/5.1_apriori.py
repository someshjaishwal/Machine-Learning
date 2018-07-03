# Apriori

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing
dataset = pd.read_csv('../dataset/Market_Basket_Optimisation.csv', header = None)
# creating list of lists for apyori.py
transactions = []
for i in range(0,7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0,20)])
 
    
# training apriori on dataset
import sys
#print(sys.path)
sys.path.append('../lib/')
from apyori import apriori

rules = apriori(transactions, min_support = 0.003, min_confidence = 0.20,
                min_lift = 3, min_length = 2) # min_length : min #items in baskets

# visualize results
results = list(rules)
