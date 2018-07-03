# kmeans clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('../dataset/Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values

# figuring out optimal number of clusters by Elbow Method
from sklearn.cluster import KMeans
WCSS = [] # within-cluster sums of squares 
for k in range(1,11): # want to optimum k within 1 to 11
    kmeans = KMeans(n_clusters = k, init = 'k-means++' , n_init = 10 , 
                      max_iter = 300, random_state = 0)
    kmeans.fit(X)
    WCSS.append(kmeans.inertia_)
    
plt.plot(range(1,11),WCSS)
plt.xlabel('#clusters')
plt.ylabel('WCSS')
plt.title('Elbow Method')
plt.show()

# we choose k = 5 from elbow method (view graph)

# applying kmeans to mall dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++' , n_init = 10 , max_iter = 300,
                random_state = 0)
y_pred = kmeans.fit_predict(X)

# visualizing the clusters
'''
finalX = np.append(X,np.reshape(y_pred,(200,1)),1)
a = y_pred==0
b = X[a,0]
'''

plt.scatter(X[y_pred==0,0],X[y_pred==0,1],s=100 ,c='red',label = 'Miser')
plt.scatter(X[y_pred==1,0],X[y_pred==1,1],s=100 ,c='green',label = 'Balanced')
plt.scatter(X[y_pred==2,0],X[y_pred==2,1],s=100 ,c='cyan',label = 'Spendthrift')
plt.scatter(X[y_pred==3,0],X[y_pred==3,1],s=100 ,c='magenta',label = 'Careless')
plt.scatter(X[y_pred==4,0],X[y_pred==4,1],s=100 ,c='blue',label = 'Sensible')
plt.title('KMeans Clustering')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
