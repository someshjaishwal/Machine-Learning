# hierarchical clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('../dataset/Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values

# figuring optimal number of cluster via dendograms
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
# ward to minimize variance (similar to WCSS in KMeans)
plt.title('Dendogram')
plt.xlabel('clusters number')
plt.ylabel('Euclidean Distance')
plt.show()

# from dendogram , we choose k = 5

# applying hierarchical clustering in Mall dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean',
                             linkage = 'ward')
y_pred = hc.fit_predict(X)

# visualizing the clusters
plt.scatter(X[y_pred==0,0],X[y_pred==0,1],s=100 ,c='red',label = 'Miser')
plt.scatter(X[y_pred==1,0],X[y_pred==1,1],s=100 ,c='green',label = 'Balanced')
plt.scatter(X[y_pred==2,0],X[y_pred==2,1],s=100 ,c='cyan',label = 'Spendthrift')
plt.scatter(X[y_pred==3,0],X[y_pred==3,1],s=100 ,c='magenta',label = 'Careless')
plt.scatter(X[y_pred==4,0],X[y_pred==4,1],s=100 ,c='blue',label = 'Sensible')
plt.title('Hierarchical Clustering')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
