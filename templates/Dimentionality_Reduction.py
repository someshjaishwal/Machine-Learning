# method 1: Feature selection
* Back Propagation
* Forward Propagation
* Bidirectional Propagation

# method 2 :  Feature Extraction

################################# PCA Reduction ################
from sklearn.decomposition import PCA # linear dimentionality reduction
pca = PCA(n_components = None)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_


################################# LDA Reduction ################
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components = 2)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform()

################################# applying kernel_pca ################
from sklearn.decomposition import KernelPCA
kpca = KernelPCA(n_components = 2 , kernel = 'rbf')
X_train = kpca.fit_transform(X_train)
X_test = kpca.transform(X_test)
