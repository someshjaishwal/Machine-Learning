################################# making confusion matrix ################
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred


############## evaluating model via k-fold cross validation ##############
from sklearn.cross_validation import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y= y_train, cv = 10)
final_accuracy  = accuracies.mean()
std_deviation = accuracies.std()
