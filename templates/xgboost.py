
######################### Extreme Gradient Boosting ##############################

# Preprocessing Dataset

# encoding categorical data
# splitting training and test set
# no need of feature scaling in xtreme gradient boosting

# Fitting xgboost on training set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_text)

# Evaluate model and tweak with hyperparameters

