######################### Grid Search ##################
# tune hyperparameters to improve model performance

from sklearn.grid_search import GridSearchCV  
# from sklearn.model_selection import GridSearchCV

params = [ {'C' : [1, 10, 100, 1000],
	    'kernel' : ['rbf', 'sigmoid'], 
	    'gamma' : [3.8,3.9,4,4.1,4.2,4.3] },

           {'C' : [1, 10, 100, 1000] ,
	    'kernel' : ['linear']}]

grid = GridSearchCV(estimator = classifier,
                    param_grid = params,
                    scoring = 'accuracy',
                    cv = 10,
                    n_jobs = -1)

grid = grid.fit(X_train, y_train)
best_accuracy = grid.best_score_
best_params = grid.best_params_

# see also grid.best_estimator_
