### fitting simple/multiple linear regression #################################
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)
y_pred = regressor.predict(X_test)

# building optimal model using backward feature elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones(shape = (50,1)).astype(int), values = X, axis = 1)
X_opt = X[:,[0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

# assumed significance level, i.e. sl_value to stay for a feautre to be 0.05
# if p_value for a feature < sl_value, it stays
# feature with highest p_value is chosen for stay check first

# remove x2 : index of x2 = 2 in X
X_opt = X[:,[0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

# remove x1 : index of x1 = 1 in X
X_opt = X[:,[0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

# remove x2 : index of x2 = 4 in X
X_opt = X[:,[0, 3, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

# remove x2 : index of x2 = 5 in X
X_opt = X[:,[0, 3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

''''
 conclusion of backward elimination
 feature X[:,3] is a statistically most significant feature 
 to determine dependent variable
'''

### fitting Polynomial Regression to dataset #################################
from sklearn.preprocessing import PolynomialFeatures
poly_feat = PolynomialFeatures(degree=2)
X_poly = poly_feat.fit_transform(X)

pregressor = LinearRegression()
pregressor.fit(X_poly,y)

# visualize the polynomial regression model
# getting more fine curve
X_grid = np.arange(min(X),max(X),step = 0.1)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X, y, color='red')
plt.plot(X_grid, pregressor.predict(poly_feat.fit_transform(X_grid)), color='blue')
plt.title('salary vs level of job')
plt.xlabel('level')
plt.ylabel('salary')
# prediction for level = 6.5
plt.scatter(6.5,pregressor.predict(poly_feat.fit_transform(6.5)),color='green')
plt.show()



### fitting support vector regression to dataset #################################
from sklearn.svm import SVR
regressor = SVR(kernel = "rbf")
regressor.fit(X,y)
ex = scale_X.transform(np.array([[6.5]]))
y_pred = scale_y.inverse_transform(regressor.predict(ex))

### fitting Decision Tree Regression to dataset #################################
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X,y)
y_pred = regressor.predict(6.5)

### fitting Random Forest Regression to dataset #################################
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor.fit(X,y)
y_pred = regressor.predict(6.5)
