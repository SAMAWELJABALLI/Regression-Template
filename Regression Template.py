# Regression Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(X,y,test_size= 0.2, random_state=0)
 
# Fauture Scaling 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train) 
X_test = sc_X.fit_transform(X_test)
 
# Training the Regression model on the whole dataset (linear case )
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X,y)

# Training the Regression model on the whole dataset (Polynomial case)
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)
X_poly_pred =lin_reg_2.predict(X_poly)


# Predicting a new result with Linear Regression
print(lin_reg.predict([[6.5]]))

# Predicting a new result with Polynomial Regression
print (lin_reg_2.predict(poly_reg.fit_transform([[6.5]])))


# Visualising the Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X)+ 0.1, 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
X_grid_pred = poly_reg.fit_transform(X_grid)
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(X_grid_pred), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')    
plt.show()


