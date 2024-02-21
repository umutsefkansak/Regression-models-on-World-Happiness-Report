# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 18:12:39 2024

@author: umut
"""
#%% Data
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("2019.csv")

df.head()

x = df["Healthy life expectancy"].values.reshape(-1,1)
y = df["Score"].values.reshape(-1,1)

plt.scatter(x,y)
plt.xlabel("Healthy life expectancy")
plt.ylabel("Score")
plt.show()

#%% Linear Regression
from sklearn.linear_model import LinearRegression

linear_regression = LinearRegression()
linear_regression.fit(x, y)

#%% Linear Regression - Prediction and R2 score 
from sklearn.metrics import r2_score
y_head = linear_regression.predict(x)
print("Linear Regression R2 Score: ",r2_score(y, y_head))

#%% Linear Regression Visualition
plt.scatter(x,y)
plt.plot(x,y_head,color = "red")
plt.show()


#%% Multiple Linear Regression

x = df.iloc[:,3:].values
y = df["Score"].values.reshape(-1,1)

multiple_linear_regression = LinearRegression()
multiple_linear_regression.fit(x,y)

#%%  Multiple Linear Regression - Prediction and R2 score 
import numpy as np

print("b0 : ",multiple_linear_regression.intercept_)
print("b1,b2,b3,b4,b5,b6: ",multiple_linear_regression.coef_)

multiple_linear_regression.predict(np.array([[1.542,1.412,1.200,0.601,0.255,0.621]]))

y_head = multiple_linear_regression.predict(x)

print("Multiple Linear Regression R2 Score: ",r2_score(y,y_head))



#%%Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures

x = df["Healthy life expectancy"].values.reshape(-1,1)
y = df["Score"].values.reshape(-1,1)




polynomial_regression = PolynomialFeatures(degree=2)
x_polynomial = polynomial_regression.fit_transform(x)


linear_regression2 = LinearRegression()
linear_regression2.fit(x_polynomial,y)

x_new = np.arange(min(x),max(x),0.01).reshape(-1,1)

x_new_polynomial  = polynomial_regression.transform(x_new)

#%% Polynomial Regression - Prediction Visulation
y_head = linear_regression2.predict(x_new_polynomial)

plt.scatter(x,y)
plt.plot(x_new,y_head,color="red")
plt.show()



#%% Decision Tree

x = df["Healthy life expectancy"].values.reshape(-1,1)
y = df["Score"].values.reshape(-1,1)

from sklearn.tree import DecisionTreeRegressor

decision_tree_reg = DecisionTreeRegressor()
decision_tree_reg.fit(x,y)

#%% Decision Tree - Prediction and visulation

x_ = np.arange(min(x),max(x),0.001).reshape(-1,1)

y_head = decision_tree_reg.predict(x_)

plt.scatter(x, y)
plt.plot(x_,y_head,color="red")
plt.show()

#%% Decision Tree Score
y_head_decision_tree = decision_tree_reg.predict(x)
print("Decision Tree Score : ",r2_score(y, y_head_decision_tree))

#%% Random Forest
from sklearn.ensemble import RandomForestRegressor

x = df["Healthy life expectancy"].values.reshape(-1,1)
y = df["Score"].values.reshape(-1,1)

random_forest_reg = RandomForestRegressor(n_estimators=40,random_state=42)
random_forest_reg.fit(x,y)

#%% Random Forest Prediction and Visualition 
x_ = np.arange(min(x),max(x),0.001).reshape(-1,1)
y_head = random_forest_reg.predict(x_)

plt.scatter(x, y)
plt.plot(x_,y_head,color="red")
plt.show()

#%% Random Forest Score

y_score_random_forest = random_forest_reg.predict(x)
print("Random Forest Score : ",r2_score(y, y_score_random_forest))



