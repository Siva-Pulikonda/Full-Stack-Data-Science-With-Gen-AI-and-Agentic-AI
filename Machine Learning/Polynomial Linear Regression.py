import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv(r'C:\Users\sivap\OneDrive\Desktop\S\FSDS\December\Dec 23 2025\emp_sal.csv')

x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

# Linear Model
from sklearn.linear_model import LinearRegression    # Creates a linear model
lin_reg=LinearRegression()
lin_reg.fit(x,y)    # Model 1   # Creates a Straight Line Model Tries to Fit y=mx+c

# Linear Regression Visualization
plt.scatter(x,y,color='red')     # Red Dots actual Salary Data
plt.plot(x,lin_reg.predict(x),color='blue')  # Blue line is Predicted Line
plt.title('Linear Regression Graph')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
# The actual and Predicted data points are not matching because data is non linear(curved), Straight Line Cannot Fit Properly

lin_model_pred=lin_reg.predict([[6.5]])  # predicted salary for level 6.5 is too high because linear model assumes straight growth
print(lin_model_pred)
# This model is predicting hig salary 
# 3.30 lakh is between VP and CTO Salary
# Thats why we will use polynomial Linear Regression
# Linear Regression not enough to handle non linear dataset 

# Polynomial Model
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=5)      # why Degree =5, Higher degree → more flexibility
x_poly=poly_reg.fit_transform(x)   # Model 2
# Convert input x → [1, x, x², x³, x⁴, x⁵] into this x = 2 → [1, 2, 4, 8, 16, 32]
# if you have not mentioned any number in degree by default it will take 2

poly_reg.fit(x_poly,y)

lin_reg_2=LinearRegression()    # we are Still using LinearRegression, But input is transformed so model becomes non-linear
lin_reg_2.fit(x_poly,y)       # Model 3

# Linear Regression Visualization
plt.scatter(x, y, color = 'red')   # actual values 
plt.plot(x, lin_reg_2.predict(poly_reg.fit_transform(x)), color = 'blue') # curved Prediction 
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
# Now the graph show Non Linear curve, fits data better and captures non linear pattern


poly_model_pred=lin_reg_2.predict(poly_reg.fit_transform([[6.5]])) # Predicts salary for level 6.5, More realistic than linear model
print(poly_model_pred)


# Now this is showing less salary when compared to before
# Till now we have used system parameters
# Salary is reduced but the graphs are still not looking good
# Actual & Predicted Points are still far away

#So we increase degree by degree and check which degree is more accurate
# In this case Degree 5 is more accurate



# Linear Regression → like drawing a straight road
# Polynomial Regression → like drawing a curvy road

# Your data is curvy → so polynomial works better






