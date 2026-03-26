import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Import libraries

dataset=pd.read_csv(r"C:\Users\sivap\OneDrive\Desktop\S\FSDS\December\Dec 17 2025\Investment.csv")
# Read the dataset 

x=dataset.iloc[:,:-1]
y=dataset.iloc[:,4]
# Seperating the attributes

X = pd.get_dummies(x,dtype=int)
# This will convert state column categorical data into integer data, Creates dummy variables (0s and 1s)
# We have already defined x while seperating attributes, so we use Capital X Here

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# Divide the data into train test split ( Used Captial X)

from sklearn.linear_model import LinearRegression   # Create Regression Model
regressor = LinearRegression()
regressor.fit(X_train,y_train)

y_pred=regressor.predict(X_test)   # Predicts profit using test data

# We Build MLR Model Now
# Currently one thing is missing in the dataset that is constant 

m=regressor.coef_
print(m)
# We Got 6 slopes, each on each attribute

c=regressor.intercept_
print(c)
# we found constant

X =np.append(arr=np.full((50,1),42467).astype(int),values=X,axis=1)
# in capital X table, constant is missing so we append constant to that whole table.

# Now We find out which attribute is best to get profit
# for that we use statsmodel

import statsmodels.api as sm      # Performs Ordinary Least Squares (OLS), Gives detailed statistics:p-values, t-values, R² score
X_opt=X[:,[0,1,2,3,4,5]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

# When X table has 7 columns why take only 6  
# Open X table if you see,in 4,5,6 columns if one columns is one and the others two columsn are zero,
# so machine will understand if a column has 1 other two columns are zeros.
# if you dont want to confuse simply take 6 attributes
# We find out which attribute is best using api & ols 
# endog means input, exog means remove 

# Now you got a regression table in output
# if you check the p value in table p value for X4 is 0.990
# but it should be 0.05, So we reject that attribute 
# so remove x4 and execute the same code again 
# in the above table why t test, why not z value, because the data is sample

import statsmodels.api as sm
X_opt=X[:,[0,1,2,3,5]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

# x4=0.940 >0.05 so we reject this attribute

import statsmodels.api as sm
X_opt=X[:,[0,1,2,3]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

# x2=0.602 >0.05 so we reject this x2 also

import statsmodels.api as sm
X_opt=X[:,[0,1]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

# This is called backward elimination(recursive feature elimination) 
# which ever column has p value > 0.05 remove it, you will get your profit attribute

dataset.columns
# X1 is more relaible if we invest in 'DigitalMarketing' we can get profit

bias=regressor.score(X_train, y_train)
bias
 
variance=regressor.score(X_test, y_test)
variance

#bias score is 0.95 so we come to conclusion this is a good model





