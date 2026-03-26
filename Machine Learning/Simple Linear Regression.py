#15/12/2025 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv(r'C:\Users\sivap\OneDrive\Desktop\S\FSDS\December\Dec 15 2025\Salary_Data.csv')
#Reading the Dataset

x=dataset.iloc[:,:-1]   #Independant variable
y=dataset.iloc[:,-1]    #Dependant variable

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)  # random_state=0 → ensures same split every run
#Split the data 

from sklearn.linear_model import LinearRegression   # Training the Model
regressor=LinearRegression()   #Creating object
regressor.fit(x_train,y_train)
# This is where the model learns from data. fit() finds the best straight line (y = mx + c) that minimizes the error between predicted and actual salaries.
# regressor is model
# linear regression is algorithm

y_pred=regressor.predict(x_test)    # Predict salary for test data
# passing y_test to the regression model

plt.scatter(x_test, y_test, color = 'red')    # Red dots → actual test data
plt.plot(x_train, regressor.predict(x_train), color = 'blue')   # Blue line → regression line (best fit line)
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
# Plotting Best fit line Labels and displays graph

dataset 
# in this dataset most experienced employee is 10.5 years, tomorrow he resigns and
#a new employee got hired with 12 years experience. so how to find new slope

m_slope=regressor.coef_     # Slope (m) → how much salary increases per year
print(m_slope)
# finding slope

c_intercept=regressor.intercept_     #  Intercept (c) → salary when experience = 0
print(c_intercept) 
# Finds Intercept

y_12=m_slope*12+c_intercept       # Predict salary for 12 years experience
print(y_12)
# Now model predicts new slope 

y_20=m_slope*20+c_intercept     # Predict salary for 20 years experience
print(y_20)
# For 20 Years experineced guy salary future prediction

bias_score=regressor.score(x_train,y_train)   # Model Accuracy =  R² score on training data
print(bias_score)
#train score

variance_score=regressor.score(x_test,y_test)     # R² score on test data
print(variance_score)
#test score 

# If both bias and variance ~ 0.9 → good model

#16/12/2025
# stats integration to ml
dataset.mean()

dataset['Salary'].mean()   # Average values

dataset.median()      # Median → middle value

dataset['Salary'].mode()   # Mode → most frequent value

dataset.var()    #  Variance → spread of data

dataset.std()    # Std → square root of variance

dataset['Salary'].std()

#for calculating coefficient of variance we have to import a library
from scipy.stats import variation
variation(dataset.values)
# This will give cv of entire dataframe , CV = std / mean, Measures relative variability

variation(dataset['Salary'])

#Correlation
dataset.corr()    # Shows relationship between variables, Close to 1 → strong positive relation

dataset['Salary'].corr(dataset['YearsExperience'])

#Skewness
dataset.skew()   # Measures data symmetry, 0 → normal, +ve → right skew, -ve → left skew

dataset['Salary'].skew()    # Accuracy of sample mean

#standard error
dataset.sem()

# z-score
# for calculating z- score we have to import a library
import scipy.stats as stats             
dataset.apply(stats.zscore)               # Standardizes data
# this will give z score for entire dataframe

stats.zscore(dataset['Salary'])           # Mean = 0, Std = 1
# this will give z score to particular column

# Degree of freedom (not important)
a=dataset.shape[0] # will give no of rows
b=dataset.shape[1] # will give no of columns
degree_of_freedom=a-b
print(degree_of_freedom)
# will give degree of freedom to entire dataset

# ANOVA
# SSR
y_mean=np.mean(y)
# in entire dataset we dont have mean for y
# to find out mean of y variable 
SSR=np.sum((y_pred-y_mean)**2)    # Variation explained by model
print(SSR)

#SSE
y=y[0:6]
SSE=np.sum((y-y_pred)**2)     # Error between actual & predicted
print(SSE)

#SST Total Variation
mean_total=np.mean(dataset.values)
SST=np.sum((dataset.values-mean_total)**2)
print(SST)

# R2    Model performance, Close to 1 Means Better
r_square=1-SSR/SST
print(r_square)
# Result is 09.4 so this is best model

bias=regressor.score(x_train,y_train)
print(bias)

variance=regressor.score(x_train, y_train)
print(variance)

# R2 , bias , variance all values are 0.9
# so this model is best fit model

# we built a prediction model, now we create front end


# this is file handling in python, save the trained model to disk
import pickle
filename='linear_regression_model.pkl'
with open(filename,'wb') as file:    #wb = write binary as file
    pickle.dump(regressor,file)
print('Model has been pickled and saved as linear_regression_model.pkl')
    
import os
os.getcwd()    

# for a pickle file to work dataset, backend code and pickle file should be in the same folder
# now open the saved folder and run streamlit
