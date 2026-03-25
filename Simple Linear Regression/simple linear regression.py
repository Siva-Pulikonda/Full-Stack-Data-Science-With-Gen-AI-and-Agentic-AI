#15/12/2025 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset=pd.read_csv(r'C:\Users\sivap\OneDrive\Desktop\S\FSDS\December\Dec 15 2025\Salary_Data.csv')
#Reading the Dataset

x=dataset.iloc[:,:-1]   #Independant variable
y=dataset.iloc[:,-1]    #Dependant variable


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
#Split the data 


regressor=LinearRegression()   #Creating object
regressor.fit(x_train,y_train)

# regressor is model
# linear regression is algorithm

y_pred=regressor.predict(x_test)
# passing y_test to the regression model

plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
# Plotting Best fit line

dataset 
# in this dataset most experienced employee is 10.5 years, tomorrow he resigns and
#a new employee got hired with 12 years experience. so how to find new slope

m_slope=regressor.coef_
print(m_slope)
# finding slope

c_intercept=regressor.intercept_
print(c_intercept)

y_12=m_slope*12+c_intercept
print(y_12)
# Now model predicts new slope 

y_20=m_slope*20+c_intercept
print(y_20)
# For 20 Years experineced guy salary future prediction

bias_score=regressor.score(x_train,y_train)
print(bias_score)
#train score

variance_score=regressor.score(x_test,y_test)
print(variance_score)
#test score 

#16/12/2025
# stats integration to ml
dataset.mean()

dataset['Salary'].mean()

dataset.median()

dataset['Salary'].mode()

dataset.var()

dataset.std()

dataset['Salary'].std()

#for calculating coefficient of variance we have to import a library
from scipy.stats import variation
variation(dataset.values)
# This will give cv of entire dataframe

variation(dataset['Salary'])

#Correlation
dataset.corr()

dataset['Salary'].corr(dataset['YearsExperience'])

#Skewness
dataset.skew()

dataset['Salary'].skew()

#standard error
dataset.sem()

# z-score
# for calculating z- score we have to import a library
import scipy.stats as stats
dataset.apply(stats.zscore)
# this will give z score for entire dataframe

stats.zscore(dataset['Salary'])
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
SSR=np.sum((y_pred-y_mean)**2)
print(SSR)

#SSE
y=y[0:6]
SSE=np.sum((y-y_pred)**2)
print(SSE)

#SST
mean_total=np.mean(dataset.values)
SST=np.sum((dataset.values-mean_total)**2)
print(SST)

# R2
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


