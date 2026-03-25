import numpy as np        # Used for Numerical Operations

import matplotlib.pyplot as plt      # Used for data Visualizations

import pandas as pd         # used to load and manipulate the datasets
# Import Dataset
dataset=pd.read_csv(r'C:\Users\sivap\OneDrive\Desktop\S\FSDS\December\Dec 12 2025\Data.csv')  # This will read the file into pandas dataframe

x=dataset.iloc[:,:-1].values 
# In the Dataset Three Columns are x which is independant variables

y=dataset.iloc[:,3].values 
# only one dependant that is y so we seperate dependant and independant

from sklearn.impute import SimpleImputer
# Sckitlearn is a library to deal with null values
imputer=SimpleImputer()    # replaces missing values with mean

imputer=imputer.fit(x[:,1:3])    # find the missing values columns and replace them with mean
x[:,1:3] =imputer.transform(x[:,1:3]) 

from sklearn.preprocessing import LabelEncoder    # converts text to numbers zeros and ones

labelencoder_x=LabelEncoder()
x[:,0]=labelencoder_x.fit_transform(x[:,0])
# Using label encoder turn categorical to numerical(ex:mumbai=2)

labelencoder_y=LabelEncoder()
y=labelencoder_y.fit_transform(y)
# Label encoder turn your yes/no categorical to 0 or 1 numerical

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=0)  
# this will split the dataset into training (train the model) and testing(evaluate the model)











