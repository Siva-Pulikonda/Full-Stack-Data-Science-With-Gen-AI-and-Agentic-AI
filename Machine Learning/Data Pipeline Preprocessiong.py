import numpy as np        # NumPy (np) → used for numerical operations (arrays, math)

import matplotlib.pyplot as plt      # Matplotlib (plt) → used for plotting graphs

import pandas as pd         # Pandas (pd) → used to load and handle datasets
# Import Dataset
dataset=pd.read_csv(r'C:\Users\sivap\OneDrive\Desktop\S\FSDS\December\Dec 12 2025\Data.csv')  # This will read the file into pandas dataframe
# r'' → raw string (avoids errors with backslashes \), Now dataset looks like a table (rows & columns)

x=dataset.iloc[:,:-1].values 
# In the Dataset Three Columns are x which is independant variables
# x (Independent Variables),in iloc[:,:-1], : → all rows, :-1 → all columns except last, .values → converts DataFrame to NumPy array
# These are your inputs/features

y=dataset.iloc[:,3].values 
# only one dependant that is y so we seperate dependant and independant
# y (Dependent Variable), iloc[:,3], Selects the 4th column (index 3) 
# This is your target/output

from sklearn.impute import SimpleImputer   # Import SimpleImputer Used to fill missing values
# Sckitlearn is a library to deal with null values
imputer=SimpleImputer()    # replaces missing values with mean

imputer=imputer.fit(x[:,1:3])    # find the missing values columns and replace them with mean
x[:,1:3] =imputer.transform(x[:,1:3]) 

from sklearn.preprocessing import LabelEncoder    # converts text to numbers zeros and ones

labelencoder_x=LabelEncoder()
x[:,0]=labelencoder_x.fit_transform(x[:,0])    # Instead of LabelEncoder for x[:,0], from sklearn.preprocessing import OneHotEncoder, Avoids wrong assumptions about category order
# Using label encoder turn categorical to numerical(ex:mumbai=2)

labelencoder_y=LabelEncoder()
y=labelencoder_y.fit_transform(y)
# Label encoder turn your yes/no categorical to 0 or 1 numerical

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=0)  
# this will split the dataset into training (train the model) and testing(evaluate the model)

# x_train → features for training
# x_test → features for testing
# y_train → labels for training
# y_test → labels for testing
# Ensures same split every time (reproducibility)


# 1. Load dataset
# 2. Separate inputs (X) and output (Y)
# 3. Handle missing values
# 4. Convert text → numbers
# 5. Split into training & testing sets 









