import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv(r"C:\Users\sivap\OneDrive\Desktop\S\FSDS\December\Dec 24 2025\emp_sal.csv")
x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

# Fitting SVR to dataset
from sklearn.svm import SVR  # Svr tries to Fit a curve within a margin (tolerance area), Instead of minimizing error exactly, it allows small errors
regressor=SVR(kernel='poly',degree=4,gamma='auto',C=4.0)
regressor.fit(x,y)
# Regressor is model & svr is algorithm 

y_pred_svr=regressor.predict([[6.5]])
print(y_pred_svr)

# Now in line 11 change different kernals and change different degrees,
# and test the accuracy 
# ex:kernal=sigmoid,degree=4 & kernel=ploy,degree=5 etc
# Try with changing gamma and C also but still the output is between 130 to 164
# regressor=SVR(kernel='poly',degree=4,gamma='auto',C=5.0) for this values we are getting 175 
# The moment when you do hyper parameter tuning you got to know your accuracies

# KNN 
from sklearn.neighbors import KNeighborsRegressor
knn_reg=KNeighborsRegressor(n_neighbors=4,weights='distance')
knn_reg.fit(x,y)
# Knn works like : "Look at nearby points and average their values"

y_pred_knn=knn_reg.predict([[6.5]])     # Average of nearby salaries around level 6.5
print(y_pred_knn)
# in line 27 use different parameter and try different accuracies
























