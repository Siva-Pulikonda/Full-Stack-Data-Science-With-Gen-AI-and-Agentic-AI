import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv(r"C:\Users\sivap\OneDrive\Desktop\S\FSDS\December\Dec 30 2025\logit classification.csv")

x=dataset.iloc[:, [2,3]].values
y=dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)
# Note: if you write x_train, y_train it will through error in next step as expected 2d array got 1d array error
# so always write x_train,x_test,y_train,y_test

  
# Feature scaling techniques are standardization and normalization
# Open x_train table, you see age is 2 digit and salary is 5 digit, salary cant be 2 digit and age cant be 5 digit 
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)
# Now if you check x_train, x_test values these are scaled between 0 to 1

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()   # in this step if you want to parameter tuning try with ctrl i
classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)
# Keep y_test and y_pred tables side by side even though no 9 did not purchase, it is predicting a purchase
# so miscalculation happening. I want to find out how many records misclassify


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)
print(cm)
# TP=65,TN=24,FP=3,FN=8

from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test, y_pred)
print(ac)
# TP+TN/Total = 65+24/100 = 0.89

from sklearn.metrics import classification_report
cr=classification_report(y_test,y_pred)
print(cr)

# with test_size=0.25 with scaling - accuracy is 89%
# with test_size=0.20 with scaling - accuracy is 92.5%

# Keep Line 18 to 21 as # or ''' to get output withput scaling
# with test_size=0.25 without scaling - accuracy is 89%
# with test_size=0.20 without scaling - accuracy is 91.25%

# Test_size=0.20 with scaling with random_state=11 is 82.5%
# Test_size=0.20 with scaling with random_state=21 is 86.25%
# Test_size=0.20 with scaling with random_state=41 is 78.75%
# Test_size=0.20 with scaling with random_state=51 is 88.75%
# Test_size=0.20 with scaling with random_state=100 is 82.5%

# Model is built what next? 
# Every DB consists of future records. if i book a ticket next week it means future record
# future records doesnot have a predicted value or dependent variable
# How to pass future records to existing model
# For Ex new datset given to you with 10 customers and without a dependent variable
# you are predicting by using logit regression that this customer will buy a car
# so i build my model on historical data, i pass the future data to the built in model, the model predicted the future
# if the customer actual comes tomorrow to buy a car you will get actual data
# This dataset is called validation set
# Question is how to pass future prediction to current model
   


# 31/12/2025
# ------------------- FUTURE PREDICTION --------------------

dataset1=pd.read_csv(r'C:\Users\sivap\OneDrive\Desktop\S\FSDS\December\Dec 31 2025\15. Logistic regression with future prediction\Future prediction1.csv')

d2=dataset1.copy()

# in training data i passed 2 attributes, now in future data if i pass 5 attributes the model will give dimension error
# so we pass only two attributes

dataset1=dataset1.iloc[:,[2,3]].values

#Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
M=sc.fit_transform(dataset1)

# i create a empty dataframe
y_pred1=pd.DataFrame()

# in that empty dataframe what is classifier? = model. What is M? = future records
# this future records i want to pass to my model

d2['y_pred1']=classifier.predict(M)

d2.to_csv('Future prediction1.csv')

# Now you got future prediction y_pred1
# in company you will write a sql query to pull the future records. That future records you need to pass to the model 
# which you build on historical data
# Ridge and lasso is all about reduce the coefficient
# scale reduce the range between z score 
# in regression the future prediction is continuous variable ex gold price, it would be a number it would be a value 
# but in this case the future prediction is yes or no


# if you do model testing 1 now all your predictions are accurate and only 1 misclassify(one customer did not purchase)
# you got 95% accuracy in model testing 1
# now if you do model testing 2 again with live data vs future data now you got 90 % accuracy 
# now if you do model testing 3 again with live data vs future data now you got 85 % accuracy
# average accuracy is >80% . so now you can deploy the code
# if i share this code does client understand this ? no. So we build graphs


from sklearn.metrics import roc_auc_score, roc_curve
y_pred_prob = classifier.predict_proba(x_test)[:, 1]

auc_score = roc_auc_score(y_test, y_pred_prob)
auc_score

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)


plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {auc_score:.2f})')
plt.plot([0,1], [0,1], 'k--')  # Random classifier line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()



# Build Streamlit if you can for this model
















