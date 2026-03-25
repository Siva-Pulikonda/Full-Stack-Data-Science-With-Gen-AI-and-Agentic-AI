import streamlit as st
import pickle
import numpy as np

# load the saved model
model=pickle.load(open(r'C:\Users\sivap\AVSCODE\MACHINE LEARNING\linear_regression_model.pkl','rb'))
# rb= read binary

# set the title of streamlit app
st.title("Salary Prediction App")

# add a brief description
st.write("This app predicts the salary based on years of experience using simple linear regression")

#add input widget for user to enter years of experience
years_experience=st.number_input("Enter Years of Experience:",min_value=0.0,max_value=50.0,value=1.0,step=0.5)

#when the button is clicked, make predictions
if st.button("Predict Salary"):                  # make a prediction using trained model
    experience_input=np.array([[years_experience]])
    prediction= model.predict(experience_input)
    
    # Display the result
    st.success(f"The predicted salary for {years_experience} years of experience is: ${prediction[0]:,.2f}")

# Display information about the model    
st.write("The Model was trained using a dataset Salaries and years of experience build by Siva Pulikonda")
