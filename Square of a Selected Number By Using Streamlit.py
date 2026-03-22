# 1.Import Streamlit
import streamlit as st
import numpy as np
import pandas as pd

# 2. Add a title to your app
st.title("My first streamlit app created by siva pulikonda")

# 3. Add some text
st.write("This app calculates square of a number")

# 4. Create a interactive slider
st.header("Select a Number")
number=st.slider("Pick a number",0,100,25)   # Min,Max,Default

# 5. Calculate and display the result
st.subheader("Result")
squared_number=number*number
st.write(f"The square of **{number}** is **{squared_number}**.")