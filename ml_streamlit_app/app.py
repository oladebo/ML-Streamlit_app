import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load("model/model.pkl")

st.title("Iris Flower Prediction Model")

# Input Features
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.0)
sepal_width = st.slider("Sepal Width (cm)", 5.0, 2.0, 3.0)
petal_length = st.slider("Petal Length(cm)", 1.0, 7.0, 4.0)
petal_width = st.slider("Petal Length (cm)", 0.1, 2.5, 1.0)

# Prepare input Data
input_df = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# predict
prediction = model.predict(input_df)
predicted_class ={0:"setosa", 1: "versicolor",2: "virginica"}[prediction[0]]

# Display the result
st.write(f"Predicted Irish species: **{predicted_class}**")