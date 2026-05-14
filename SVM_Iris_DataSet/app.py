import streamlit as st
import numpy as np
import pickle

# Load Model
model = pickle.load(open("svm_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Title
st.title("🌸 Iris Flower Prediction using SVM")

st.write("Enter Flower Measurements")

# Inputs
sepal_length = st.number_input("Sepal Length")
sepal_width = st.number_input("Sepal Width")
petal_length = st.number_input("Petal Length")
petal_width = st.number_input("Petal Width")

# Predict Button
if st.button("Predict"):

    input_data = np.array([
        [
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]
    ])

    # Scaling
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)

    species = [
        "🌼 Setosa",
        "🌺 Versicolor",
        "🌸 Virginica"
    ]

    st.success(
        "Predicted Flower: " +
        species[prediction[0]]
    )
