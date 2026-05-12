import streamlit as st
import joblib
import pandas as pd

# 1. Load the saved model
model = joblib.load('titanic_model.joblib')

st.title("🚢 Titanic Survival Predictor")
st.write("Enter passenger details to see if they would have survived.")

# 2. Create Input Fields
pclass = st.selectbox("Travel Class (1st, 2nd, 3rd)", [1, 2, 3])
sex = st.radio("Gender", ["male", "female"])
age = st.slider("Age", 0, 100, 25)
sibsp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)

# 3. Convert inputs to match model format
sex_encoded = 0 if sex == "male" else 1
features = pd.DataFrame([[pclass, sex_encoded, age, sibsp]], 
                        columns=['Pclass', 'Sex', 'Age', 'SibSp'])

# 4. Prediction Logic
if st.button("Predict"):
    prediction = model.predict(features)
    
    if prediction[0] == 1:
        st.success("The passenger likely survived! 🎉")
    else:
        st.error("The passenger likely did not survive. 😔")
