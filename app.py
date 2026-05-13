import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("employee_salary_dataset.csv")

# Encode categorical columns
le = LabelEncoder()

df["Gender"] = le.fit_transform(df["Gender"])
df["Education"] = le.fit_transform(df["Education"])
df["Department"] = le.fit_transform(df["Department"])

# Features and target
X = df.drop("Salary", axis=1)
y = df["Salary"]

# Train model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

# Title
st.title("Employee Salary Prediction System")

st.write("Enter Employee Details")

# User Inputs
employee_id = st.number_input(
    "Employee ID",
    min_value=1,
    value=1
)

age = st.slider(
    "Age",
    20,
    60,
    25
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

education = st.selectbox(
    "Education",
    ["UG", "PG", "PhD"]
)

department = st.selectbox(
    "Department",
    ["HR", "IT", "Sales", "Finance"]
)

experience = st.slider(
    "Experience",
    1,
    20,
    5
)

working_hours = st.slider(
    "Working Hours",
    30,
    60,
    40
)

performance_score = st.slider(
    "Performance Score",
    1,
    10,
    7
)

projects_completed = st.slider(
    "Projects Completed",
    1,
    20,
    5
)

# Encoding
gender_value = 1 if gender == "Male" else 0

education_dict = {
    "UG": 0,
    "PG": 1,
    "PhD": 2
}

department_dict = {
    "HR": 0,
    "IT": 1,
    "Sales": 2,
    "Finance": 3
}

education_value = education_dict[education]
department_value = department_dict[department]

# Prediction
if st.button("Predict Salary"):

    input_data = pd.DataFrame([{
        "Employee_ID": employee_id,
        "Age": age,
        "Gender": gender_value,
        "Education": education_value,
        "Department": department_value,
        "Experience": experience,
        "Working_Hours": working_hours,
        "Performance_Score": performance_score,
        "Projects_Completed": projects_completed
    }])

    prediction = model.predict(input_data)

    st.success(
        f"Predicted Salary: ₹{prediction[0]:,.2f}"
    )

# Dataset Preview
st.subheader("Dataset Preview")

st.dataframe(df.head())
