import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load your uploaded data
df = pd.read_csv('titanic (1).csv')

# Simple preprocessing for the example
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Survived']].dropna()

X = df.drop('Survived', axis=1)
y = df['Survived']

# Train the model
model = RandomForestClassifier()
model.fit(X, y)

# Save the model
joblib.dump(model, 'titanic_model.joblib')
print("Model saved as titanic_model.joblib")
