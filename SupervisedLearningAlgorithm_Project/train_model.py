import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import joblib

# 1. Load dataset
df = pd.read_csv('student_performance_dataset.csv.csv')

# 2. Encode categorical variables (converting text to numbers)
le_dict = {}
df_encoded = df.copy()
categorical_cols = df.select_dtypes(include=['object']).columns

for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df[col])
    le_dict[col] = le

# 3. Split features and target
X = df_encoded.drop(columns=['G3'])
y = df_encoded['G3']

# 4. Train Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# 5. Save the "Brain" files
joblib.dump(model, 'student_model.pkl')
joblib.dump(le_dict, 'encoders.pkl')
joblib.dump(X.columns.tolist(), 'feature_names.pkl')

print("Success: Model files (pkl) have been created!")