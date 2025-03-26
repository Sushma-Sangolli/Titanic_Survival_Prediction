import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import joblib

# Load Titanic dataset (update the path if needed)
df = pd.read_csv("titanic_data.csv")  # Ensure you have the dataset

# Select features and target variable
features = ["Pclass", "Sex", "Age", "Fare"]
target = "Survived"

# Handle missing values
df["Age"].fillna(df["Age"].median(), inplace=True)
df["Fare"].fillna(df["Fare"].median(), inplace=True)

# Encode categorical variables (Sex)
df["Sex"] = LabelEncoder().fit_transform(df["Sex"])  # Male = 1, Female = 0

# Split dataset into train and test sets
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

# Save the trained model
joblib.dump(xgb_model, "xgboost_model.pkl")

print("✅ XGBoost Model Trained and Saved Successfully!")
