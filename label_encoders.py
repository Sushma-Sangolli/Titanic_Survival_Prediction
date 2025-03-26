import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the saved test data
X_test, y_test = joblib.load("test_data.pkl")

# Encode categorical columns in X_test
label_encoders = {}

for col in X_test.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X_test[col] = le.fit_transform(X_test[col])  # Fit & transform on X_test itself
    label_encoders[col] = le  # Store encoders

# Save the label encoders for consistency
joblib.dump(label_encoders, "label_encoders.pkl")

# Save the transformed X_test
joblib.dump((X_test, y_test), "X_test_encoded.pkl")

print("✅ Encoding completed! X_test is now numerical and saved.")
