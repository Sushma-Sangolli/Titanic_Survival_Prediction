import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib  # To save the model

# Load Titanic dataset (Ensure the file exists in the directory)
data = pd.read_csv("titanic_data.csv")  # Adjust filename if necessary

# Select relevant features and handle missing values
data = data[["Pclass", "Sex", "Age", "Fare", "Survived"]].dropna()

# Convert categorical 'Sex' column to numerical
data["Sex"] = data["Sex"].map({"male": 1, "female": 0})

# Define features (X) and target (y)
X = data[["Pclass", "Sex", "Age", "Fare"]]
y = data["Survived"]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree Model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Save trained model
joblib.dump(dt_model, "decision_tree_model.pkl")
print("✅ Decision Tree Model Trained and Saved Successfully!")
