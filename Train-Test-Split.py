import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

# Load your Titanic dataset
df = pd.read_csv("titanic_data.csv")  # Use your actual dataset file

# Select features and target variable
X = df[['Pclass', 'Sex', 'Age', 'Fare']]
y = df['Survived']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save test data for evaluation
joblib.dump((X_test, y_test), "test_data.pkl")

print("✅ Models trained and test data saved successfully!")
