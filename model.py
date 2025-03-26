import numpy as np
import pandas as pd
import joblib  # To load saved models

# Load trained models
rf_model = joblib.load("random_forest_model.pkl")
xgb_model = joblib.load("xgboost_model.pkl")
dt_model = joblib.load("decision_tree_model.pkl")  # Decision Tree model

# Function to take user input
def get_user_input():
    age = int(input("Enter Passenger Age: "))
    pclass = int(input("Enter Passenger Class (1,2,3): "))
    gender = input("Enter Gender (male/female): ").strip().lower()
    fare = input("Enter Fare Amount (or press Enter to use average): ")
    
    # Convert gender to numerical value
    sex = 1 if gender == "male" else 0  
    
    # Default fare if not entered
    fare = float(fare) if fare else 32.2  

    # Convert input into NumPy array
    return np.array([[pclass, sex, age, fare]])

# Function to predict survival for all models
def predict_survival(user_input):
    rf_prediction = rf_model.predict(user_input)[0]
    xgb_prediction = xgb_model.predict(user_input)[0]
    dt_prediction = dt_model.predict(user_input)[0]

    return (
        "Survived" if rf_prediction == 1 else "Did Not Survive",
        "Survived" if xgb_prediction == 1 else "Did Not Survive",
        "Survived" if dt_prediction == 1 else "Did Not Survive"
    )

# Main execution
if __name__ == "__main__":
    print("\n💡 Model expects features: ['Pclass', 'Sex', 'Age', 'Fare']\n")
    
    user_input = get_user_input()  # Get input data
    rf_result, xgb_result, dt_result = predict_survival(user_input)  # Get predictions
    
    # Display results
    print("\n📊 **Survival Predictions**:")
    print(f"🔹 Random Forest Model: **{rf_result}**")
    print(f"🔹 XGBoost Model: **{xgb_result}**")
    print(f"🔹 Decision Tree Model: **{dt_result}**")
