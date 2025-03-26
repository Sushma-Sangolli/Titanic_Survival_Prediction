import joblib

# Load the model
model = joblib.load("titanic_model.pkl")

# Print the type of model
print(type(model))
