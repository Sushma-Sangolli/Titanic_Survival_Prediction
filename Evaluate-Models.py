import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# Load the saved encoded test data
X_test, y_test = joblib.load("X_test_encoded.pkl")

# Load trained models
rf_model = joblib.load("random_forest_model.pkl")
xgb_model = joblib.load("xgboost_model.pkl")
dt_model = joblib.load("decision_tree_model.pkl")

# Make predictions
rf_pred = rf_model.predict(X_test)
xgb_pred = xgb_model.predict(X_test)
dt_pred = dt_model.predict(X_test)

# Evaluate models
models = {
    "Random Forest": rf_pred,
    "XGBoost": xgb_pred,
    "Decision Tree": dt_pred
}

for name, preds in models.items():
    print(f"\n📊 {name} - Classification Report:")
    print(classification_report(y_test, preds))

# Confusion Matrices
plt.figure(figsize=(12, 4))
for i, (name, preds) in enumerate(models.items(), 1):
    plt.subplot(1, 3, i)
    sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt='d', cmap='Blues')
    plt.title(f"{name} - Confusion Matrix")
plt.show()

# ROC Curve
plt.figure(figsize=(8, 6))
for name, preds in models.items():
    fpr, tpr, _ = roc_curve(y_test, preds)
    plt.plot(fpr, tpr, label=name)

plt.plot([0, 1], [0, 1], 'k--')  # Random chance line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

print("✅ Model evaluation completed!")
