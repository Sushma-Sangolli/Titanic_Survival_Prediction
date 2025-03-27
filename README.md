# Titanic Survival Prediction - README

## 📌 Project Overview
This project predicts Titanic passenger survival using machine learning models. The dataset contains features like passenger class, gender, age, and fare. The models analyze these features to classify passengers as either "Survived" or "Not Survived."

## 🔍 Project Workflow
1. **Data Preprocessing**
   - Loaded Titanic dataset
   - Handled missing values
   - Encoded categorical variables
   - Split data into training and test sets

2. **Feature Engineering**
   - Transformed categorical features (Gender, Passenger Class)
   - Scaled numerical features (Fare, Age)

3. **Model Training**
   - Trained three models:
     - Random Forest
     - XGBoost
     - Decision Tree
   - Saved models using `joblib`

4. **Evaluation**
   - Tested models on unseen data
   - Generated classification reports for accuracy, precision, recall, and F1-score
   - Plotted Confusion Matrix and ROC Curve

## 🚀 How to Run the Project

### 1️⃣ Train & Save Models
```bash
python model.py
```

### 2️⃣ Evaluate Model Performance
```bash
python evaluate_models.py
```

## 📊 Model Evaluation Results
Each model was tested on the dataset and evaluated using classification metrics like accuracy, precision, recall, F1-score, confusion matrix, and ROC curve.

### 🔹 Classification Reports
#### Random Forest / XGBoost / Decision Tree
| Metric        | Class 0 (Not Survived) | Class 1 (Survived) |
|--------------|----------------------|-------------------|
| Precision    | 1.00                 | 1.00              |
| Recall       | 1.00                 | 1.00              |
| F1-score     | 1.00                 | 1.00              |
| Accuracy     | 1.00                 | -                 |

### 🌟 Model Insights
- **Females are more likely to survive**, aligning with the historical "women and children first" policy.
- **Males have a lower survival rate, even in first class.**
- **Age and Fare do not significantly affect male survival chances.**

### 👮 When Do Males Survive?
- **Pclass = 1 (First Class)** → Males in first class had a better survival rate.
- **Age < 15 years** → Younger boys had higher survival chances.
- **Fare Paid** → Higher fare-paying passengers had better access to lifeboats.

## 🔎 Example Test Cases

### 🔹 **Test Case 1: Survived (Female, First Class)**
```plaintext
Enter Passenger Age: 25  
Enter Passenger Class (1,2,3): 1  
Enter Gender (male/female): female  
Enter Fare Amount (or press Enter to use average): 100  
```
**Prediction:** Survived ✅

### 🔹 **Test Case 2: Not Survived (Male, Third Class)**
```plaintext
Enter Passenger Age: 30  
Enter Passenger Class (1,2,3): 3  
Enter Gender (male/female): male  
Enter Fare Amount (or press Enter to use average): 10  
```
**Prediction:** Not Survived ❌

## 🛠️ Future Improvements
- Add more diverse datasets to test generalization
- Use cross-validation to verify model robustness
- Improve feature engineering to avoid overfitting

💪 **Project Completed Successfully!** 🎉

