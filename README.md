# Diabetes Prediction Using Logistic Regression

This project uses machine learning to predict whether a person has diabetes or not based on various health-related features. The dataset contains information about pregnancies, glucose levels, blood pressure, insulin, BMI, and other health factors.

## Project Overview

The goal of this project is to predict whether a person has diabetes (1) or not (0) using Logistic Regression. The model is trained using the features from the dataset, and its performance is evaluated based on accuracy, precision, recall, and F1-score.

## Dataset

The dataset used in this project is the Pima Indians Diabetes Database. It contains 768 entries, each with 8 features related to health measurements. The target variable is `diabetes`, which has two possible values:
- `1` for diabetes
- `0` for no diabetes

The dataset includes the following columns:
- `pregnancies`: Number of times the patient has been pregnant
- `glucose`: Plasma glucose concentration after 2 hours in an oral glucose tolerance test
- `diastolic`: Diastolic blood pressure (mm Hg)
- `triceps`: Triceps skinfold thickness (mm)
- `insulin`: 2-Hour serum insulin (mu U/ml)
- `bmi`: Body mass index (weight in kg / (height in m)^2)
- `dpf`: Diabetes pedigree function (a function that scores likelihood of diabetes based on family history)
- `age`: Age of the patient
- `diabetes`: Target variable indicating whether the patient has diabetes (1) or not (0)

## Libraries Used

- `pandas`: Data manipulation and analysis
- `sklearn`: Machine learning and model evaluation
- `matplotlib`: Data visualization (optional)

## Steps Involved

1. **Data Loading**: The dataset is loaded from a CSV file.
2. **Data Preprocessing**: The target variable `diabetes` is separated from the feature set.
3. **Train-Test Split**: The dataset is split into training and testing sets (90% training, 10% testing).
4. **Model Training**: A Logistic Regression model is trained on the training set.
5. **Model Evaluation**: The model's performance is evaluated using the test set, and the results are displayed using metrics like accuracy, precision, recall, and F1-score.

## Code

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Load dataset
diabetes = pd.read_csv('https://github.com/YBIFoundation/Dataset/raw/main/Diabetes.csv')

# Preprocessing
y = diabetes['diabetes']
X = diabetes.drop(['diabetes'], axis=1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, random_state=2529)

# Train model
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Confusion Matrix:")
print(conf_matrix)
print("\nAccuracy Score:", accuracy)
print("\nClassification Report:")
print(class_report)

**Results**
Accuracy: 77.92%
Precision: 74% (No Diabetes), 88% (Diabetes)
Recall: 93% (No Diabetes), 60% (Diabetes)
F1-Score: 82% (No Diabetes), 71% (Diabetes)

**Confusion Matrix**
```lua
[[39,  3]
 [14, 21]]

**Conclusion**
The Logistic Regression model performs well with an accuracy of 77.92%. The model is able to predict whether a person has diabetes or not with a good balance of precision and recall.
