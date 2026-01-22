# Titanic Survival Prediction

## Objective
To build and compare machine learning models to predict whether a passenger survived the Titanic disaster.

---

## Dataset
- Source: Kaggle Titanic Dataset (train.csv)
- Target: Survived (0 = No, 1 = Yes)

## Features Used
- Pclass
- Sex
- Age
- Fare
- Embarked_Q
- Embarked_S

---

## Models Used
- Logistic Regression
- k-Nearest Neighbors (k-NN)
- Decision Tree Classifier

The best-performing model was selected based on accuracy and saved using Joblib.

---

## Model Evaluation Metrics
- Accuracy
- Confusion Matrix
- Precision
- Recall
- F1-score

---

## Files
- titanic_survival.ipynb – Notebook with complete workflow
- best_titanic_model.pkl – Saved trained model
- README.md – Documentation

---

## Model Inference Example

```python
import joblib

model = joblib.load("best_titanic_model.pkl")
sample = [[3, 0, 22, 7.25, 0, 1]]
prediction = model.predict(sample)

print("Survived" if prediction[0] == 1 else "Did not survive")
```

---

## Tools & Technologies
- Python
- Scikit-learn
- Pandas, NumPy
- Matplotlib, Seaborn
- Joblib

---

## Internship
Completed as part of the Alfido Tech Internship Machine Learning tasks.
