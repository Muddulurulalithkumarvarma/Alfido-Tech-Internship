# Iris Flower Classification

## Objective
To build and compare supervised machine learning models for classifying Iris flowers into
Setosa, Versicolor, and Virginica using sepal and petal measurements.

---

## Dataset
- Name: Iris Dataset
- Source: scikit-learn built-in dataset
- Features Used:
  1. Sepal Length
  2. Sepal Width
  3. Petal Length
  4. Petal Width
- Target: Iris Species

---

## Tasks Performed
- Data loading and preprocessing
- Exploratory Data Analysis (EDA)
- Data visualization for class separability
- Model training and evaluation
- Model comparison
- Saving the best-performing model
- Performing inference on new data

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

## Files in This Folder
- iris_classification.ipynb – Jupyter Notebook with complete workflow
- best_iris_model.pkl – Saved trained model
- README.md – Project documentation

---

## Model Inference Example

```python
import joblib

# Load the saved model
model = joblib.load("best_iris_model.pkl")

# Sample input: [sepal length, sepal width, petal length, petal width]
sample = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(sample)

species = ["Setosa", "Versicolor", "Virginica"]
print("Predicted Iris Species:", species[prediction[0]])
```

---

## Tools & Technologies
- Python
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Joblib

---

## Result
The trained machine learning model accurately classifies Iris flower species and can be reused
for future predictions without retraining.

---

## Internship
This project is completed as part of the Alfido Tech Internship Machine Learning tasks.
