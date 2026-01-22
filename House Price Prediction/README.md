# House Price Prediction

## Goal
Build a regression model to predict house prices using feature engineering, missing value handling, and model selection.

## Models Used
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

## Metrics
- RMSE
- MAE
- Residual Analysis

## Files
- `house_price_prediction.ipynb` : Notebook with EDA, training, evaluation
- `house_price_model.pkl` : Saved best model
- `data.csv` : Dataset (not included)

## How to Run
1. Install dependencies:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```
2. Run the notebook.
3. Use the saved `.pkl` file for inference.

## Inference Example
```python
import joblib
model = joblib.load("house_price_model.pkl")
prediction = model.predict(sample_data)
```
