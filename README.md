# Deposit Lens: Bank Marketing Classification

## Problem Statement

This project predicts whether a bank client will subscribe to a term deposit using the UCI Bank Marketing dataset. Five classification models are trained on the same stratified train/test split and exposed through a Streamlit application.

## Dataset Description

The dataset is the UCI Bank Marketing `bank-full.csv` dataset. It contains 45,211 client records, 16 predictor variables, and the binary target `y` (`yes` or `no`). Predictors describe client demographics, account information, contact campaign details, and the outcome of previous campaigns. The source file is semicolon-separated and contains no missing values in this experiment.

## Links

- GitHub Repository: [https://github.com/nirmal-bits-pilani/ml_assignmnet](https://github.com/nirmal-bits-pilani/ml_assignmnet)
- Live Streamlit App: **Add the Streamlit Community Cloud URL after deployment.**

## Models Used

The split is stratified, uses 80% training and 20% testing, and has `random_state=42`. Categorical columns are label encoded and numeric values are standardized using a scaler fitted only on the training data.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8914 | 0.8726 | 0.5945 | 0.2259 | 0.3274 | 0.3205 |
| Decision Tree | 0.8995 | 0.8516 | 0.5849 | 0.4849 | 0.5302 | 0.4771 |
| kNN | 0.8923 | 0.8089 | 0.5717 | 0.3166 | 0.4075 | 0.3724 |
| Naive Bayes | 0.8380 | 0.8127 | 0.3554 | 0.4726 | 0.4057 | 0.3183 |
| Random Forest (Ensemble) | 0.9064 | 0.9250 | 0.6563 | 0.4206 | 0.5127 | 0.4777 |

## Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong AUC and accuracy, but low recall shows that its linear boundary misses many subscribers. |
| Decision Tree | Offers the best F1 score and nearly the best MCC, balancing precision and recall better than the linear model. |
| kNN | Produces reasonable accuracy but lower AUC and F1, indicating weaker ranking and minority-class detection. |
| Naive Bayes | Has relatively high recall but the lowest accuracy and precision because its independence assumption is restrictive for this data. |
| Random Forest (Ensemble) | Best AUC, accuracy, precision, and MCC; it captures nonlinear relationships and is the strongest overall general-purpose model here. |
| Overall Winner | Random Forest, based on the best AUC, accuracy, precision, and MCC; Decision Tree has the highest F1 and recall tradeoff. |

## Running the Project

```powershell
.\venv\Scripts\Activate.ps1
python solution\model\train_models.py
streamlit run solution\app.py
```

Upload `test_data.csv` in the app. It includes the target column so the application can display all six evaluation metrics and the confusion matrix.