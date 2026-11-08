# Car Insurance Claim Prediction — ML Assignment 2

**Course:** Machine Learning  
**Programme:** M.Tech (Artificial Intelligence and Machine Learning)  
**Institution:** BITS Pilani WILP

## 1. Problem Statement

The objective of this project is to predict whether a customer will make a car insurance claim based on demographic information, driving history, and vehicle-related attributes.

This is a binary classification problem, where:

- **0:** No insurance claim
- **1:** Insurance claim

Five machine learning classification algorithms are implemented and evaluated to compare their predictive performance.

## 2. Dataset Description

**Dataset Name:** Car Insurance Claim  
**Dataset Source:** [Kaggle — Car Insurance Data](https://www.kaggle.com/datasets/sagnik1511/car-insurance-data)

| Attribute | Description |
|---|---|
| Total records | 10,000 |
| Total original columns | 19 |
| Input features | 17 |
| Target variable | OUTCOME |
| Problem type | Binary Classification |
| Training samples | 8,000 |
| Testing samples | 2,000 |

The dataset contains customer demographic details, driving experience, credit scores, vehicle characteristics, speeding violations, DUIs, and past accidents.

### Target Distribution

| Class | Records | Percentage |
|---|---:|---:|
| No claim (0) | 6,867 | 68.67% |
| Claim (1) | 3,133 | 31.33% |

## 3. GitHub Repository

[Car-Insurance-ML-Assignment-2](https://github.com/kanika010ds/Car-Insurance-ML-Assignment-2)

## 4. Data Preprocessing

The following preprocessing steps were performed:

1. Removed the `ID` column and separated the target variable `OUTCOME`.
2. Split the dataset into 80% training and 20% testing using stratified sampling.
3. Handled missing values using median imputation.
4. Converted categorical variables into numerical representations using ordinal mappings, binary encoding, and one-hot encoding.
5. Applied `StandardScaler` for feature scaling.
6. Saved the original held-out test data as `test_data.csv`.

### Missing Values

| Feature | Missing Values | Method |
|---|---:|---|
| CREDIT_SCORE | 982 | Median Imputation |
| ANNUAL_MILEAGE | 957 | Median Imputation |

After preprocessing, the dataset contains 20 encoded input features.

## 5. Classification Models

The following five classifiers are included in the assignment:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Naive Bayes Classifier
5. Support Vector Machine (SVM)

Each model will be evaluated using Accuracy, AUC, Precision, Recall, F1 Score and Matthews Correlation Coefficient (MCC).

## 6. Model Performance Comparison

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8455 | 0.9158 | 0.7548 | 0.7512 | 0.7530 | 0.6406 |
| Decision Tree | Pending | Pending | Pending | Pending | Pending | Pending |
| KNN | Pending | Pending | Pending | Pending | Pending | Pending |
| Naive Bayes | Pending | Pending | Pending | Pending | Pending | Pending |
| SVM | Pending | Pending | Pending | Pending | Pending | Pending |

## 7. Model Performance Observations

### Logistic Regression

The Logistic Regression model achieved an accuracy of 84.55% and an F1 score of 0.7530 on the test dataset.

The model correctly classified 1,691 out of 2,000 records. Its precision and recall were nearly equal, indicating balanced performance in identifying insurance claims.

The AUC score of 0.9158 indicates strong discrimination between the two classes.

### Other Models

Performance observations for Decision Tree, KNN, Naive Bayes and SVM will be added after evaluation.

### Overall Winning Model

To be determined after evaluating all five classifiers.

## 8. Streamlit Web Application

An interactive Streamlit application will be developed to allow users to:

- Upload a test CSV file.
- Select a trained classification model.
- View model evaluation metrics.
- Display a confusion matrix or classification report.
- Compare the classification models.

**Live Streamlit App:** To be added after deployment.

## 9. Project Status

**Completed:**
- Dataset selection and inspection
- Data preprocessing
- Train-test split
- Logistic Regression training and evaluation

**Remaining:**
- Decision Tree, KNN, Naive Bayes and SVM
- Model comparison and observations
- Streamlit application and deployment
- Final submission PDF
