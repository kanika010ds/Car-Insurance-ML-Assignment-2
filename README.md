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
| Decision Tree (Depth 5) | 0.8395 | 0.9001 | 0.7812 | 0.6778 | 0.7259 | 0.6163 |
| KNN (K=11) | 0.8145 | 0.8871 | 0.7000 | 0.7145 | 0.7072 | 0.5715 |
| Gaussian Naive Bayes | 0.7885 | 0.8692 | 0.6278 | 0.7990 | 0.7032 | 0.5521 |
| SVM | Pending | Pending | Pending | Pending | Pending | Pending |

## 7. Model Performance Observations

### Logistic Regression

The Logistic Regression model achieved an accuracy of 84.55% and an F1 score of 0.7530 on the test dataset.

The model correctly classified 1,691 out of 2,000 records. Its precision and recall were nearly equal, indicating balanced performance in identifying insurance claims.

The AUC score of 0.9158 indicates strong discrimination between the two classes.

### Decision Tree Classifier — Performance Analysis

The Decision Tree Classifier was implemented using Entropy as the splitting criterion. Two experiments were conducted with different maximum depths (`max_depth=4` and `max_depth=5`) to study how tree complexity affects classification performance.

#### Comparison of Maximum Tree Depth

| Evaluation Metric | Depth 4 | Depth 5 |
|---|---:|---:|
| Accuracy | 0.8260 | **0.8395** |
| AUC | 0.8893 | **0.9001** |
| Precision | 0.7321 | **0.7812** |
| Recall | **0.7018** | 0.6778 |
| F1 Score | 0.7166 | **0.7259** |
| MCC | 0.5914 | **0.6163** |

#### Confusion Matrix Comparison

| Confusion Matrix Component | Depth 4 | Depth 5 |
|---|---:|---:|
| True Negative (TN) | 1212 | 1254 |
| False Positive (FP) | 161 | 119 |
| False Negative (FN) | 187 | 202 |
| True Positive (TP) | 440 | 425 |

#### Observations

Increasing the maximum depth from 4 to 5 improved Accuracy, AUC, Precision, F1 Score, and MCC. The F1 Score increased from 0.7166 to 0.7259, indicating a small improvement in the balance between Precision and Recall.

However, Recall decreased from 0.7018 to 0.6778. This means that the depth-5 model missed more actual insurance claims, although it also reduced false-positive predictions.

#### Selected Configuration

For the current experiment, `max_depth=5` was retained because it achieved better F1 Score and MCC on the test dataset.

This is a provisional selection based on exploratory results. A more rigorous selection would use cross-validation on the training data.

**Current Decision Tree Result (Depth 5):**
- Accuracy: 83.95%
- AUC: 0.9001
- Precision: 0.7812
- Recall: 0.6778
- F1 Score: 0.7259
- MCC: 0.6163

### K-Nearest Neighbors (KNN)

K-Nearest Neighbors was implemented using standardized features. Different values of K from 1 to 15 were evaluated using 5-fold cross-validation on the training data.

The highest average cross-validation F1 Score was 0.7252 at K=11. Therefore, K=11 was selected for the final KNN classifier.

The model achieved an Accuracy of 81.45%, AUC of 0.8871, Precision of 0.7000, Recall of 0.7145, F1 Score of 0.7072, and MCC of 0.5715 on the test dataset.

The confusion matrix showed that the model correctly classified 1,629 of 2,000 records, including 448 actual insurance claims. It missed 179 insurance claims and incorrectly classified 192 no-claim records as claims.

Compared with Logistic Regression and the depth-5 Decision Tree, KNN achieved lower Accuracy, AUC, F1 Score and MCC. However, its Recall was higher than the Decision Tree's Recall, indicating that KNN detected a larger proportion of actual claims.

Overall, KNN provided reasonable classification performance but was not the best-performing model among the first three classifiers.

### Gaussian Naive Bayes — Performance Observation

The Gaussian Naive Bayes classifier was implemented using the encoded features of the Car Insurance Claim dataset. It achieved an Accuracy of 78.85%, AUC of 0.8692, Precision of 0.6278, Recall of 0.7990, F1 Score of 0.7032, and MCC of 0.5521.

The confusion matrix showed that the classifier correctly identified 501 actual insurance claims and missed 126 claims. However, it incorrectly classified 297 no-claim records as claims.

Among the four models evaluated so far, Gaussian Naive Bayes achieved the highest Recall but the lowest Precision and Accuracy. This indicates that the classifier is relatively effective at identifying positive insurance claim cases but produces more false-positive predictions.

Overall, Naive Bayes provides a useful baseline when detecting actual claims is important, although Logistic Regression currently offers a better balance of Precision and Recall as measured by F1 Score.

### Other Models

Performance observations for  SVM will be added after evaluation.

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
