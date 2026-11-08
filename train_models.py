
# ==========================================
# MACHINE LEARNING ASSIGNMENT 2
# CLASSIFICATION MODEL IMPLEMENTATION
# ==========================================

# Import Logistic Regression model
from sklearn.linear_model import LogisticRegression


# Import StandardScaler for feature scaling in KNN
from sklearn.preprocessing import StandardScaler

# Import the six required evaluation metrics
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

# Import prepared data from preprocessing file
from data_preprocessing import (
    X_train_scaled,
    X_test_scaled,
    y_train,
    y_test
)

print("Libraries and prepared datasets loaded.")


# ==========================================
# MODEL 1: LOGISTIC REGRESSION
# ==========================================

# Create the Logistic Regression model
logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

# Train the model using training features and labels
logistic_model.fit(X_train_scaled, y_train)

# Print confirmation
print("Logistic Regression training completed.")


# ==========================================
# MODEL 1: LOGISTIC REGRESSION PREDICTIONS
# ==========================================

# Predict the insurance claim class (0 or 1)
# using the testing dataset
y_pred_logistic = logistic_model.predict(X_test_scaled)

# Predict the probability of an insurance claim
# [:, 1] selects the probability of class 1
y_prob_logistic = logistic_model.predict_proba(
    X_test_scaled
)[:, 1]

# Display the first ten predicted values
print("\nFirst 10 predicted values:")
print(y_pred_logistic[:10])

# Display the first ten actual values
print("\nFirst 10 actual values:")
print(y_test.to_numpy()[:10])


# ==========================================
# MODEL 1: LOGISTIC REGRESSION EVALUATION
# ==========================================

# Calculate accuracy
logistic_accuracy = accuracy_score(
    y_test, y_pred_logistic
)

# Calculate AUC using predicted probabilities
logistic_auc = roc_auc_score(
    y_test, y_prob_logistic
)

# Calculate precision
logistic_precision = precision_score(
    y_test, y_pred_logistic
)

# Calculate recall
logistic_recall = recall_score(
    y_test, y_pred_logistic
)

# Calculate F1 score
logistic_f1 = f1_score(
    y_test, y_pred_logistic
)

# Calculate Matthews Correlation Coefficient
logistic_mcc = matthews_corrcoef(
    y_test, y_pred_logistic
)

# Display evaluation results
print("\n===================================")
print("LOGISTIC REGRESSION PERFORMANCE")
print("===================================")

print("Accuracy :", round(logistic_accuracy, 4))
print("AUC      :", round(logistic_auc, 4))
print("Precision:", round(logistic_precision, 4))
print("Recall   :", round(logistic_recall, 4))
print("F1 Score :", round(logistic_f1, 4))
print("MCC      :", round(logistic_mcc, 4))


# ==========================================
# MODEL 1: CONFUSION MATRIX
# ==========================================

# Import confusion matrix and classification report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Calculate the confusion matrix
logistic_cm = confusion_matrix(
    y_test, y_pred_logistic
)

# Display the confusion matrix
print("\nConfusion Matrix:")
print(logistic_cm)

# Display precision, recall and F1 by class
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred_logistic
    )
)


# ==========================================
# MODEL 1: SAVE TRAINED MODEL
# ==========================================

# Import joblib to save trained models
import joblib

# Import os to create directories
import os

# Create the models folder if it does not exist
os.makedirs("models", exist_ok=True)

# Save the trained Logistic Regression model
joblib.dump(
    logistic_model,
    "models/logistic_regression.joblib"
)

# Print confirmation
print("\nLogistic Regression model saved successfully.")


# ==========================================
# MODEL 2: DECISION TREE CLASSIFIER
# ==========================================

# Import Decision Tree Classifier from scikit-learn
from sklearn.tree import DecisionTreeClassifier

# Import encoded training and testing features
# Decision Tree does not require feature scaling
from data_preprocessing import X_train_encoded, X_test_encoded

# Create Decision Tree model using Entropy
# max_depth=4 limits tree growth to reduce overfitting
decision_tree_model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=5,
    random_state=42
)

# Train Decision Tree using training features and labels
decision_tree_model.fit(X_train_encoded, y_train)

# Display confirmation
print("\nDecision Tree training completed.")


# ==========================================
# DECISION TREE: PREDICTIONS
# ==========================================

# Predict class labels for test data
y_pred_tree = decision_tree_model.predict(
    X_test_encoded
)

# Predict probabilities for class 1
# These probabilities are used to calculate AUC
y_prob_tree = decision_tree_model.predict_proba(
    X_test_encoded
)[:, 1]

# Display first ten predictions
print("\nFirst 10 Decision Tree predictions:")
print(y_pred_tree[:10])

# Display first ten actual labels
print("\nFirst 10 actual values:")
print(y_test.to_numpy()[:10])


# ==========================================
# DECISION TREE: MODEL EVALUATION
# ==========================================

# Calculate Accuracy
tree_accuracy = accuracy_score(y_test, y_pred_tree)

# Calculate ROC-AUC using class probabilities
tree_auc = roc_auc_score(y_test, y_prob_tree)

# Calculate Precision
tree_precision = precision_score(y_test, y_pred_tree)

# Calculate Recall
tree_recall = recall_score(y_test, y_pred_tree)

# Calculate F1 Score
tree_f1 = f1_score(y_test, y_pred_tree)

# Calculate Matthews Correlation Coefficient
tree_mcc = matthews_corrcoef(y_test, y_pred_tree)

# Display all evaluation metrics
print("\n===================================")
print("DECISION TREE PERFORMANCE")
print("===================================")

print("Accuracy :", round(tree_accuracy, 4))
print("AUC      :", round(tree_auc, 4))
print("Precision:", round(tree_precision, 4))
print("Recall   :", round(tree_recall, 4))
print("F1 Score :", round(tree_f1, 4))
print("MCC      :", round(tree_mcc, 4))


# ==========================================
# DECISION TREE: CONFUSION MATRIX
# ==========================================

# Calculate the confusion matrix
tree_cm = confusion_matrix(
    y_test,
    y_pred_tree
)

# Display confusion matrix
print("\nDecision Tree Confusion Matrix:")
print(tree_cm)

# Display classification report
print("\nDecision Tree Classification Report:")
print(
    classification_report(
        y_test,
        y_pred_tree
    )
)


# ==========================================
# DECISION TREE: VISUALIZATION
# ==========================================

# Import Matplotlib for plotting
import matplotlib.pyplot as plt

# Import plot_tree to visualize the Decision Tree
from sklearn.tree import plot_tree

# Create a figure for the tree
plt.figure(figsize=(20, 10))

# Plot the trained Decision Tree
plot_tree(
    decision_tree_model,
    feature_names=X_train_encoded.columns,
    class_names=["No Claim", "Claim"],
    filled=True,
    rounded=True,
    fontsize=8
)

# Add a title
plt.title("Decision Tree - Car Insurance Claim Prediction")

# Adjust the figure layout
plt.tight_layout()

# Display the tree
plt.show()

# Save the Decision Tree visualization as an image
plt.savefig("decision_tree_visualization.png", dpi=200, bbox_inches="tight")

# ==========================================
# DECISION TREE: SAVE MODEL
# ==========================================

# Save the trained Decision Tree model
joblib.dump(
    decision_tree_model,
    "models/decision_tree.joblib"
)

# Display confirmation
print("\nDecision Tree model saved successfully.")


# ==========================================
# MODEL 3: K-NEAREST NEIGHBORS (KNN)
# ==========================================

# Import KNN classifier
from sklearn.neighbors import KNeighborsClassifier

# Import cross-validation function
from sklearn.model_selection import cross_val_score

# Import Pipeline for combining scaling and KNN
from sklearn.pipeline import Pipeline

# Import NumPy for numerical operations
import numpy as np

# Create a list of K values from 1 to 15
k_values = range(1, 16)

# Create an empty list to store F1 scores
knn_f1_scores = []

# Try each K value
for k in k_values:

    # Create a pipeline that scales data before applying KNN
    knn_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k))
    ])

    # Calculate F1 scores using 5-fold cross-validation
    # Only training data is used to choose K
    scores = cross_val_score(
        knn_pipeline,
        X_train_encoded,
        y_train,
        cv=5,
        scoring="f1"
    )

    # Calculate the average F1 score
    mean_f1 = scores.mean()

    # Store the average score
    knn_f1_scores.append(mean_f1)

    # Display the current K and its score
    print("K =", k, "| Average F1 =", round(mean_f1, 4))

# Find the position of the highest F1 score
best_index = np.argmax(knn_f1_scores)

# Get the corresponding K value
best_k = list(k_values)[best_index]

# Display the selected K
print("\nBest K:", best_k)


# ==========================================
# KNN: VISUALIZE K VALUES
# ==========================================

# Create a figure
plt.figure(figsize=(9, 5))

# Plot K values against average F1 scores
plt.plot(
    list(k_values),
    knn_f1_scores,
    marker="o"
)

# Add the chart title
plt.title("KNN - Number of Neighbors vs F1 Score")

# Add labels
plt.xlabel("Number of Neighbors (K)")
plt.ylabel("Cross-Validation F1 Score")

# Display grid
plt.grid(True)

# Save the figure
plt.savefig(
    "knn_k_selection.png",
    dpi=150,
    bbox_inches="tight"
)

# Display the chart
plt.show()


# ==========================================
# KNN: TRAIN FINAL MODEL
# ==========================================

# Create final KNN pipeline using the best K
knn_model = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=best_k))
])

# Train the pipeline using training data
knn_model.fit(X_train_encoded, y_train)

# Display confirmation
print("\nKNN training completed.")

# Predict the test class labels
y_pred_knn = knn_model.predict(X_test_encoded)

# Predict probabilities for class 1
y_prob_knn = knn_model.predict_proba(
    X_test_encoded
)[:, 1]

# Display the first ten predictions
print("\nFirst 10 KNN predictions:")
print(y_pred_knn[:10])


# ==========================================
# KNN: MODEL EVALUATION
# ==========================================

# Calculate Accuracy
knn_accuracy = accuracy_score(y_test, y_pred_knn)

# Calculate AUC
knn_auc = roc_auc_score(y_test, y_prob_knn)

# Calculate Precision
knn_precision = precision_score(y_test, y_pred_knn)

# Calculate Recall
knn_recall = recall_score(y_test, y_pred_knn)

# Calculate F1 Score
knn_f1 = f1_score(y_test, y_pred_knn)

# Calculate MCC
knn_mcc = matthews_corrcoef(y_test, y_pred_knn)

# Display all metrics
print("\n===================================")
print("KNN PERFORMANCE")
print("===================================")

print("Best K   :", best_k)
print("Accuracy :", round(knn_accuracy, 4))
print("AUC      :", round(knn_auc, 4))
print("Precision:", round(knn_precision, 4))
print("Recall   :", round(knn_recall, 4))
print("F1 Score :", round(knn_f1, 4))
print("MCC      :", round(knn_mcc, 4))


# ==========================================
# KNN: CONFUSION MATRIX
# ==========================================

# Calculate the confusion matrix
knn_cm = confusion_matrix(y_test, y_pred_knn)

# Display the confusion matrix
print("\nKNN Confusion Matrix:")
print(knn_cm)

# Display the classification report
print("\nKNN Classification Report:")
print(classification_report(y_test, y_pred_knn))


# ==========================================
# KNN: SAVE TRAINED MODEL
# ==========================================

# Save the complete KNN pipeline
joblib.dump(
    knn_model,
    "models/knn.joblib"
)

# Display confirmation
print("\nKNN model saved successfully.")


# ==========================================
# MODEL 4: GAUSSIAN NAIVE BAYES CLASSIFIER
# ==========================================

# Import Gaussian Naive Bayes from scikit-learn
from sklearn.naive_bayes import GaussianNB

# Create a Gaussian Naive Bayes classifier
naive_bayes_model = GaussianNB()

# Train the model using encoded training features
# Scaling is not required for Gaussian Naive Bayes
naive_bayes_model.fit(X_train_encoded, y_train)

# Print confirmation after training
print("\nGaussian Naive Bayes training completed.")


# ==========================================
# NAIVE BAYES: PREDICTIONS
# ==========================================

# Predict insurance claim classes using test data
y_pred_nb = naive_bayes_model.predict(
    X_test_encoded
)

# Predict the probability of insurance claim class 1
y_prob_nb = naive_bayes_model.predict_proba(
    X_test_encoded
)[:, 1]

# Display the first ten predicted classes
print("\nFirst 10 Naive Bayes predictions:")
print(y_pred_nb[:10])

# Display the first ten actual classes
print("\nFirst 10 actual values:")
print(y_test.to_numpy()[:10])


# ==========================================
# NAIVE BAYES: MODEL EVALUATION
# ==========================================

# Calculate accuracy
nb_accuracy = accuracy_score(y_test, y_pred_nb)

# Calculate AUC using the predicted probabilities
nb_auc = roc_auc_score(y_test, y_prob_nb)

# Calculate precision
nb_precision = precision_score(y_test, y_pred_nb)

# Calculate recall
nb_recall = recall_score(y_test, y_pred_nb)

# Calculate F1 Score
nb_f1 = f1_score(y_test, y_pred_nb)

# Calculate Matthews Correlation Coefficient
nb_mcc = matthews_corrcoef(y_test, y_pred_nb)

# Display the six evaluation metrics
print("\n===================================")
print("NAIVE BAYES PERFORMANCE")
print("===================================")

print("Accuracy :", round(nb_accuracy, 4))
print("AUC      :", round(nb_auc, 4))
print("Precision:", round(nb_precision, 4))
print("Recall   :", round(nb_recall, 4))
print("F1 Score :", round(nb_f1, 4))
print("MCC      :", round(nb_mcc, 4))


# ==========================================
# NAIVE BAYES: CONFUSION MATRIX
# ==========================================

# Calculate the confusion matrix
nb_cm = confusion_matrix(
    y_test,
    y_pred_nb
)

# Display the confusion matrix
print("\nNaive Bayes Confusion Matrix:")
print(nb_cm)

# Display the classification report
print("\nNaive Bayes Classification Report:")
print(
    classification_report(
        y_test,
        y_pred_nb
    )
)


# ==========================================
# NAIVE BAYES: SAVE TRAINED MODEL
# ==========================================

# Save the Gaussian Naive Bayes classifier
joblib.dump(
    naive_bayes_model,
    "models/naive_bayes.joblib"
)

# Display confirmation
print("\nNaive Bayes model saved successfully.")
