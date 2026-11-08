
# ==========================================
# MACHINE LEARNING ASSIGNMENT 2
# CLASSIFICATION MODEL IMPLEMENTATION
# ==========================================

# Import Logistic Regression model
from sklearn.linear_model import LogisticRegression

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

