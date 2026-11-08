
# ==========================================
# ML ASSIGNMENT 2 - DATA PREPROCESSING
# Dataset: Car Insurance Claim
# ==========================================

# Import pandas for reading and analyzing data
import pandas as pd

# Import kagglehub to download the dataset
import kagglehub

# Import NumPy for numerical operations
import numpy as np

# Import os to work with file paths
import os


# ==========================================
# STEP 1: LOAD THE DATASET
# ==========================================

# Download the Car Insurance dataset from Kaggle
dataset_path = kagglehub.dataset_download(
    "sagnik1511/car-insurance-data"
)

# Create the complete path to the CSV file
csv_path = os.path.join(
    dataset_path,
    "Car_Insurance_Claim.csv"
)

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_path)

# Display the first 5 rows
print("\nFirst 5 rows:")
print(df.head())


# ==========================================
# STEP 2: UNDERSTAND THE DATASET
# ==========================================

# Display the total number of rows and columns
print("\nDataset shape:")
print(df.shape)

# Display all column names
print("\nColumn names:")
print(df.columns.tolist())

# Display the data type of each column
print("\nColumn data types:")
print(df.dtypes)

# Check missing values in every column
print("\nMissing values:")
print(df.isnull().sum())

# Check how many records belong to each target class
print("\nTarget distribution:")
print(df["OUTCOME"].value_counts())

# Check percentage of each target class
print("\nTarget distribution (%):")
print(df["OUTCOME"].value_counts(normalize=True) * 100)


# ==========================================
# STEP 3: SEPARATE FEATURES AND TARGET
# ==========================================

# Remove ID because it is only a record identifier
# Remove OUTCOME because it is our target variable
X = df.drop(columns=["ID", "OUTCOME"])

# Store the target column separately
y = df["OUTCOME"]

# Display the dimensions of input and target
print("\nInput features shape:", X.shape)
print("Target shape:", y.shape)


# ==========================================
# STEP 4: TRAIN-TEST SPLIT
# ==========================================

# Import the function for splitting the dataset
from sklearn.model_selection import train_test_split

# Divide the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,                  # Input features
    y,                  # Target variable
    test_size=0.20,     # 20% for testing, 80% for training
    random_state=42,    # Keep the split reproducible
    stratify=y          # Maintain target class proportions
)

# Display training and testing dimensions
print("\nTraining features:", X_train.shape)
print("Testing features:", X_test.shape)

# Display target dimensions
print("Training target:", y_train.shape)
print("Testing target:", y_test.shape)


# ==========================================
# STEP 5: HANDLE MISSING VALUES
# ==========================================

# Import SimpleImputer to fill missing values
from sklearn.impute import SimpleImputer

# Make copies to preserve the original split data
X_train_clean = X_train.copy()
X_test_clean = X_test.copy()

# Define the columns containing missing values
missing_columns = ["CREDIT_SCORE", "ANNUAL_MILEAGE"]

# Create an imputer that uses the median
imputer = SimpleImputer(strategy="median")

# Calculate medians from training data
# Replace missing values in the training dataset
X_train_clean[missing_columns] = imputer.fit_transform(
    X_train_clean[missing_columns]
)

# Apply the same training medians to the test dataset
X_test_clean[missing_columns] = imputer.transform(
    X_test_clean[missing_columns]
)

# Display the calculated median for each column
print("\nTraining median values:")

for column, median in zip(missing_columns, imputer.statistics_):
    print(column, ":", median)

# Check whether missing values remain in the training set
print("\nMissing values in training data:")
print(X_train_clean.isnull().sum().sum())

# Check whether missing values remain in the test set
print("\nMissing values in testing data:")
print(X_test_clean.isnull().sum().sum())


# ==========================================
# STEP 6: ENCODE ORDINAL CATEGORICAL FEATURES
# ==========================================

# Create copies of the cleaned datasets
X_train_encoded = X_train_clean.copy()
X_test_encoded = X_test_clean.copy()

# Define numerical values for age groups
age_mapping = {
    "16-25": 0,
    "26-39": 1,
    "40-64": 2,
    "65+": 3
}

# Define numerical values for driving experience
experience_mapping = {
    "0-9y": 0,
    "10-19y": 1,
    "20-29y": 2,
    "30y+": 3
}

# Define numerical values for education levels
education_mapping = {
    "none": 0,
    "high school": 1,
    "university": 2
}

# Define numerical values for income groups
income_mapping = {
    "poverty": 0,
    "working class": 1,
    "middle class": 2,
    "upper class": 3
}

# Combine all mappings into one dictionary
ordinal_mappings = {
    "AGE": age_mapping,
    "DRIVING_EXPERIENCE": experience_mapping,
    "EDUCATION": education_mapping,
    "INCOME": income_mapping
}

# Apply each mapping to training and testing data
for column, mapping in ordinal_mappings.items():

    # Convert training categories into numbers
    X_train_encoded[column] = X_train_encoded[column].map(mapping)

    # Convert testing categories using the same mapping
    X_test_encoded[column] = X_test_encoded[column].map(mapping)

# Display encoded columns
print("\nEncoded ordinal features:")
print(
    X_train_encoded[
        ["AGE", "DRIVING_EXPERIENCE", "EDUCATION", "INCOME"]
    ].head()
)


# ==========================================
# STEP 7: ENCODE BINARY CATEGORICAL FEATURES
# ==========================================

# Define numerical values for gender
gender_mapping = {
    "female": 0,
    "male": 1
}

# Define numerical values for race categories
race_mapping = {
    "majority": 0,
    "minority": 1
}

# Define numerical values for vehicle year
vehicle_year_mapping = {
    "before 2015": 0,
    "after 2015": 1
}

# Define numerical values for vehicle type
vehicle_type_mapping = {
    "sedan": 0,
    "sports car": 1
}

# Combine the binary feature mappings
binary_mappings = {
    "GENDER": gender_mapping,
    "RACE": race_mapping,
    "VEHICLE_YEAR": vehicle_year_mapping,
    "VEHICLE_TYPE": vehicle_type_mapping
}

# Apply mappings to both datasets
for column, mapping in binary_mappings.items():

    # Encode training values
    X_train_encoded[column] = X_train_encoded[column].map(mapping)

    # Encode testing values
    X_test_encoded[column] = X_test_encoded[column].map(mapping)

# Display the encoded binary features
print("\nEncoded binary features:")
print(
    X_train_encoded[
        ["GENDER", "RACE", "VEHICLE_YEAR", "VEHICLE_TYPE"]
    ].head()
)


# ==========================================
# STEP 8: ONE-HOT ENCODE POSTAL_CODE
# ==========================================

# Import OneHotEncoder from scikit-learn
from sklearn.preprocessing import OneHotEncoder

# Create the one-hot encoder
postal_encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

# Learn postal code categories from training data
postal_train = postal_encoder.fit_transform(
    X_train_encoded[["POSTAL_CODE"]]
)

# Apply the same encoder to testing data
postal_test = postal_encoder.transform(
    X_test_encoded[["POSTAL_CODE"]]
)

# Get names for the new encoded columns
postal_columns = postal_encoder.get_feature_names_out(
    ["POSTAL_CODE"]
)

# Convert encoded training values into a DataFrame
postal_train_df = pd.DataFrame(
    postal_train,
    columns=postal_columns,
    index=X_train_encoded.index
)

# Convert encoded testing values into a DataFrame
postal_test_df = pd.DataFrame(
    postal_test,
    columns=postal_columns,
    index=X_test_encoded.index
)

# Remove the original POSTAL_CODE column
X_train_encoded = X_train_encoded.drop(columns=["POSTAL_CODE"])
X_test_encoded = X_test_encoded.drop(columns=["POSTAL_CODE"])

# Add the encoded postal code columns
X_train_encoded = pd.concat(
    [X_train_encoded, postal_train_df],
    axis=1
)

# Add the same columns to testing data
X_test_encoded = pd.concat(
    [X_test_encoded, postal_test_df],
    axis=1
)

# Display the dimensions after encoding
print("\nTraining shape after encoding:", X_train_encoded.shape)
print("Testing shape after encoding:", X_test_encoded.shape)


# ==========================================
# STEP 9: VERIFY CATEGORICAL ENCODING
# ==========================================

# Identify non-numerical columns in training data
non_numeric_train = X_train_encoded.select_dtypes(
    exclude=np.number
).columns

# Identify non-numerical columns in testing data
non_numeric_test = X_test_encoded.select_dtypes(
    exclude=np.number
).columns

# Display remaining non-numerical columns
print("\nNon-numeric training columns:")
print(non_numeric_train.tolist())

print("\nNon-numeric testing columns:")
print(non_numeric_test.tolist())

# Display missing-value counts
print("\nTraining missing values:",
      X_train_encoded.isnull().sum().sum())

print("Testing missing values:",
      X_test_encoded.isnull().sum().sum())

# Confirm training and testing columns match
print("\nSame columns in both datasets:",
      X_train_encoded.columns.equals(X_test_encoded.columns))


# ==========================================
# STEP 10: FEATURE SCALING
# ==========================================

# Import StandardScaler from scikit-learn
from sklearn.preprocessing import StandardScaler

# Create a StandardScaler object
scaler = StandardScaler()

# Learn the mean and standard deviation from training data
# Then scale the training features
X_train_scaled = scaler.fit_transform(
    X_train_encoded
)

# Scale test features using the training mean and standard deviation
X_test_scaled = scaler.transform(
    X_test_encoded
)

# Display the shape of the scaled training data
print("\nScaled training shape:")
print(X_train_scaled.shape)

# Display the shape of the scaled testing data
print("\nScaled testing shape:")
print(X_test_scaled.shape)

# Display the first five rows of scaled training features
print("\nFirst five scaled training records:")
print(X_train_scaled[:5])


# ==========================================
# STEP 11: SAVE TEST DATA
# ==========================================

# Import os to work with file paths
import os

# Create a copy of the original test features
test_data = X_test.copy()

# Add the actual target values to the test dataset
test_data["OUTCOME"] = y_test

# Save the test data in the project folder
test_data.to_csv("test_data.csv", index=False)

# Display confirmation
print("\nTest data saved successfully.")

# Display the test dataset shape
print("Test data shape:", test_data.shape)


# ==========================================
# SAVE PREPROCESSING OBJECTS FOR STREAMLIT
# ==========================================


# ==========================================
# SAVE PREPROCESSING OBJECTS FOR STREAMLIT
# ==========================================

# Import joblib for saving objects
import joblib

# Import os for creating a folder
import os

# Create models folder if it does not exist
os.makedirs("models", exist_ok=True)

# Save the fitted median imputer
joblib.dump(imputer, "models/imputer.joblib")

# Save the fitted postal code encoder
joblib.dump(postal_encoder, "models/postal_encoder.joblib")

# Save the fitted feature scaler
joblib.dump(scaler, "models/scaler.joblib")

# Save the exact encoded feature columns used during training
joblib.dump(
    X_train_encoded.columns.tolist(),
    "models/feature_columns.joblib"
)
# Print success message
print("\nPreprocessing objects saved successfully.")


