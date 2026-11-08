
# ==========================================
# ML ASSIGNMENT 2 - DATA PREPROCESSING
# Dataset: Car Insurance Claim
# ==========================================

# Import pandas for reading and analyzing data
import pandas as pd

# Import kagglehub to download the dataset
import kagglehub

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

