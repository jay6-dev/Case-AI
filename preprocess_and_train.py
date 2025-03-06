import sqlite3
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ----------------------------
# Load Data from SQLite
# ----------------------------
db_filename = "case_management.db"
conn = sqlite3.connect(db_filename)

# Read data from database
df_cases = pd.read_sql("SELECT * FROM cases", conn)
df_clients = pd.read_sql("SELECT * FROM clients", conn)
df_assignees = pd.read_sql("SELECT * FROM assignees", conn)

conn.close()

# ----------------------------
# Data Preprocessing
# ----------------------------
# Drop non-numeric and unnecessary columns
df_cases = df_cases.drop(columns=["case_id", "description", "creation_date"], errors="ignore")

# Merge with clients and assignees
df = df_cases.merge(df_clients, on="client_id", how="left").merge(df_assignees, on="assignee_id", how="left")

# Drop ID and redundant name columns
df = df.drop(columns=["client_id", "assignee_id", "name_x", "name_y"], errors="ignore")

# Handle missing values
imputer = SimpleImputer(strategy="most_frequent")  
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
df_imputed.index = df.index

# Separate categorical and numerical features
categorical_cols = ["case_type", "status", "priority", "risk_level", "role", "outcome"]
numerical_cols = ["resolution_time", "age", "previous_cases", "resolved_cases", "pending_cases"]

# One-hot encode categorical variables (final fix)
encoder = OneHotEncoder(drop="first")  # No more sparse_output issue
encoded_categorical = pd.DataFrame(encoder.fit_transform(df_imputed[categorical_cols]).toarray(),
                                   columns=encoder.get_feature_names_out(categorical_cols),
                                   index=df.index)  # Ensure indexing matches

# Standardize numerical features
scaler = StandardScaler()
scaled_numerical = pd.DataFrame(scaler.fit_transform(df_imputed[numerical_cols]), 
                                columns=numerical_cols,
                                index=df.index)  # Ensure indexing matches

# Combine processed data
df_processed = pd.concat([encoded_categorical, scaled_numerical], axis=1)

# Ensure target column exists and is correctly formatted
if "status_Resolved" in df_processed.columns:
    target = df_processed["status_Resolved"]
else:
    raise ValueError("Target column 'status_Resolved' not found in dataset.")

# Drop target column from features
X = df_processed.drop(columns=["status_Resolved"], errors="ignore")

# ----------------------------
# Train Machine Learning Model
# ----------------------------
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.2, random_state=42)

# Ensure train-test split matches in shape
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)  # Align test columns with train

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# ----------------------------
# Evaluate the Model
# ----------------------------
y_pred = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

# ----------------------------
# Save the Model
# ----------------------------
joblib.dump(model, "model.pkl")
print("✅ Model saved as model.pkl")
