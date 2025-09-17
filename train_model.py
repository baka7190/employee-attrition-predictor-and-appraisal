import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# Feature selection
X = df.drop(['Attrition', 'EmployeeNumber', 'Over18', 'StandardHours', 'EmployeeCount'], axis=1)
y = df['Attrition']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Categorical and numerical
categorical_columns = ['JobRole', 'Department', 'MaritalStatus', 'Gender', 'OverTime', 'BusinessTravel', 'EducationField']
numerical_columns = [col for col in X.columns if col not in categorical_columns]

# One-hot encoding
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoder.fit(X_train[categorical_columns])

X_train_encoded = encoder.transform(X_train[categorical_columns])
X_test_encoded = encoder.transform(X_test[categorical_columns])

X_train_final = np.hstack((X_train[numerical_columns].values, X_train_encoded))
X_test_final = np.hstack((X_test[numerical_columns].values, X_test_encoded))

# Train model
model = LogisticRegression(
    solver="liblinear",          # converges faster for binary problems
    class_weight="balanced",     # keep your improved minority recall
    max_iter=5000,               # plenty of iterations to fully converge
    C=0.5,                       # a bit more regularization helps stability
    random_state=42
)

model.fit(X_train_final, y_train)

# Evaluation
y_pred = model.predict(X_test_final)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save files
joblib.dump(model, 'model.pkl')
joblib.dump(encoder, 'encoder.pkl')

# Save training-time column order so inference uses the exact same order
joblib.dump(
    {"categorical_columns": categorical_columns, "numerical_columns": numerical_columns},
    "columns.joblib"
)
print("✅ Saved columns.joblib with training-time feature order.")


print("✅ Model and encoder saved successfully.")
