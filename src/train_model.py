import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)

# Ensure artifacts directory exists
os.makedirs("artifacts", exist_ok=True)

# Load the cleaned dataset
df = pd.read_csv(r'C:\Users\IfeomaAugustaAdigwe\Desktop\creditscore\data\clean_data.csv')
print("📌 Columns in loaded data:", df.columns.tolist())

# Handle invalid age entries
df['Age'].replace(0, np.nan, inplace=True)
df['Age'].fillna(df['Age'].mean(), inplace=True)

# Encode categorical variables
label_encoders = {}
categorical_columns = df.select_dtypes(include=['object', 'category']).columns

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Drop identifier columns if present
for col in ['ID', 'Customer_ID']:
    df.drop(columns=col, inplace=True, errors='ignore')

# Define features and target
target_col = 'Credit_Score'
y = df[target_col]
X = df.drop(columns=target_col)

# Feature selection using Random Forest
rf_selector = RandomForestClassifier(random_state=42)
rf_selector.fit(X, y)
feature_importances = pd.Series(rf_selector.feature_importances_, index=X.columns).sort_values(ascending=False)
selected_features = feature_importances.head(10).index.tolist()

# Plot and save top feature importances
plt.figure(figsize=(12, 6))
feature_importances[selected_features].sort_values().plot(kind='barh', color='steelblue')
plt.title("Top 10 Feature Importances for Credit Score")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("artifacts/feature_importance.png")
plt.close()

# Train-test split using selected features
X_selected = df[selected_features]
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save artifacts
joblib.dump(scaler, "src/scaler.pkl")
joblib.dump((X_train_scaled, X_test_scaled, y_train, y_test), "src/train_test_split.pkl")

print("✅ Training pipeline completed successfully.")
print("📦 Top 10 selected features:", selected_features)
