# Bank Customer Churn Prediction 

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from xgboost import XGBClassifier

#  Load the cleaned dataset

DATA_PATH = "data/cleaned_bank_churn.csv"
df = pd.read_csv(DATA_PATH)

print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")

#  Prepare features and target
X = df.drop('churn', axis=1)
y = df['churn']

#  Scale numeric features

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Train XGBoost Classifier (Final Model)

print("\n Training XGBoost model...")

xgb_clf = XGBClassifier(
    eval_metric='logloss',
    random_state=42
)

# Optional tuning - comment out if already tuned
param_grid = {
    'n_estimators': [200],
    'max_depth': [5],
    'learning_rate': [0.05],
    'subsample': [0.8]
}

grid = GridSearchCV(xgb_clf, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
print("\n Best model parameters:", grid.best_params_)

#  Evaluate the final model

y_pred = best_model.predict(X_test)
print("\n Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1]))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

#  Save model and scaler

os.makedirs("models", exist_ok=True)

pickle.dump(best_model, open("models/model.pkl", "wb"))
pickle.dump(scaler, open("models/scaler.pkl", "wb"))

print("\n Model and scaler saved successfully in /models folder!")
print(" Training completed.")
