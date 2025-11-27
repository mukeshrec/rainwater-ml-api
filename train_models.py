# train_models.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report,
    mean_squared_error, r2_score
)
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np
import joblib
import json

# ==============================
# Load dataset
# ==============================
df = pd.read_csv("validstructtrainedfinal.csv")

# ==============================
# Encode categorical features
# ==============================
label_encoders = {}
categorical_cols = ["roof_type", "recommended_structure", "feasibility_classification"]

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# ==============================
# Feature set for all models
# ==============================
features = [
    "latitude", "longitude", "roof_area", "open_space", "roof_type",
    "annual_rainfall", "max_daily_rainfall",
    "clay", "sand", "silt",
    "elevation", "evaporation"
]

X = df[features]

# ==============================
# 1️⃣ Feasibility Prediction (Classification)
# ==============================
y_fea = df["feasibility_classification"]
X_train_fea, X_test_fea, y_train_fea, y_test_fea = train_test_split(
    X, y_fea, test_size=0.2, random_state=42
)

clf_feasibility = RandomForestClassifier(n_estimators=300, random_state=42)
clf_feasibility.fit(X_train_fea, y_train_fea)

y_pred_fea = clf_feasibility.predict(X_test_fea)
print("\n=== Feasibility Classification ===")
print("Accuracy:", accuracy_score(y_test_fea, y_pred_fea))
print(classification_report(y_test_fea, y_pred_fea))

# ==============================
# 2️⃣ Structure Type Prediction (Classification)
# ==============================
y_struct = df["recommended_structure"]
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X, y_struct, test_size=0.2, random_state=42
)

clf_structure = DecisionTreeClassifier(max_depth=6, random_state=42)
clf_structure.fit(X_train_s, y_train_s)

y_pred_s = clf_structure.predict(X_test_s)
print("\n=== Structure Recommendation ===")
print("Accuracy:", accuracy_score(y_test_s, y_pred_s))
print(classification_report(y_test_s, y_pred_s))

# Optional: Plot Structure Decision Tree (for visualization only)
plt.figure(figsize=(20, 10))
plot_tree(clf_structure, feature_names=features, filled=True, fontsize=8)
plt.title("Structure Recommendation Decision Tree")
plt.tight_layout()
plt.savefig("structure_decision_tree.png", dpi=300)
plt.close()

# ==============================
# 3️⃣ Annual Runoff Regression
# ==============================
y_runoff = df["annual_runoff"]
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X, y_runoff, test_size=0.2, random_state=42
)

reg_runoff = RandomForestRegressor(n_estimators=300, random_state=42)
reg_runoff.fit(X_train_r, y_train_r)

y_pred_r = reg_runoff.predict(X_test_r)
print("\n=== Annual Runoff Regression ===")
print("Runoff RMSE:", np.sqrt(mean_squared_error(y_test_r, y_pred_r)))
print("Runoff R2 Score:", r2_score(y_test_r, y_pred_r))

# ==============================
# 4️⃣ Infiltration Regression
# ==============================
y_inf = df["infiltration"]
X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
    X, y_inf, test_size=0.2, random_state=42
)

reg_infil = RandomForestRegressor(n_estimators=300, random_state=42)
reg_infil.fit(X_train_i, y_train_i)

y_pred_i = reg_infil.predict(X_test_i)
print("\n=== Infiltration Regression ===")
print("Infiltration RMSE:", np.sqrt(mean_squared_error(y_test_i, y_pred_i)))
print("Infiltration R2 Score:", r2_score(y_test_i, y_pred_i))

# ==============================
# Export Structure Decision Tree Rules (Text)
# ==============================
print("\n=== Structure Decision Tree Rules ===\n")
print(export_text(clf_structure, feature_names=features))

# ==============================
# Save models & utilities for API
# ==============================
print("\nSaving models and encoders...")

joblib.dump(clf_feasibility, "model_feasibility.pkl")
joblib.dump(clf_structure, "model_structure.pkl")
joblib.dump(reg_runoff, "model_runoff.pkl")
joblib.dump(reg_infil, "model_infiltration.pkl")

# Save label encoders
joblib.dump(label_encoders, "label_encoders.pkl")

# Save feature names as JSON (to keep order consistent)
with open("features.json", "w") as f:
    json.dump(features, f)

print("All models and encoders saved successfully.")
