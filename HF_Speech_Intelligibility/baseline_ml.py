import pandas as pd

# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("sample_dataset.csv")

print("Dataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDataset statistics:")
print(df.describe())

# ==========================================
# 2. DEFINE FEATURES AND TARGET
# ==========================================

FEATURES = [
    "RMS_Energy",
    "Zero_Crossing_Rate",
    "Spectral_Centroid_Hz",
    "Spectral_Bandwidth_Hz",
    "Spectral_Rolloff_Hz",
    "Spectral_Flatness"
]

TARGET = "STOI"

X = df[FEATURES]
y = df[TARGET]

print("\nFeature matrix shape:", X.shape)
print("Target shape:", y.shape)

print("\nFeatures being used:")
print(FEATURES)

print("\nFirst 5 feature rows:")
print(X.head())

print("\nFirst 5 target values:")
print(y.head())

from sklearn.model_selection import GroupShuffleSplit

# Use utterance_id as the grouping variable
groups = df["utterance_id"]

# Split by utterance, not by individual rows
gss = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_idx, test_idx = next(gss.split(X, y, groups=groups))

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

print("\nTraining set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

print("\nTraining utterances:")
print(df.iloc[train_idx]["utterance_id"].unique())

print("\nTest utterances:")
print(df.iloc[test_idx]["utterance_id"].unique())

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Create Linear Regression model
linear_model = LinearRegression()

# Train the model
linear_model.fit(X_train, y_train)

# Predict STOI for unseen test utterance
y_pred = linear_model.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n===== Linear Regression Results =====")
print("MAE :", mae)
print("RMSE:", rmse)
print("R²  :", r2)

print("\nActual vs Predicted STOI:")
for actual, predicted in zip(y_test, y_pred):
    print(f"Actual: {actual:.3f} | Predicted: {predicted:.3f}")

    from sklearn.ensemble import RandomForestRegressor

# Create Random Forest model
rf_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

# Train
rf_model.fit(X_train, y_train)

# Predict
rf_pred = rf_model.predict(X_test)

# Evaluation
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)

print("\n===== Random Forest Results =====")
print("MAE :", rf_mae)
print("RMSE:", rf_rmse)
print("R²  :", rf_r2)

print("\nActual vs Predicted STOI:")
for actual, predicted in zip(y_test, rf_pred):
    print(f"Actual: {actual:.3f} | Predicted: {predicted:.3f}")

    # Random Forest feature importance

importance = rf_model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n===== Feature Importance =====")
print(feature_importance)

import matplotlib.pyplot as plt

plt.figure(figsize=(7, 6))

plt.scatter(y_test, y_pred, label="Linear Regression")
plt.scatter(y_test, rf_pred, label="Random Forest")

# Perfect prediction line
min_val = min(y_test.min(), y_pred.min(), rf_pred.min())
max_val = max(y_test.max(), y_pred.max(), rf_pred.max())

plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    linestyle="--",
    label="Perfect Prediction"
)

plt.xlabel("Actual STOI")
plt.ylabel("Predicted STOI")
plt.title("Actual vs Predicted STOI")
plt.legend()
plt.grid(True)

plt.show()