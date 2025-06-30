import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# === 1. Load Training (2021–2023) and Test (2024) ===
train_df = pd.read_excel(r"..\data\bristol_2021_2023_model_data.csv.xlsx")
test_df = pd.read_excel(r"..\data\bristol_2024_model_data.csv.xlsx")

# Keep matching columns
train_features = ["Postcode_Prefix", "Property_Type", "New_Build", "Tenure", "District", "Year", "Month", "Quarter"]
test_features = train_features

# Drop rows with nulls in any required column
train_df.dropna(subset=train_features + ["Price"], inplace=True)
test_df.dropna(subset=test_features + ["Price"], inplace=True)

# === 2. One-Hot Encoding (align columns) ===
X_train = pd.get_dummies(train_df[train_features], drop_first=True)
X_test = pd.get_dummies(test_df[test_features], drop_first=True)

# Align test columns to train (same dummies)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

y_train = train_df["Price"]
y_test = test_df["Price"]

# === 3. Train Linear Regression ===
model = LinearRegression()
model.fit(X_train, y_train)

# === 4. Predict & Evaluate ===
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(f"\n📊 Linear Regression Forecasting Evaluation (on 2024 data):")
print(f"RMSE: £{rmse:,.0f}")
print(f"MAE : £{mae:,.0f}")

# === 5. Optional: Compare Predictions per Postcode Prefix ===
test_df["Predicted_Price"] = y_pred
postcode_summary = test_df.groupby("Postcode_Prefix")[["Price", "Predicted_Price"]].mean()
print("\n📍 Average Real vs Predicted Price by Postcode_Prefix:")
print(postcode_summary)
