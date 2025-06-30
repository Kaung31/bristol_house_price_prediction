# train_rf_forecast.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import joblib

# === 1. Load Data ===
train_df = pd.read_excel(r"..\data\bristol_2021_2023_model_data.csv.xlsx")
test_df = pd.read_excel(r"..\data\bristol_2024_model_data.csv.xlsx")

# === 2. Drop unnecessary columns ===
drop_cols = ["Price"]
X_train = train_df.drop(columns=drop_cols)
y_train = train_df["Price"]

X_test = test_df.drop(columns=drop_cols)
y_test = test_df["Price"]

# === 3. One-Hot Encode Categorical Variables ===
combined = pd.concat([X_train, X_test], axis=0)
combined_encoded = pd.get_dummies(combined, drop_first=True)

X_train_encoded = combined_encoded.iloc[:len(X_train), :]
X_test_encoded = combined_encoded.iloc[len(X_train):, :]

# === 4. Train Random Forest Model ===
model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
model.fit(X_train_encoded, y_train)

# === 5. Predictions and Evaluation ===
y_pred = model.predict(X_test_encoded)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print("\nRandom Forest Forecasting Evaluation (on 2024 data):")
print(f"RMSE: £{rmse:,.0f}")
print(f"MAE : £{mae:,.0f}")

# === 6. Compare by Postcode Prefix ===
test_df["Predicted_Price"] = y_pred
grouped = test_df.groupby("Postcode_Prefix")[["Price", "Predicted_Price"]].mean()
print("\n📍 Average Real vs Predicted Price by Postcode_Prefix:")
print(grouped)

# === 7. Save Model ===
joblib.dump(model, "../models/rf_bristol_model.pkl")
print("\nModel saved to → models/rf_bristol_model.pkl")
