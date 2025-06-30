import pandas as pd
import numpy as np
from xgboost import XGBRegressor, plot_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# === 1. Load Data ===
train_df = pd.read_excel(r"..\data\bristol_2021_2023_model_data.csv.xlsx")
test_df = pd.read_excel(r"..\data\bristol_2024_model_data.csv.xlsx")

# === 2. Prepare Features and Target ===
X_train = train_df.drop(columns=["Price"])
y_train = train_df["Price"]
X_test = test_df.drop(columns=["Price"])
y_test = test_df["Price"]

# === 3. Encode Categorical Features ===
combined = pd.concat([X_train, X_test])
combined_encoded = pd.get_dummies(combined, drop_first=True)

X_train_encoded = combined_encoded.iloc[:len(X_train), :]
X_test_encoded = combined_encoded.iloc[len(X_train):, :]

# === 4. Hyperparameter Tuning ===
params = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}

search = RandomizedSearchCV(
    estimator=XGBRegressor(random_state=42),
    param_distributions=params,
    n_iter=20,
    scoring="neg_root_mean_squared_error",
    cv=5,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

search.fit(X_train_encoded, y_train)
best_model = search.best_estimator_

print("\n[Best Parameters from Tuning]")
print(search.best_params_)

# === 5. Predictions and Evaluation ===
y_pred = best_model.predict(X_test_encoded)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(f"\n📊 XGBoost (Tuned) Forecasting Evaluation (on 2024 data):")
print(f"RMSE: £{rmse:,.0f}")
print(f"MAE : £{mae:,.0f}")

# === 6. Compare Real vs Predicted by Postcode ===
test_df["Predicted_Price"] = y_pred
grouped = test_df.groupby("Postcode_Prefix")[["Price", "Predicted_Price"]].mean()
print("\n📍 Average Real vs Predicted Price by Postcode_Prefix:")
print(grouped)

# === 7. Save Residual Plot ===
residuals = y_test - y_pred
sns.histplot(residuals, bins=30, kde=True)
plt.title("XGBoost (Tuned) Residual Distribution")
plt.xlabel("Prediction Error (£)")
plt.tight_layout()
plt.savefig("../plots/xgb_tuned_residuals.png")
plt.close()

# === 8. Save Feature Importance ===
plot_importance(best_model)
plt.title("XGBoost (Tuned) Feature Importance")
plt.tight_layout()
plt.savefig("../plots/xgb_tuned_feature_importance.png")
plt.close()

# === 9. Save Model ===
joblib.dump(best_model, "../models/xgb_bristol_model_tuned.pkl")
print("\n✅ Model saved → models/xgb_bristol_model_tuned.pkl")
