import pandas as pd

# === 1. Load the Datasets ===
# df_2021 = pd.read_excel("D:/bristol_house_price_prediction/data/bristol_houseprice_dataset_2021.csv.xlsx")
# df_2022 = pd.read_excel("D:/bristol_house_price_prediction/data/bristol_houseprice_dataset_2022.csv.xlsx")
# df_2023 = pd.read_excel("D:/bristol_house_price_prediction/data/bristol_houseprice_dataset_2023.csv.xlsx")
df_2024 = pd.read_excel("D:/bristol_house_price_prediction/data/bristol_houseprice_dataset_2024.csv.xlsx")

# === 2. Add Year Info ===
# df_2021["Year"] = 2021
# df_2022["Year"] = 2022
# df_2023["Year"] = 2023
df_2024["Year"] = 2024

# === 3. Combine All Years ===
# df = pd.concat([df_2021, df_2022, df_2023], ignore_index=True)
df = pd.concat([df_2024], ignore_index=True)

# === 4. Convert Date ===
df["Date"] = pd.to_datetime(df["Date"], errors='coerce')

# === 5. Feature Engineering ===
df["Month"] = df["Date"].dt.month
df["Quarter"] = df["Date"].dt.quarter
df["Postcode_Prefix"] = df["Postcode"].str.extract(r"^(BS\d{1,2})")

# === 6. Keep Only Useful Columns ===
columns_to_keep = [
    "Price", "Year", "Month", "Quarter", "Postcode_Prefix",
    "Property_Type", "New_Build", "Tenure", "District"
]
df_clean = df[columns_to_keep].dropna()

# === 7. Sort for Visual Comparison (BS1 → BS...) ===
df_clean.sort_values(by=["Postcode_Prefix", "Year", "Month"], inplace=True)

# === 8. Save to CSV ===
df_clean.to_csv("D:/bristol_house_price_prediction/data/bristol_2024_model_data.csv", index=False)

print("Dataset saved to: bristol_2021_2023_model_data.csv")

