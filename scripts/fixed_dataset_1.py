import pandas as pd

# Load the full UK house price dataset
df = pd.read_csv(r"C:\Users\kaung\Downloads\pp-2021.csv", header=None)

# Assign correct column names
df.columns = [
    "TransactionID", "Price", "Date", "Postcode", "Property_Type", "New_Build",
    "Tenure", "Address_Number", "Blank", "Street", "Locality", "Town_City",
    "District", "County", "Category_Type", "Record_Status"
]

# Filter for Bristol (postcodes starting with 'BS')
bristol_df = df[df["Postcode"].str.startswith("BS", na=False)].copy()

# Convert date column
bristol_df["Date"] = pd.to_datetime(bristol_df["Date"], errors='coerce')

# Drop unused column
bristol_df.drop(columns=["Blank"], inplace=True)

# Save to Excel
# bristol_df.to_excel("D:/bristol_house_price_prediction/data/bristol_houseprice_dataset_2022.xlsx", sheet_name="bristol_houseprice_2022", index=False)

# Or save to CSV
bristol_df.to_csv("D:/bristol_house_price_prediction/data/bristol_houseprice_dataset_2021.csv", index=False) 
