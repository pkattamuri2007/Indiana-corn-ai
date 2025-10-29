import pandas as pd

# Load the combined CSV
df = pd.read_csv("combined_data.csv")

# --- Basic checks ---
print("\n✅ Data Overview:")
print(df.info())

print("\n📊 First 5 rows:")
print(df.head())

# --- Check for missing or duplicate data ---
print("\n🔍 Missing values per column:")
print(df.isna().sum())

print("\n📋 Duplicate rows:", df.duplicated().sum())

# --- Remove any duplicates or nulls ---
df = df.dropna().drop_duplicates()

# --- Check column names ---
print("\n🧾 Column names:")
print(df.columns.tolist())

# --- Save a cleaned version ---
df.to_csv("combined_data_cleaned.csv", index=False)
print("\n💾 Cleaned CSV saved as combined_data_cleaned.csv")
