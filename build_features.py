import pandas as pd

# Load cleaned data
df = pd.read_csv("combined_data_cleaned.csv")

# --- Convert date column to datetime type ---
df["date"] = pd.to_datetime(df["date"])

# --- Sort data for each location ---
df = df.sort_values(by=["latitude", "longitude", "date"])

# --- Moving averages (7-day and 30-day) ---
df["T2M_7d_avg"] = df.groupby(["latitude", "longitude"])["T2M"].transform(lambda x: x.rolling(7, min_periods=1).mean())
df["T2M_30d_avg"] = df.groupby(["latitude", "longitude"])["T2M"].transform(lambda x: x.rolling(30, min_periods=1).mean())

df["PRECTOTCORR_7d_sum"] = df.groupby(["latitude", "longitude"])["PRECTOTCORR"].transform(lambda x: x.rolling(7, min_periods=1).sum())
df["PRECTOTCORR_30d_sum"] = df.groupby(["latitude", "longitude"])["PRECTOTCORR"].transform(lambda x: x.rolling(30, min_periods=1).sum())

# --- Growing Degree Days (GDD) approximation ---
base_temp = 10  # base temperature for corn in °C
df["GDD"] = df["T2M"] - base_temp
df.loc[df["GDD"] < 0, "GDD"] = 0
df["Cumulative_GDD"] = df.groupby(["latitude", "longitude"])["GDD"].cumsum()

# --- Save processed dataset ---
df.to_csv("combined_features.csv", index=False)
print("🌱 Feature-engineered dataset saved as combined_features.csv")
print(df.head())
