import pandas as pd
import numpy as np
from date_utils import parse_date_col

# Load cleaned data
df = pd.read_csv("combined_data_cleaned.csv")

# --- Convert date column to datetime type (not epoch) ---
df["date"] = parse_date_col(df["date"])

# --- Sort data for each location ---
df = df.sort_values(by=["latitude", "longitude", "date"])

# --- Moving averages (7-day and 30-day) ---
df["T2M_7d_avg"] = df.groupby(["latitude", "longitude"])["T2M"].transform(lambda x: x.rolling(7, min_periods=1).mean())
df["T2M_30d_avg"] = df.groupby(["latitude", "longitude"])["T2M"].transform(lambda x: x.rolling(30, min_periods=1).mean())

df["PRECTOTCORR_7d_sum"] = df.groupby(["latitude", "longitude"])["PRECTOTCORR"].transform(lambda x: x.rolling(7, min_periods=1).sum())
df["PRECTOTCORR_30d_sum"] = df.groupby(["latitude", "longitude"])["PRECTOTCORR"].transform(lambda x: x.rolling(30, min_periods=1).sum())

# --- Growing Degree Days (GDD) ---
base_temp = 10  # °C
df["GDD"] = (df["T2M"] - base_temp).clip(lower=0)

# Reset cumulative each calendar year (per location)
df["year"] = df["date"].dt.year
df["Cumulative_GDD_year"] = (
    df.groupby(["latitude", "longitude", "year"])["GDD"]
      .cumsum()
)

# (optional) keep your old column name for backwards compatibility
df["Cumulative_GDD"] = df["Cumulative_GDD_year"]


# --- Save processed dataset ---
df.to_csv("combined_features.csv", index=False)
print("🌱 Feature-engineered dataset saved as combined_features.csv")
print(df.head())

# --- Extra agro features ---
# 1) Vapor Pressure Deficit (VPD) ~ dryness driver (kPa)
#    Saturation vapor pressure (Tetens), T in °C

es = 0.6108 * np.exp((17.27 * df["T2M"]) / (df["T2M"] + 237.3))
ea = es * (df["RH2M"] / 100.0)
df["VPD_kPa"] = (es - ea).clip(lower=0)

# 2) Reference ET0 (Hargreaves-Samani) — mm/day (coarse but useful)
#    Needs daily Tmin/Tmax ideally; POWER T2M is mean. We’ll approximate range.
#    If you later add T2M_MAX / T2M_MIN, replace the range line with (tmax - tmin).
t_range = 8.0  # °C proxy diurnal range — replace with (T2M_MAX - T2M_MIN) if you add them
Ra = df["ALLSKY_SFC_SW_DWN"] * 0.408  # convert MJ/m^2/day to mm equivalent factor
df["ET0_mm"] = 0.0023 * (df["T2M"] + 17.8) * (t_range ** 0.5) * Ra

# 3) Cumulative & tails
df["Rain_14d_sum"] = df.groupby(["latitude","longitude"])["PRECTOTCORR"].transform(lambda x: x.rolling(14, min_periods=1).sum())
df["Rain_3d_sum"]  = df.groupby(["latitude","longitude"])["PRECTOTCORR"].transform(lambda x: x.rolling(3,  min_periods=1).sum())
df["ET0_7d_sum"]   = df.groupby(["latitude","longitude"])["ET0_mm"].transform(lambda x: x.rolling(7,  min_periods=1).sum())
df["VPD_7d_avg"]   = df.groupby(["latitude","longitude"])["VPD_kPa"].transform(lambda x: x.rolling(7,  min_periods=1).mean())

