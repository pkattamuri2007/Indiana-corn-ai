import pandas as pd
import numpy as np
from date_utils import parse_date_col


df = pd.read_csv("combined_features.csv")
df["date"] = parse_date_col(df["date"])


# Ensure sorted per location
df = df.sort_values(["latitude", "longitude", "date"])

# --- Helper rolling stats we still need ---
# 7-day averages/sums for solar radiation and RH
for col, fn, out in [
    ("ALLSKY_SFC_SW_DWN", "mean", "ALLSKY_7d_avg"),
    ("RH2M", "mean", "RH2M_7d_avg"),
    ("PRECTOTCORR", "sum", "PRECTOTCORR_7d_sum_recalc"),  # safe recompute
    ("T2M", "mean", "T2M_7d_avg_recalc"),
]:
    df[out] = (
        df.groupby(["latitude", "longitude"])[col]
          .transform(lambda x: x.rolling(7, min_periods=1).mean() if fn=="mean"
                     else x.rolling(7, min_periods=1).sum())
    )

# Use previously built features if present, else fall back to these:
df["T2M_7d_avg"] = df.get("T2M_7d_avg", df["T2M_7d_avg_recalc"])
df["PRECTOTCORR_7d_sum"] = df.get("PRECTOTCORR_7d_sum", df["PRECTOTCORR_7d_sum_recalc"])

# Location-wise median solar to define "sunny"
solar_median = df.groupby(["latitude","longitude"])["ALLSKY_7d_avg"].transform("median")

# --- Proxy labels ---

# 1) Planting window (corn-ish heuristics)
cond_gdd = (df["Cumulative_GDD"] >= 100) & (df["Cumulative_GDD"] <= 200)
cond_temp = (df["T2M_7d_avg"] >= 10) & (df["T2M_7d_avg"] <= 25)
cond_rain = (df["PRECTOTCORR_7d_sum"] >= 5) & (df["PRECTOTCORR_7d_sum"] <= 30)
df["planting_window"] = (cond_gdd & cond_temp & cond_rain).astype(int)

# 2) Irrigation flag (dry + sunny + warm)
cond_dry = df["PRECTOTCORR_7d_sum"] < 10
cond_sunny = df["ALLSKY_7d_avg"] > solar_median
cond_warm = df["T2M_7d_avg"] > 22
df["irrigation_flag"] = (cond_dry & cond_sunny & cond_warm).astype(int)

# Save labeled data
df.to_csv("labeled_features.csv", index=False)

# Quick summary
print("Saved labeled_features.csv")
print("Planting window days:", df["planting_window"].sum())
print("Irrigation flag days:", df["irrigation_flag"].sum())
print(df[["date","latitude","longitude","T2M_7d_avg","PRECTOTCORR_7d_sum",
          "ALLSKY_7d_avg","Cumulative_GDD","planting_window","irrigation_flag"]].head(12))
