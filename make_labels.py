import pandas as pd
import numpy as np
from date_utils import parse_date_col

# ----------------------------
# Tunable label heuristics
# ----------------------------
# Planting (broadened)
GDD_MIN, GDD_MAX = 50, 300          # was 100–200
T7_MIN,  T7_MAX  = 8.0, 28.0        # was 10–25
RAIN7_MIN, RAIN7_MAX = 0.0, 40.0    # was 5–30
WORKABLE_RAIN_3D_MAX = 20.0         # new: soil workability proxy

# Irrigation (easier trigger + ET deficit)
DRY_RAIN7_MAX     = 15.0            # was <10
WARM_T7_MIN       = 18.0            # was >22
SUN_Q             = 0.60            # use 60th pct (not median) for “sunny”
ET_DEFICIT_7D_MIN = 8.0             # ET0_7d_sum - Rain_7d_sum > this

# ----------------------------
# Load / prep
# ----------------------------
df = pd.read_csv("combined_features.csv")
df["date"] = parse_date_col(df["date"])
df = df.sort_values(["latitude", "longitude", "date"]).reset_index(drop=True)

# Ensure rolling helpers exist (or recompute)
for col, fn, out in [
    ("ALLSKY_SFC_SW_DWN", "mean", "ALLSKY_7d_avg"),
    ("RH2M",               "mean", "RH2M_7d_avg"),
    ("PRECTOTCORR",        "sum",  "PRECTOTCORR_7d_sum_recalc"),
    ("T2M",                "mean", "T2M_7d_avg_recalc"),
]:
    if out not in df.columns:
        df[out] = (
            df.groupby(["latitude", "longitude"])[col]
              .transform(lambda x: x.rolling(7, min_periods=1).mean() if fn == "mean"
                         else x.rolling(7, min_periods=1).sum())
        )

if "T2M_7d_avg" not in df.columns:
    df["T2M_7d_avg"] = df["T2M_7d_avg_recalc"]
if "PRECTOTCORR_7d_sum" not in df.columns:
    df["PRECTOTCORR_7d_sum"] = df["PRECTOTCORR_7d_sum_recalc"]

# ----------------------------
# Per-year cumulative GDD
# ----------------------------
if "GDD" not in df.columns:
    df["GDD"] = (df["T2M"] - 10.0).clip(lower=0)

df["year"] = df["date"].dt.year
df["Cumulative_GDD_year"] = (
    df.groupby(["latitude", "longitude", "year"])["GDD"].cumsum()
)
df["Cumulative_GDD"] = df["Cumulative_GDD_year"]  # backward compat

# Convenience windows we reference
if "Rain_3d_sum" not in df.columns:
    df["Rain_3d_sum"] = df.groupby(["latitude","longitude"])["PRECTOTCORR"]\
                          .transform(lambda x: x.rolling(3, min_periods=1).sum())
if "ET0_7d_sum" not in df.columns and "ET0_mm" in df.columns:
    df["ET0_7d_sum"] = df.groupby(["latitude","longitude"])["ET0_mm"]\
                          .transform(lambda x: x.rolling(7, min_periods=1).sum())
if "ET0_7d_sum" not in df.columns:
    df["ET0_7d_sum"] = 0.0  # safe default if ET not computed yet

# ----------------------------
# Planting label (RELAXED)
# ----------------------------
cond_gdd  = (df["Cumulative_GDD_year"] >= GDD_MIN) & (df["Cumulative_GDD_year"] <= GDD_MAX)
cond_temp = (df["T2M_7d_avg"] >= T7_MIN) & (df["T2M_7d_avg"] <= T7_MAX)
cond_rain = (df["PRECTOTCORR_7d_sum"] >= RAIN7_MIN) & (df["PRECTOTCORR_7d_sum"] <= RAIN7_MAX)
cond_work = (df["Rain_3d_sum"] <= WORKABLE_RAIN_3D_MAX)
df["planting_window"] = (cond_gdd & cond_temp & cond_rain & cond_work).astype(int)

# ----------------------------
# Irrigation label (RELAXED V2)
# ----------------------------
# “Sunny” threshold: median per location (50th pct), not 60th
sun_cut = df.groupby(["latitude","longitude"])["ALLSKY_7d_avg"]\
            .transform(lambda x: x.quantile(0.50))

# Easier dryness + warmth; smaller ET deficit
DRY_RAIN7_MAX     = 20.0    # was 15
WARM_T7_MIN       = 15.0    # was 18 (and previously 22)
ET_DEFICIT_7D_MIN = 4.0     # was 8

cond_dry   = df["PRECTOTCORR_7d_sum"] <= DRY_RAIN7_MAX
cond_warm  = df["T2M_7d_avg"]        >= WARM_T7_MIN
cond_sunny = df["ALLSKY_7d_avg"]     >= sun_cut
et_deficit = (df["ET0_7d_sum"] - df["PRECTOTCORR_7d_sum"]) >= ET_DEFICIT_7D_MIN

# Require dry + warm and (either sunny or ET deficit)
df["irrigation_flag"] = (cond_dry & cond_warm & (cond_sunny | et_deficit)).astype(int)


# ----------------------------
# Save & report
# ----------------------------
df.to_csv("labeled_features.csv", index=False)

print("Saved labeled_features.csv")
print("Planting positives:", int(df["planting_window"].sum()))
print("Irrigation positives:", int(df["irrigation_flag"].sum()))

# Prevalence by year (quick view)
print("\nPlanting prevalence by year:")
print(pd.crosstab(df["date"].dt.year, df["planting_window"]))
print("\nIrrigation prevalence by year:")
print(pd.crosstab(df["date"].dt.year, df["irrigation_flag"]))
