import os, glob, json
import pandas as pd
from math import isnan

PATTERN = "PowerData_Lat_*_Lon_*.json"

def load_one_json(path):
    with open(path, "r") as f:
        data = json.load(f)

    # Skip API error files
    if isinstance(data.get("header"), str) and "failed" in data["header"].lower():
        print(f"⚠️ Skipping error file: {os.path.basename(path)}")
        return pd.DataFrame()

    lat = data["geometry"]["coordinates"][1]
    lon = data["geometry"]["coordinates"][0]
    params = data["properties"]["parameter"]  # dict: {VAR: {date: value}}

    # Build long rows: one row per (date, var, value, lat, lon)
    rows = []
    for var, datedict in params.items():
        for date_str, val in datedict.items():
            rows.append({
                "date": date_str,          # e.g., "20150101"
                "variable": var,           # e.g., "T2M"
                "value": val,
                "latitude": lat,
                "longitude": lon
            })
    return pd.DataFrame(rows)

def main():
    files = glob.glob(PATTERN)
    if not files:
        print(f"❌ No files matched pattern: {PATTERN}")
        return

    long_parts = []
    for fp in files:
        df_part = load_one_json(fp)
        if not df_part.empty:
            long_parts.append(df_part)

    if not long_parts:
        print("❌ No usable JSONs found.")
        return

    long_df = pd.concat(long_parts, ignore_index=True)

    # Replace sentinel -999/-999.0 with NaN
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
    long_df.loc[long_df["value"] <= -999, "value"] = pd.NA

    # Pivot: columns become variables, rows are (date, lat, lon)
    tidy = long_df.pivot_table(
        index=["date", "latitude", "longitude"],
        columns="variable",
        values="value"
    ).reset_index()

    # Ensure 'date' stays a column (not index) and sort
    tidy = tidy.sort_values(["date", "latitude", "longitude"])

    # Save both long and wide (tidy) for debugging/inspection if needed
    long_df.to_csv("combined_long_debug.csv", index=False)
    tidy.to_csv("combined_data.csv", index=False)

    print("✅ Rebuilt CSVs:")
    print("  • combined_data.csv (tidy, one row per date per point)")
    print("  • combined_long_debug.csv (long format for debugging)")
    print(f"📄 Columns: {tidy.columns.tolist()}")
    print(f"📊 Shape: {tidy.shape}")

if __name__ == "__main__":
    main()
