import json
import pandas as pd
import glob
import os

def load_json_to_long_df(filepath):
    """
    Load a single NASA POWER JSON file and flatten it into a long DataFrame:
    one row per (date, variable, value, lat, lon).
    """
    with open(filepath, "r") as f:
        data = json.load(f)

    lat = data["geometry"]["coordinates"][1]
    lon = data["geometry"]["coordinates"][0]
    params = data["properties"]["parameter"]  # dict: var -> { YYYYMMDD: value }

    rows = []
    for var, datedict in params.items():
        for date_key, value in datedict.items():
            rows.append({
                "date": str(date_key),      # POWER gives YYYYMMDD
                "variable": var,
                "value": value,
                "latitude": lat,
                "longitude": lon
            })

    return pd.DataFrame(rows)

def rebuild(folder=".", out_csv="combined_data.csv", out_long_debug="combined_long_debug.csv"):
    files = glob.glob(os.path.join(folder, "PowerData_Lat_*_Lon_*.json"))
    if not files:
        print("No POWER JSON files found.")
        return

    # Long form from all files
    long_parts = [load_json_to_long_df(p) for p in files]
    long_df = pd.concat(long_parts, ignore_index=True)

    # Parse POWER YYYYMMDD to calendar date (no epoch!)
    long_df["date"] = pd.to_datetime(
        long_df["date"].astype(str).str.zfill(8),
        format="%Y%m%d",
        errors="coerce"
    )

    # Optional: save a debug version (helps trace any odd rows)
    long_df.sort_values(["latitude", "longitude", "date", "variable"], inplace=True)
    long_df.to_csv(out_long_debug, index=False)

    # Pivot into wide form: one row per (date, lat, lon)
    wide = long_df.pivot_table(
        index=["date", "latitude", "longitude"],
        columns="variable",
        values="value",
        aggfunc="mean"  # if dupes exist, average them
    ).reset_index()

    wide.drop_duplicates(inplace=True)
    wide.sort_values(["latitude", "longitude", "date"], inplace=True)
    wide.to_csv(out_csv, index=False)

    print(f"✅ Rebuilt {out_csv}")
    print(f"🪪 Debug long CSV: {out_long_debug}")
    print(f"Columns: {wide.columns.tolist()}")
    print(f"Shape: {wide.shape}")

if __name__ == "__main__":
    rebuild()
