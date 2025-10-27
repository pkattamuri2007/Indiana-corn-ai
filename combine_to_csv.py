#deprecated—use rebuild_combined.py


import json
import pandas as pd
import glob
import os

def load_json_to_df(filepath):
    """Load a single NASA POWER JSON file and flatten it into a long DataFrame."""
    with open(filepath, 'r') as file:
        data = json.load(file)

    lat = data["geometry"]["coordinates"][1]
    lon = data["geometry"]["coordinates"][0]
    params = data["properties"]["parameter"]

    # Build a list of rows: each date-variable pair becomes a row
    rows = []
    for var, date_dict in params.items():
        for date, value in date_dict.items():
            rows.append({
                "date": date,
                "variable": var,
                "value": value,
                "latitude": lat,
                "longitude": lon
            })

    return pd.DataFrame(rows)

def combine_jsons_to_csv(folder='.', output='combined_data.csv'):
    """Combine multiple JSON files into a single clean CSV."""
    all_files = glob.glob(os.path.join(folder, "PowerData_Lat_*_Lon_*.json"))
    print(f"🔍 Found {len(all_files)} JSON files.")

    if len(all_files) == 0:
        print("⚠️ No files found. Check your folder or filename pattern.")
        return

    # Load and combine all JSONs
    all_dfs = [load_json_to_df(f) for f in all_files]
    combined = pd.concat(all_dfs, ignore_index=True)

    # Pivot variables into columns (e.g., T2M, PRECTOTCORR, RH2M, etc.)
    combined = combined.pivot_table(
        index=["date", "latitude", "longitude"],
        columns="variable",
        values="value"
    ).reset_index()

    # Save clean combined dataset
    combined.to_csv(output, index=False)
    print(f"✅ Combined {len(all_files)} JSON files into {output}")
    print(f"📄 Columns: {combined.columns.tolist()}")
    print(f"📊 Shape: {combined.shape}")

if __name__ == "__main__":
    combine_jsons_to_csv()
