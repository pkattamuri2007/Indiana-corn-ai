import json
import pandas as pd
import glob
import os

def load_json_to_df(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)

    lat = data['geometry']['coordinates'][1]
    lon = data['geometry']['coordinates'][0]
    params = data['properties']['parameter']

    df = pd.DataFrame(params)
    df = df.T.reset_index().rename(columns={'index': 'date'})
    df['latitude'] = lat
    df['longitude'] = lon
    return df

def combine_jsons_to_csv(folder='.', output='combined_data.csv'):
    all_files = glob.glob(os.path.join(folder, "PowerData_Lat_*.json"))
    all_dfs = [load_json_to_df(f) for f in all_files]
    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(output, index=False)
    print(f"✅ Combined {len(all_files)} JSON files into {output}")

if __name__ == "__main__":
    combine_jsons_to_csv()
