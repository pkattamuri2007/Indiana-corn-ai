import pandas as pd
df = pd.read_csv("combined_data.csv")
print(df.head())
print("\nColumns:", df.columns.tolist())
print("\nShape:", df.shape)
