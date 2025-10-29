import pandas as pd

def parse_date_col(s):
    # Robust: try ISO first, then YYYYMMDD
    s = s.astype(str)
    d = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    miss = d.isna()
    if miss.any():
        d.loc[miss] = pd.to_datetime(s.loc[miss], format="%Y%m%d", errors="coerce")
    return d
