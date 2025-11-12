import pandas as pd

def parse_date_col(s):
    """
    Parse a pandas Series of dates that may be ISO (YYYY-MM-DD) or POWER-style (YYYYMMDD).
    Returns datetime64[ns] (naive). Robust to stray whitespace and mixed formats.
    """
    # If it's already datetime-like, just normalize to datetime64[ns] and return.
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s)

    s = s.astype(str).str.strip()

    # 1) Try ISO first
    d = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")

    # 2) Anything still NaT but looks like 8 digits -> POWER YYYYMMDD
    miss = d.isna()
    if miss.any():
        looks_power = s.str.match(r"^\d{8}$", na=False) & miss
        if looks_power.any():
            d.loc[looks_power] = pd.to_datetime(s.loc[looks_power], format="%Y%m%d", errors="coerce")

    return d
