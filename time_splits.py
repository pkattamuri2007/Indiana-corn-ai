# time_splits.py
import numpy as np
import pandas as pd

def year_group_splits(df, n_test_years=2, min_years_train=2):
    """
    Yield (train_idx, test_idx) by rolling year windows.
    Assumes df has a datetime 'date' column and the index aligns to X/y.
    """
    d = df.copy()
    d["year"] = d["date"].dt.year
    years = sorted(d["year"].unique())
    for i in range(min_years_train, len(years) - n_test_years + 1):
        train_years = years[:i]
        test_years  = years[i:i + n_test_years]
        tr_idx = d.index[d["year"].isin(train_years)]
        te_idx = d.index[d["year"].isin(test_years)]
        if len(tr_idx) and len(te_idx):
            yield (np.array(tr_idx), np.array(te_idx))
