import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from date_utils import parse_date_col

recs = pd.read_csv("recommendations.csv")
labeled = pd.read_csv("labeled_features.csv")

recs["date"] = parse_date_col(recs["date"])
labeled["date"] = parse_date_col(labeled["date"])

# Join on date+location to access ground truth labels
key = ["date","latitude","longitude"]
df = pd.merge(recs, labeled[key+["planting_window","irrigation_flag"]], on=key, how="inner")

def best_threshold(prob_col, y_col):
    best_t, best_f1 = 0.5, -1
    for t in np.linspace(0.3, 0.8, 26):
        f1 = f1_score(df[y_col], (df[prob_col] >= t).astype(int))
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1

pt, pf1 = best_threshold("planting_prob", "planting_window")
it, if1 = best_threshold("irrigation_prob", "irrigation_flag")

print(f"Suggested thresholds → planting: {pt:.2f} (F1={pf1:.3f}), irrigation: {it:.2f} (F1={if1:.3f})")
