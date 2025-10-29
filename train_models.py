# train_models.py
import pandas as pd
import numpy as np
from date_utils import parse_date_col


from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# -----------------------
# 1) Load labeled data
# -----------------------
df = pd.read_csv("labeled_features.csv")
df["date"] = parse_date_col(df["date"])

# Identify groups (location) for grouped CV so we don't leak between same field
groups = df[["latitude", "longitude"]].astype(str).agg("_".join, axis=1)

# Columns to exclude from features
exclude = {
    "date", "latitude", "longitude",
    "planting_window", "irrigation_flag"
}
feature_cols = [c for c in df.columns if c not in exclude]

# Safety: drop any remaining rows with NaNs in features
X = df[feature_cols].replace([np.inf, -np.inf], np.nan).dropna(axis=0)
valid_idx = X.index
y_plant = df.loc[valid_idx, "planting_window"].astype(int)
y_irrig = df.loc[valid_idx, "irrigation_flag"].astype(int)
groups_valid = groups.loc[valid_idx]

print(f"Using {len(feature_cols)} features:\n{feature_cols}\n")
print(f"Training rows: {len(valid_idx)}; Locations: {groups_valid.nunique()}")

# -----------------------
# 2) Define candidate models
# -----------------------
pipe_logreg = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
])

pipe_rf = Pipeline([
    # (RF is tree-based, no scaling needed, but keep a uniform API)
    ("clf", RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=2,
        class_weight="balanced_subsample", random_state=42
    ))
])

models = {
    "logreg": pipe_logreg,
    "rf": pipe_rf,
}

# -----------------------
# 3) Grouped CV evaluation
# -----------------------
def evaluate_label(y, label_name):
    print(f"\n=== Evaluating label: {label_name} ===")
    gkf = GroupKFold(n_splits=min(5, groups_valid.nunique()))  # up to 5 folds

    results = {}
    for name, mdl in models.items():
        auc = cross_val_score(
            mdl, X, y, cv=gkf.split(X, y, groups_valid),
            scoring="roc_auc", n_jobs=-1
        )
        f1 = cross_val_score(
            mdl, X, y, cv=gkf.split(X, y, groups_valid),
            scoring="f1", n_jobs=-1
        )
        results[name] = {"auc_mean": auc.mean(), "auc_std": auc.std(),
                         "f1_mean": f1.mean(), "f1_std": f1.std()}
        print(f"{name:>7}  ROC-AUC: {auc.mean():.3f} ± {auc.std():.3f} | F1: {f1.mean():.3f} ± {f1.std():.3f}")

    # pick best by AUC
    best_name = max(results, key=lambda k: results[k]["auc_mean"])
    print(f"→ Best model for {label_name}: {best_name} (by ROC-AUC)")
    return best_name, models[best_name]

best_plant_name, best_plant_model = evaluate_label(y_plant, "planting_window")
best_irrig_name, best_irrig_model = evaluate_label(y_irrig, "irrigation_flag")

# -----------------------
# 4) Fit best models on ALL data
# -----------------------
best_plant_model.fit(X, y_plant)
best_irrig_model.fit(X, y_irrig)

# -----------------------
# 5) Make probabilities & actions
# -----------------------
plant_prob = pd.Series(best_plant_model.predict_proba(X)[:, 1], index=valid_idx, name="planting_prob")
irrig_prob = pd.Series(best_irrig_model.predict_proba(X)[:, 1], index=valid_idx, name="irrigation_prob")

out = df.loc[valid_idx, ["date", "latitude", "longitude"]].copy()
out["planting_prob"] = plant_prob.values
out["irrigation_prob"] = irrig_prob.values

# Thresholds – tweak as you like
PLANT_TH = 0.60
IRRIG_TH = 0.60

out["planting_recommendation"] = np.where(
    out["planting_prob"] >= PLANT_TH, "Consider planting this week", "Hold"
)
out["irrigation_recommendation"] = np.where(
    out["irrigation_prob"] >= IRRIG_TH, "Plan irrigation soon", "No action"
)

# Optional: sort by location/date for readability
out = out.sort_values(["latitude", "longitude", "date"])

# -----------------------
# 6) Save outputs
# -----------------------
out.to_csv("recommendations.csv", index=False)

# A tiny on-screen preview
print("\nSaved recommendations.csv")
print(out.groupby(["latitude","longitude"]).tail(5))
