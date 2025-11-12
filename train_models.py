# train_models.py
import numpy as np
import pandas as pd
from date_utils import parse_date_col

from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

from time_splits import year_group_splits

# -----------------------
# 1) Load labeled data
# -----------------------
df = pd.read_csv("labeled_features.csv")
df["date"] = parse_date_col(df["date"])

# Identify groups (location) for grouped CV so we don't leak between same field
groups = df[["latitude", "longitude"]].astype(str).agg("_".join, axis=1)

# Columns to exclude from features
exclude = {"date", "latitude", "longitude", "planting_window", "irrigation_flag"}
feature_cols = [c for c in df.columns if c not in exclude]

# === Diagnostics: label prevalence & NaNs that might be dropping positives ===
X_raw = df[feature_cols].copy().replace([np.inf, -np.inf], np.nan)

print("\nTop columns with NaNs (before any dropping):")
print(X_raw.isna().sum().sort_values(ascending=False).head(20))

print("\nLabel prevalence BEFORE dropping NaNs:")
print("planting:", df["planting_window"].value_counts().to_dict())
print("irrigation:", df["irrigation_flag"].value_counts().to_dict())

valid_mask = ~X_raw.isna().any(axis=1)
print(f"\nRows total: {len(X_raw)} | rows kept if we dropna(): {int(valid_mask.sum())}")

print("\nLabel prevalence among rows that would be kept by dropna():")
print("planting:", df.loc[valid_mask, "planting_window"].value_counts().to_dict())
print("irrigation:", df.loc[valid_mask, "irrigation_flag"].value_counts().to_dict())

# === Build feature matrix (numeric only); impute inside pipelines ===
X = df[feature_cols].replace([np.inf, -np.inf], np.nan).select_dtypes(include=["number"])
valid_idx = X.index
y_plant = df.loc[valid_idx, "planting_window"].astype(int)
y_irrig = df.loc[valid_idx, "irrigation_flag"].astype(int)
groups_valid = groups.loc[valid_idx]
dates_valid  = df.loc[valid_idx, ["date"]].copy()

print("\nLabel prevalence (overall):")
print("planting_window:", y_plant.value_counts(dropna=False).to_dict())
print("irrigation_flag:", y_irrig.value_counts(dropna=False).to_dict())

yr = dates_valid["date"].dt.year
print("\nLabel prevalence by year (planting):")
print(pd.crosstab(yr, y_plant))
print("\nLabel prevalence by year (irrigation):")
print(pd.crosstab(yr, y_irrig))

print(f"\nUsing {len(feature_cols)} features:\n{feature_cols}\n")
print(f"Training rows: {len(valid_idx)}; Locations: {groups_valid.nunique()}")

# -----------------------
# 2) Define candidate models (with imputation)
# -----------------------
imputer = SimpleImputer(strategy="median")

pipe_logreg = Pipeline([
    ("imputer", imputer),
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
])

pipe_rf = Pipeline([
    ("imputer", imputer),
    ("clf", RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=2,
        class_weight="balanced_subsample", random_state=42
    ))
])

models = {"logreg": pipe_logreg, "rf": pipe_rf}

# -----------------------
# 3) Evaluation helpers (robust to single-class folds)
# -----------------------
def _safe_scores(y_true, proba, pred):
    auc = None
    if len(set(y_true)) > 1:
        auc = roc_auc_score(y_true, proba)
    f1 = f1_score(y_true, pred, zero_division=0)
    return auc, f1

def evaluate_label_groupcv(y, label_name):
    print(f"\n=== Grouped K-Fold: {label_name} ===")
    gkf = GroupKFold(n_splits=min(5, groups_valid.nunique()))
    results = {}
    for name, mdl in models.items():
        aucs, f1s = [], []
        for tr, te in gkf.split(X, y, groups_valid):
            y_tr, y_te = y.iloc[tr], y.iloc[te]
            if len(set(y_tr)) < 2 or len(set(y_te)) < 2:
                print(f"  [skip fold: single-class] {name}")
                continue
            mdl.fit(X.iloc[tr], y_tr)
            proba = mdl.predict_proba(X.iloc[te])[:, 1]
            pred  = (proba >= 0.5).astype(int)
            auc, f1 = _safe_scores(y_te, proba, pred)
            if auc is not None: aucs.append(auc)
            f1s.append(f1)
        a_m = np.mean(aucs) if aucs else float("nan")
        a_s = np.std(aucs)  if aucs else float("nan")
        f_m = np.mean(f1s) if f1s else 0.0
        f_s = np.std(f1s)  if f1s else 0.0
        results[name] = (a_m, a_s, f_m, f_s)
        print(f"{name:>7}  ROC-AUC: {a_m:.3f} ± {a_s:.3f} | F1: {f_m:.3f} ± {f_s:.3f}")
    best = max(results, key=lambda k: (results[k][0] if not np.isnan(results[k][0]) else -1))
    print(f"→ Best (grouped) for {label_name}: {best}")
    return best, models[best]

def evaluate_label_timeaware(y, label_name):
    print(f"\n=== Time-aware backtest: {label_name} ===")
    results = {}
    for name, mdl in models.items():
        aucs, f1s = [], []
        for tr_idx, te_idx in year_group_splits(dates_valid):
            y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
            if len(set(y_tr)) < 2 or len(set(y_te)) < 2:
                print(f"  [skip fold: single-class] {name}")
                continue
            mdl.fit(X.iloc[tr_idx], y_tr)
            proba = mdl.predict_proba(X.iloc[te_idx])[:, 1]
            pred  = (proba >= 0.5).astype(int)
            auc, f1 = _safe_scores(y_te, proba, pred)
            if auc is not None: aucs.append(auc)
            f1s.append(f1)
        a_m = np.mean(aucs) if aucs else float("nan")
        a_s = np.std(aucs)  if aucs else float("nan")
        f_m = np.mean(f1s) if f1s else 0.0
        f_s = np.std(f1s)  if f1s else 0.0
        results[name] = (a_m, a_s, f_m, f_s)
        print(f"{name:>7}  ROC-AUC: {a_m:.3f} ± {a_s:.3f} | F1: {f_m:.3f} ± {f_s:.3f}")
    best = max(results, key=lambda k: (results[k][0] if not np.isnan(results[k][0]) else -1))
    print(f"→ Best (time-aware) for {label_name}: {best}")
    return best, models[best]

# -----------------------
# 4) Evaluate & choose models
# -----------------------
# Planting (both classes present)
_, best_plant_model = evaluate_label_groupcv(y_plant, "planting_window")
_, best_plant_model = evaluate_label_timeaware(y_plant, "planting_window")

# Irrigation — only evaluate if two classes exist overall
if len(np.unique(y_irrig)) < 2:
    print("\n⚠️ Irrigation label has a single class overall; skipping irrigation evaluation/training.")
    train_irrigation = False
    best_irrig_model = None
else:
    train_irrigation = True
    _, best_irrig_model = evaluate_label_groupcv(y_irrig, "irrigation_flag")
    _, best_irrig_model = evaluate_label_timeaware(y_irrig, "irrigation_flag")

# -----------------------
# 5) Fit best models on ALL data
# -----------------------
best_plant_model.fit(X, y_plant)
if train_irrigation:
    best_irrig_model.fit(X, y_irrig)

# -----------------------
# 6) Make probabilities & actions
# -----------------------
plant_prob = pd.Series(best_plant_model.predict_proba(X)[:, 1], index=valid_idx, name="planting_prob")
if train_irrigation:
    irrig_prob_vals = best_irrig_model.predict_proba(X)[:, 1]
else:
    irrig_prob_vals = np.zeros(len(valid_idx))

irrig_prob = pd.Series(irrig_prob_vals, index=valid_idx, name="irrigation_prob")

out = df.loc[valid_idx, ["date", "latitude", "longitude"]].copy()
out["planting_prob"] = plant_prob.values
out["irrigation_prob"] = irrig_prob.values

PLANT_TH = 0.60
IRRIG_TH = 0.60

out["planting_recommendation"]  = np.where(out["planting_prob"]  >= PLANT_TH, "Consider planting this week", "Hold")
out["irrigation_recommendation"] = np.where(out["irrigation_prob"] >= IRRIG_TH, "Plan irrigation soon", "No action")

out = out.sort_values(["latitude", "longitude", "date"])
out.to_csv("recommendations.csv", index=False)

print("\nSaved recommendations.csv")
print(out.groupby(["latitude","longitude"]).tail(5))

# --- Quick sanity checks ---
print("\nTop planting days by probability:")
print(out.sort_values("planting_prob", ascending=False).head(10))

print("\nTop irrigation days by probability:")
print(out.sort_values("irrigation_prob", ascending=False).head(10))

# How many recommendations at current thresholds?
print("\nCounts at thresholds:")
print(out["planting_recommendation"].value_counts())
print(out["irrigation_recommendation"].value_counts())

# Month-level signal (should light up spring for planting)
tmp = out.copy()
tmp["month"] = tmp["date"].dt.month
print("\nMonthly mean probs (planting):")
print(tmp.groupby("month")["planting_prob"].mean().round(3))
print("\nMonthly mean probs (irrigation):")
print(tmp.groupby("month")["irrigation_prob"].mean().round(3))

