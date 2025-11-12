# calibrate_models.py
import pandas as pd, numpy as np, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from date_utils import parse_date_col

# Load labels/features
df = pd.read_csv("labeled_features.csv")
df["date"] = parse_date_col(df["date"])

# Features/targets
exclude = {"date","latitude","longitude","planting_window","irrigation_flag"}
X = df[[c for c in df.columns if c not in exclude]].replace([np.inf,-np.inf], np.nan).dropna()
idx = X.index
yP = df.loc[idx, "planting_window"].astype(int)
yI = df.loc[idx, "irrigation_flag"].astype(int)

# Base classifier (same flavor as your RF)
base_rf = RandomForestClassifier(
    n_estimators=300, max_depth=None, min_samples_leaf=2,
    class_weight="balanced_subsample", random_state=42
)

# Calibrate with isotonic (better shape than sigmoid for tabular)
plant_cal = CalibratedClassifierCV(base_rf, method="isotonic", cv=5).fit(X, yP)
irrig_cal = CalibratedClassifierCV(base_rf, method="isotonic", cv=5).fit(X, yI)

joblib.dump(plant_cal, "model_planting_calibrated.joblib")
joblib.dump(irrig_cal, "model_irrigation_calibrated.joblib")
print("✅ Saved: model_planting_calibrated.joblib, model_irrigation_calibrated.joblib")
