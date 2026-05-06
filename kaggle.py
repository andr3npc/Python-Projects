"""
Kaggle Playground Series S6E5 - Predicting F1 Pit Stops
========================================================
OOF AUC ~0.947+ (quick run), expected 0.95+ with full settings below.

Usage (VSCode):
  1. pip install pandas numpy scikit-learn lightgbm xgboost
  2. Place train.csv, test.csv, sample_submission.csv in same directory
  3. python f1_pitstop_prediction.py
  4. Submit submission.csv to Kaggle
"""

import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
N_FOLDS = 5
SEED = 42
DATA_DIR = "."

# ============================================================
# 1. LOAD DATA
# ============================================================
train = pd.read_csv(f"{DATA_DIR}/train.csv")
test = pd.read_csv(f"{DATA_DIR}/test.csv")
sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")
TARGET = "PitNextLap"

print(f"Train: {train.shape}, Test: {test.shape}")
print(f"Target split: {train[TARGET].value_counts(normalize=True).to_dict()}")

# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================
def feature_engineering(df):
    df = df.copy()

    # --- Compound encoding ---
    compound_ord = {"SOFT": 0, "MEDIUM": 1, "HARD": 2, "INTERMEDIATE": 3, "WET": 4}
    df["Compound_ord"] = df["Compound"].map(compound_ord)

    # --- Tyre life (most critical features) ---
    df["TyreLife_sq"] = df["TyreLife"] ** 2
    df["TyreLife_log"] = np.log1p(df["TyreLife"])
    df["TyreLife_cubed"] = df["TyreLife"] ** 3

    # Approximate normalized tyre life (the removed feature!)
    max_life = {"SOFT": 25, "MEDIUM": 35, "HARD": 45, "INTERMEDIATE": 30, "WET": 40}
    df["Expected_MaxLife"] = df["Compound"].map(max_life)
    df["TyreLife_pct"] = df["TyreLife"] / df["Expected_MaxLife"]
    df["TyreLife_pct_sq"] = df["TyreLife_pct"] ** 2
    df["TyreLife_remaining"] = df["Expected_MaxLife"] - df["TyreLife"]
    df["TyreLife_remaining_pct"] = df["TyreLife_remaining"] / df["Expected_MaxLife"]

    # --- Stint ---
    df["Stint_sq"] = df["Stint"] ** 2
    df["TyreLife_per_stint"] = df["TyreLife"] / df["Stint"]
    df["IsFirstStint"] = (df["Stint"] == 1).astype(int)
    df["IsLateStint"] = (df["Stint"] >= 3).astype(int)

    # --- Race progress / lap ---
    df["LapNumber_sq"] = df["LapNumber"] ** 2
    df["LapNumber_log"] = np.log1p(df["LapNumber"])
    df["RaceProgress_sq"] = df["RaceProgress"] ** 2
    df["RaceProgress_cubed"] = df["RaceProgress"] ** 3
    df["Est_TotalLaps"] = (df["LapNumber"] / df["RaceProgress"].clip(lower=0.01)).clip(upper=100)
    df["Est_LapsRemaining"] = (df["Est_TotalLaps"] - df["LapNumber"]).clip(lower=0)
    df["PitWindow_ratio"] = df["Est_LapsRemaining"] / df["Expected_MaxLife"]

    # --- Interactions: tyre x race ---
    df["TyreLife_x_Progress"] = df["TyreLife"] * df["RaceProgress"]
    df["TyreLife_x_LapsRemaining"] = df["TyreLife"] * df["Est_LapsRemaining"]

    # --- Position ---
    df["Position_sq"] = df["Position"] ** 2
    df["Position_log"] = np.log1p(df["Position"])
    df["IsLeader"] = (df["Position"] == 1).astype(int)
    df["IsTop3"] = (df["Position"] <= 3).astype(int)
    df["IsTop10"] = (df["Position"] <= 10).astype(int)
    df["IsBackmarker"] = (df["Position"] >= 15).astype(int)
    df["Position_x_TyreLife"] = df["Position"] * df["TyreLife"]
    df["Position_x_Progress"] = df["Position"] * df["RaceProgress"]
    df["Position_Change_abs"] = df["Position_Change"].abs()
    df["IsLosingPositions"] = (df["Position_Change"] < 0).astype(int)
    df["IsGainingPositions"] = (df["Position_Change"] > 0).astype(int)

    # --- Lap time ---
    df["LapTime_log"] = np.log1p(df["LapTime (s)"].clip(lower=0))
    df["LapTime_Delta_abs"] = df["LapTime_Delta"].abs()
    df["IsSlowLap"] = (df["LapTime_Delta"] > 2).astype(int)
    df["IsFastLap"] = (df["LapTime_Delta"] < -1).astype(int)

    # --- Degradation ---
    df["Degradation_per_lap"] = df["Cumulative_Degradation"] / df["TyreLife"].clip(lower=1)
    df["Degradation_abs"] = df["Cumulative_Degradation"].abs()
    df["Degradation_sq"] = df["Cumulative_Degradation"] ** 2
    df["HighDegradation"] = (df["Cumulative_Degradation"] < -50).astype(int)
    df["Degradation_x_Progress"] = df["Cumulative_Degradation"] * df["RaceProgress"]
    df["Degradation_x_TyreLife"] = df["Cumulative_Degradation"] * df["TyreLife"]

    # --- PitStop / Compound interactions ---
    df["PitStop_x_TyreLife"] = df["PitStop"] * df["TyreLife"]
    df["Compound_x_TyreLife"] = df["Compound_ord"] * df["TyreLife"]
    df["Compound_x_Progress"] = df["Compound_ord"] * df["RaceProgress"]
    df["Compound_x_Degradation"] = df["Compound_ord"] * df["Cumulative_Degradation"]

    # --- Year ---
    df["Year_offset"] = df["Year"] - 2022

    return df


train = feature_engineering(train)
test = feature_engineering(test)

# ============================================================
# 3. ENCODE CATEGORICALS
# ============================================================
for col in ["Driver", "Compound", "Race"]:
    le = LabelEncoder()
    le.fit(pd.concat([train[col], test[col]]).astype(str))
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

# ============================================================
# 4. PREPARE
# ============================================================
features = [c for c in train.columns if c not in ["id", TARGET]]
X = train[features].values.astype(np.float32)
y = train[TARGET].values.astype(np.float32)
X_test = test[features].values.astype(np.float32)
print(f"Features: {len(features)}")

# ============================================================
# 5. TRAIN: LightGBM + XGBoost, 5-fold CV
# ============================================================
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof_lgb = np.zeros(len(X))
oof_xgb = np.zeros(len(X))
test_lgb = np.zeros(len(X_test))
test_xgb = np.zeros(len(X_test))

lgb_params = dict(
    n_estimators=3000, learning_rate=0.02, max_depth=8, num_leaves=100,
    min_child_samples=50, subsample=0.8, colsample_bytree=0.6,
    reg_alpha=0.1, reg_lambda=1.0, random_state=SEED, n_jobs=-1, verbose=-1,
)
xgb_params = dict(
    n_estimators=3000, learning_rate=0.02, max_depth=8, min_child_weight=50,
    subsample=0.8, colsample_bytree=0.6, reg_alpha=0.1, reg_lambda=1.0,
    random_state=SEED, n_jobs=-1, verbosity=0, eval_metric="auc",
    tree_method="hist", scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
)

print("\n" + "=" * 50)
for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    print(f"Fold {fold+1}/{N_FOLDS}")
    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    # LightGBM
    lgb = LGBMClassifier(**lgb_params)
    lgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
            callbacks=[__import__("lightgbm").early_stopping(150, verbose=False),
                       __import__("lightgbm").log_evaluation(0)])
    oof_lgb[va_idx] = lgb.predict_proba(X_va)[:, 1]
    test_lgb += lgb.predict_proba(X_test)[:, 1] / N_FOLDS
    print(f"  LGB AUC: {roc_auc_score(y_va, oof_lgb[va_idx]):.5f}")

    # XGBoost
    xgb = XGBClassifier(**xgb_params)
    xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_xgb[va_idx] = xgb.predict_proba(X_va)[:, 1]
    test_xgb += xgb.predict_proba(X_test)[:, 1] / N_FOLDS
    print(f"  XGB AUC: {roc_auc_score(y_va, oof_xgb[va_idx]):.5f}")

# ============================================================
# 6. ENSEMBLE
# ============================================================
print("\n" + "=" * 50)
lgb_auc = roc_auc_score(y, oof_lgb)
xgb_auc = roc_auc_score(y, oof_xgb)
print(f"LGB OOF AUC: {lgb_auc:.5f}")
print(f"XGB OOF AUC: {xgb_auc:.5f}")

w_lgb = lgb_auc / (lgb_auc + xgb_auc)
w_xgb = xgb_auc / (lgb_auc + xgb_auc)
oof_ens = w_lgb * oof_lgb + w_xgb * oof_xgb
test_ens = w_lgb * test_lgb + w_xgb * test_xgb
print(f"ENS OOF AUC: {roc_auc_score(y, oof_ens):.5f}  (LGB={w_lgb:.3f}, XGB={w_xgb:.3f})")

# ============================================================
# 7. FEATURE IMPORTANCE
# ============================================================
print("\nTop 25 features (LightGBM last fold):")
fi = pd.DataFrame({"feature": features, "imp": lgb.feature_importances_})
print(fi.sort_values("imp", ascending=False).head(25).to_string(index=False))

# ============================================================
# 8. SAVE
# ============================================================
sub[TARGET] = test_ens
sub.to_csv("submission.csv", index=False)
sub_lgb = sub.copy(); sub_lgb[TARGET] = test_lgb; sub_lgb.to_csv("submission_lgb.csv", index=False)
sub_xgb = sub.copy(); sub_xgb[TARGET] = test_xgb; sub_xgb.to_csv("submission_xgb.csv", index=False)

print(f"\nSaved: submission.csv, submission_lgb.csv, submission_xgb.csv")
print(f"Prediction stats:\n{sub[TARGET].describe()}")
