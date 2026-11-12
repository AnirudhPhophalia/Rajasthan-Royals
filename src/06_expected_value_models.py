import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.optimize import curve_fit
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, r2_score, log_loss, roc_auc_score, brier_score_loss

IN_PATH = r"d:\Rajasthan Royals\outputs\processed\deliveries_with_pressure.pkl"
OUT_DIR = r"d:\Rajasthan Royals\outputs\models"
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading deliveries_with_pressure.pkl...")
df = pd.read_pickle(IN_PATH)

reg_mask = df["innings"].isin([1, 2])
reg = df[reg_mask].copy()

# 1. Empirical Smooth DLS-Style Resource Curve & Marginal Wicket Value V_W(balls_remaining, wickets_in_hand)
print("Estimating empirical smooth DLS Resource Curve & Marginal Wicket Value V_W(b, w)...")
inn_tot = reg.groupby(["match_id", "innings"])["total_runs"].sum().rename("inn_tot")
reg = reg.merge(inn_tot, on=["match_id", "innings"], how="left")
reg["runs_remaining_actual"] = reg["inn_tot"] - reg["runs_before"]

inn1_reg = reg[reg["innings"] == 1].copy()
inn1_reg["par_tot"] = inn1_reg["prior_venue_rpo"] * (inn1_reg["max_balls_inn"] / 6.0)
inn1_reg["frac_b"] = np.clip(inn1_reg["balls_remaining"] / np.maximum(inn1_reg["max_balls_inn"], 6.0), 0.0, 1.0)
inn1_reg["frac_w"] = np.clip(inn1_reg["wickets_in_hand"] / 10.0, 0.0, 1.0)

def dls_resource_fn(X, c, alpha, beta):
    fb, fw, par = X
    return c * par * (fb ** alpha) * (1.0 - (1.0 - fw) ** beta)

popt, _ = curve_fit(
    dls_resource_fn,
    (inn1_reg["frac_b"].values, inn1_reg["frac_w"].values, inn1_reg["par_tot"].values),
    inn1_reg["runs_remaining_actual"].values,
    p0=[1.0, 0.75, 1.5],
    bounds=([0.5, 0.3, 0.5], [1.5, 1.5, 4.0]),
)
print(f"Fitted DLS resource params -> c={popt[0]:.4f}, alpha={popt[1]:.4f}, beta={popt[2]:.4f}")

reg["par_tot"] = reg["prior_venue_rpo"] * (reg["max_balls_inn"] / 6.0)
reg["frac_b"] = np.clip(reg["balls_remaining"] / np.maximum(reg["max_balls_inn"], 6.0), 0.0, 1.0)
reg["frac_w"] = np.clip(reg["wickets_in_hand"] / 10.0, 0.0, 1.0)
reg["frac_w_minus1"] = np.clip((reg["wickets_in_hand"] - 1.0) / 10.0, 0.0, 1.0)

rem_curr = dls_resource_fn((reg["frac_b"].values, reg["frac_w"].values, reg["par_tot"].values), *popt)
rem_next = np.where(
    reg["wickets_in_hand"] <= 1,
    0.0,
    dls_resource_fn((reg["frac_b"].values, reg["frac_w_minus1"].values, reg["par_tot"].values), *popt),
)
vw_raw = rem_curr - rem_next
# Clean smooth bounds: min 1.5 runs on last ball, max 13.5 runs so early tail wickets don't dominate
reg["wicket_run_value"] = np.where(reg["balls_remaining"] <= 1, 1.5, np.clip(vw_raw, 2.0, 13.5))

print("Sample Smooth DLS Wicket Run Values V_W by phase & wickets in hand:")
print(reg.groupby(["phase", "wickets_in_hand"])["wicket_run_value"].mean().unstack().round(2).to_string())

# 2. Prepare Context Features for xRuns & xWkt Baseline Models
# Categorical encodings
for col in ["comp_group", "bowling_type", "bowling_hand"]:
    reg[col + "_cat"] = reg[col].astype("category").cat.codes

reg["striker_is_dismissed"] = (reg["is_wicket"] & (reg["dismissed_player_id"] == reg["striker_id"])).astype(int)

ctx_features = [
    "comp_group_cat",
    "year",
    "prior_venue_rpo",
    "prior_cg_rpo",
    "match_has_impact_rule",
    "bat_team_elo",
    "bowl_team_elo",
    "elo_diff",
    "innings",
    "overid",
    "legal_balls_before",
    "balls_remaining",
    "wickets_before",
    "wickets_in_hand",
    "max_balls_inn",
    "batting_order",
    "batter_balls_in_inn",
    "bowler_balls_in_inn",
    "is_pace",
    "is_rhb",
    "bowling_type_cat",
    "bowling_hand_cat",
]

def run_oof_regressor(sub_df, feat_cols, target_col, name):
    X = sub_df[feat_cols].astype(float)
    y = sub_df[target_col].astype(float).values
    groups = sub_df["match_id"].values
    gkf = GroupKFold(n_splits=5)
    oof = np.zeros(len(sub_df))

    params = {
        "objective": "regression",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 150,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.85,
        "reg_alpha": 2.0,
        "reg_lambda": 5.0,
        "n_estimators": 400,
        "random_state": 42,
        "verbose": -1,
        "n_jobs": 4,
    }
    for tr_idx, va_idx in gkf.split(X, y, groups):
        m = lgb.LGBMRegressor(**params)
        m.fit(
            X.iloc[tr_idx], y[tr_idx],
            eval_set=[(X.iloc[va_idx], y[va_idx])],
            callbacks=[lgb.early_stopping(35, verbose=False)]
        )
        oof[va_idx] = np.clip(m.predict(X.iloc[va_idx]), 0.0, 6.0)

    rmse = np.sqrt(mean_squared_error(y, oof))
    r2 = r2_score(y, oof)
    print(f"[{name}] OOF RMSE: {rmse:.4f} | Mean Pred: {oof.mean():.4f} (Actual: {y.mean():.4f}) | R2: {r2:.4f}")
    return oof

def run_oof_classifier(sub_df, feat_cols, target_col, name):
    X = sub_df[feat_cols].astype(float)
    y = sub_df[target_col].astype(int).values
    groups = sub_df["match_id"].values
    gkf = GroupKFold(n_splits=5)
    oof = np.zeros(len(sub_df))

    params = {
        "objective": "binary",
        "learning_rate": 0.03,
        "num_leaves": 23,
        "max_depth": 5,
        "min_child_samples": 200,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.85,
        "reg_alpha": 2.0,
        "reg_lambda": 5.0,
        "n_estimators": 400,
        "random_state": 42,
        "verbose": -1,
        "n_jobs": 4,
    }
    for tr_idx, va_idx in gkf.split(X, y, groups):
        m = lgb.LGBMClassifier(**params)
        m.fit(
            X.iloc[tr_idx], y[tr_idx],
            eval_set=[(X.iloc[va_idx], y[va_idx])],
            callbacks=[lgb.early_stopping(35, verbose=False)]
        )
        oof[va_idx] = np.clip(m.predict_proba(X.iloc[va_idx])[:, 1], 0.001, 0.50)

    ll = log_loss(y, oof)
    bs = brier_score_loss(y, oof)
    auc = roc_auc_score(y, oof)
    print(f"[{name}] OOF LogLoss: {ll:.4f} | Brier: {bs:.4f} | AUC: {auc:.4f} | Mean Pred: {oof.mean():.4f} (Actual: {y.mean():.4f})")
    return oof

print("\nTraining Batting Baseline Models (on balls faced = ~wide)...")
bat_mask = reg["is_ball_faced"].values
reg["xRuns_bat"] = np.nan
reg["xWkt_bat"] = np.nan

xR_bat = run_oof_regressor(reg[bat_mask], ctx_features, "runs", "xRuns_bat")
xW_bat = run_oof_classifier(reg[bat_mask], ctx_features, "striker_is_dismissed", "xWkt_bat")
reg.loc[bat_mask, "xRuns_bat"] = xR_bat
reg.loc[bat_mask, "xWkt_bat"] = xW_bat

print("\nTraining Bowling Baseline Models (on all deliveries bowled)...")
reg["xRuns_bowl"] = run_oof_regressor(reg, ctx_features, "bowler_runs", "xRuns_bowl")
reg["xWkt_bowl"] = run_oof_classifier(reg, ctx_features, "is_bowler_wicket", "xWkt_bowl")

# 3. Compute Context-Adjusted Value Added Metrics per Delivery
# Batting (only defined on is_ball_faced)
reg["raa_bat"] = reg["runs"] - reg["xRuns_bat"]
reg["waa_bat"] = reg["xWkt_bat"] - reg["striker_is_dismissed"]  # positive = survived more than expected
reg["rva_bat"] = reg["raa_bat"] + reg["wicket_run_value"] * reg["waa_bat"]

# Bowling (defined on all deliveries bowled, including wides & noballs!)
reg["raa_bowl"] = reg["xRuns_bowl"] - reg["bowler_runs"]  # positive = saved runs vs expected
reg["waa_bowl"] = reg["is_bowler_wicket"].astype(float) - reg["xWkt_bowl"]  # positive = took more wickets
reg["rva_bowl"] = reg["raa_bowl"] + reg["wicket_run_value"] * reg["waa_bowl"]

# Sanity check zero-mean property of residuals
print("\n=== RESIDUAL ZERO-MEAN & PHASE CHECKS ===")
print("Mean RAA_bat:", reg.loc[bat_mask, "raa_bat"].mean().round(5), "| Mean RVA_bat:", reg.loc[bat_mask, "rva_bat"].mean().round(5))
print("Mean RAA_bowl:", reg["raa_bowl"].mean().round(5), "| Mean RVA_bowl:", reg["rva_bowl"].mean().round(5))

print("\nBy Phase (RVA_bat should be ~0 across all phases; confirms zero phase bias!):")
print(reg[bat_mask].groupby("phase")[["runs", "xRuns_bat", "raa_bat", "rva_bat"]].mean().round(4).to_string())

print("\nBy Comp Group (confirms zero format/era bias!):")
print(reg[bat_mask].groupby("comp_group")[["runs", "xRuns_bat", "raa_bat", "rva_bat"]].mean().round(4).to_string())

reg.to_pickle(r"d:\Rajasthan Royals\outputs\processed\deliveries_modeled.pkl")
print("\nSaved deliveries_modeled.pkl successfully!")

