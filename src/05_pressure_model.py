import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from sklearn.isotonic import IsotonicRegression

IN_PATH = r"d:\Rajasthan Royals\outputs\processed\deliveries.pkl"
OUT_DIR = r"d:\Rajasthan Royals\outputs\pressure"
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading deliveries.pkl...")
df = pd.read_pickle(IN_PATH)

# Sanity check rolling features
assert (df["runs_last_12"] >= 0).all(), "Negative runs_last_12!"
assert (df["wkts_last_12"] >= 0).all(), "Negative wkts_last_12!"

# Work on Innings 1 and 2 for WP modeling (38 Super Over deliveries handled separately)
reg_mask = df["innings"].isin([1, 2])
decisive_mask = reg_mask & df["bat_team_won"].notna()

# Build rich feature sets for Innings 1 and Innings 2 WP models
# Monotonicity constraints ensure counterfactual perturbations (runs+4, wkt+1) behave smoothly!
inn1_features = [
    "runs_before",           # +1 (more runs -> higher WP)
    "wickets_before",        # -1 (more wickets lost -> lower WP)
    "balls_remaining",       # +1 (more balls left at same score? wait: at same runs_before, more balls_remaining means earlier in innings -> higher projected final score!)
    "legal_balls_before",    # 0
    "crr_before",            # 0
    "prior_venue_rpo",       # 0
    "prior_cg_rpo",          # 0
    "elo_diff",              # +1
    "bat_team_elo",          # 0
    "bowl_team_elo",         # 0
    "runs_last_12",          # 0
    "wkts_last_12",          # 0
    "match_has_impact_rule", # 0
    "max_balls_inn",         # 0
]
inn1_mono = [1, -1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]

inn2_features = [
    "runs_needed",           # -1 (more runs needed -> lower WP)
    "wickets_before",        # -1 (more wickets lost -> lower WP)
    "balls_remaining",       # +1 (more balls remaining at same runs_needed -> higher WP)
    "rrr_before",            # -1 (higher RRR -> lower WP)
    "rrr_minus_crr",         # 0
    "rrr_ratio_par",         # 0
    "crr_before",            # 0
    "target",                # 0
    "prior_venue_rpo",       # 0
    "elo_diff",              # +1
    "bat_team_elo",          # 0
    "bowl_team_elo",         # 0
    "runs_last_12",          # 0
    "wkts_last_12",          # 0
    "match_has_impact_rule", # 0
    "max_balls_inn",         # 0
]
inn2_mono = [-1, -1, 1, -1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]

def train_oof_wp(sub_df, feat_cols, mono_constraints, label="Inn1"):
    train_df = sub_df[sub_df["bat_team_won"].notna()].copy()
    X = train_df[feat_cols].astype(float)
    y = train_df["bat_team_won"].astype(int).values
    groups = train_df["match_id"].values

    gkf = GroupKFold(n_splits=5)
    oof_raw = np.zeros(len(train_df))
    models = []

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.03,
        "num_leaves": 23,
        "max_depth": 5,
        "min_child_samples": 250,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.85,
        "reg_alpha": 2.0,
        "reg_lambda": 5.0,
        "monotone_constraints": mono_constraints,
        "n_estimators": 450,
        "random_state": 42,
        "verbose": -1,
        "n_jobs": 4,
    }

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups)):
        m = lgb.LGBMClassifier(**params)
        m.fit(
            X.iloc[tr_idx], y[tr_idx],
            eval_set=[(X.iloc[va_idx], y[va_idx])],
            callbacks=[lgb.early_stopping(40, verbose=False)]
        )
        oof_raw[va_idx] = m.predict_proba(X.iloc[va_idx])[:, 1]
        models.append(m)

    # Out-of-fold Isotonic Calibration (nested per fold so zero leakage)
    oof_cal = np.zeros(len(train_df))
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups)):
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(oof_raw[tr_idx], y[tr_idx])
        oof_cal[va_idx] = iso.predict(oof_raw[va_idx])

    # Fit full isotonic calibrator for counterfactual / non-decisive prediction
    full_iso = IsotonicRegression(out_of_bounds="clip")
    full_iso.fit(oof_raw, y)

    ll = log_loss(y, oof_cal)
    bs = brier_score_loss(y, oof_cal)
    auc = roc_auc_score(y, oof_cal)
    # Expected Calibration Error (10 bins)
    bins = np.linspace(0, 1, 11)
    ece = 0.0
    for i in range(10):
        mask = (oof_cal >= bins[i]) & (oof_cal < bins[i+1] if i < 9 else oof_cal <= bins[i+1])
        if mask.sum() > 0:
            ece += (mask.sum() / len(y)) * abs(y[mask].mean() - oof_cal[mask].mean())

    print(f"[{label}] 5-Fold GroupKFold OOF -> LogLoss: {ll:.4f} | Brier: {bs:.4f} | AUC: {auc:.4f} | ECE: {ece:.4f}")

    # Predict for ALL rows in sub_df (using OOF for decisive rows, fold-average for non-decisive rows)
    preds = np.zeros(len(sub_df))
    dec_idx = sub_df["bat_team_won"].notna().values
    preds[dec_idx] = oof_cal
    if (~dec_idx).sum() > 0:
        X_nd = sub_df.loc[~dec_idx, feat_cols].astype(float)
        raw_nd = np.mean([m.predict_proba(X_nd)[:, 1] for m in models], axis=0)
        preds[~dec_idx] = full_iso.predict(raw_nd)

    return preds, models, full_iso, {"logloss": ll, "brier": bs, "auc": auc, "ece": ece}

print("\nTraining Innings 1 & Innings 2 Win Probability Models...")
inn1_mask = df["innings"] == 1
inn2_mask = df["innings"] == 2

wp1, models1, iso1, metrics1 = train_oof_wp(df[inn1_mask], inn1_features, inn1_mono, "Inn1")
wp2, models2, iso2, metrics2 = train_oof_wp(df[inn2_mask], inn2_features, inn2_mono, "Inn2")

df["wp_before"] = np.nan
df.loc[inn1_mask, "wp_before"] = np.clip(wp1, 0.001, 0.999)
df.loc[inn2_mask, "wp_before"] = np.clip(wp2, 0.001, 0.999)
# Super over (innings 3 & 4): inherently 50-50 extreme climax!
df.loc[df["innings"] >= 3, "wp_before"] = 0.50

# Compute Counterfactual Leverage Index (LI) using ensemble average + full_iso
# Perturb current state by: Dot (0r, 0w, -1b), Single (1r, 0w, -1b), Four (4r, 0w, -1b), Six (6r, 0w, -1b), Wicket (0r, +1w, -1b)
def predict_wp_batch(sub_df, feat_cols, models, iso):
    X = sub_df[feat_cols].astype(float)
    raw = np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)
    return np.clip(iso.predict(raw), 0.001, 0.999)

def compute_counterfactuals(sub_df, is_inn2):
    feat_cols = inn2_features if is_inn2 else inn1_features
    models = models2 if is_inn2 else models1
    iso = iso2 if is_inn2 else iso1

    base_wp = predict_wp_batch(sub_df, feat_cols, models, iso)
    out = {}
    for name, dr, dw in [("0", 0, 0), ("1", 1, 0), ("4", 4, 0), ("6", 6, 0), ("W", 0, 1)]:
        cf = sub_df.copy()
        cf["balls_remaining"] = np.maximum(cf["balls_remaining"] - 1.0, 0.0)
        cf["legal_balls_before"] = cf["legal_balls_before"] + 1.0
        cf["wickets_before"] = np.minimum(cf["wickets_before"] + dw, 10.0)
        if is_inn2:
            cf["runs_needed"] = np.maximum(cf["runs_needed"] - dr, 0.0)
            cf["runs_before"] = cf["runs_before"] + dr
            cf["crr_before"] = cf["runs_before"] / np.maximum(cf["legal_balls_before"] / 6.0, 1/6)
            cf["rrr_before"] = np.where(
                cf["balls_remaining"] > 0,
                np.clip(cf["runs_needed"] / (cf["balls_remaining"] / 6.0), 0.0, 36.0),
                np.where(cf["runs_needed"] > 0, 36.0, 0.0)
            )
            cf["rrr_minus_crr"] = cf["rrr_before"] - cf["crr_before"]
            cf["rrr_ratio_par"] = cf["rrr_before"] / np.maximum(cf["prior_venue_rpo"], 5.0)
            wp_cf = predict_wp_batch(cf, feat_cols, models, iso)
            # Terminal rules in Inn 2:
            wp_cf = np.where(cf["runs_needed"] <= 0, 0.999,
                    np.where((cf["balls_remaining"] <= 0) | (cf["wickets_before"] >= 10), 0.001, wp_cf))
        else:
            cf["runs_before"] = cf["runs_before"] + dr
            cf["crr_before"] = cf["runs_before"] / np.maximum(cf["legal_balls_before"] / 6.0, 1/6)
            wp_cf = predict_wp_batch(cf, feat_cols, models, iso)
            # If 10 wickets lost in Inn 1, score capped
        out[name] = wp_cf

    # Expected |Delta WP| per ball using empirical T20 outcome distribution:
    # ~38% dot, ~39% 1-3 runs (approx 1), ~12% 4, ~5.5% 6, ~5.5% wicket
    exp_abs_swing = (
        0.38 * np.abs(out["0"] - base_wp) +
        0.39 * np.abs(out["1"] - base_wp) +
        0.12 * np.abs(out["4"] - base_wp) +
        0.055 * np.abs(out["6"] - base_wp) +
        0.055 * np.abs(out["W"] - base_wp)
    )
    swing_range = np.maximum(out["6"] - out["W"], 0.0)
    downside_wkt = np.maximum(base_wp - out["W"], 0.0)
    upside_six = np.maximum(out["6"] - base_wp, 0.0)
    downside_four_bowl = np.maximum(out["4"] - base_wp, 0.0)
    return base_wp, exp_abs_swing, swing_range, downside_wkt, upside_six, downside_four_bowl

print("\nComputing Counterfactual Ball-Level Leverage (LI)...")
b1, sw1, rng1, dw1, u6_1, d4_1 = compute_counterfactuals(df[inn1_mask], is_inn2=False)
b2, sw2, rng2, dw2, u6_2, d4_2 = compute_counterfactuals(df[inn2_mask], is_inn2=True)

df["exp_wp_swing"] = np.nan
df["swing_range_6_W"] = np.nan
df["wkt_wp_drop"] = np.nan
df["six_wp_gain"] = np.nan
df["four_wp_cost"] = np.nan

df.loc[inn1_mask, "exp_wp_swing"] = sw1
df.loc[inn1_mask, "swing_range_6_W"] = rng1
df.loc[inn1_mask, "wkt_wp_drop"] = dw1
df.loc[inn1_mask, "six_wp_gain"] = u6_1
df.loc[inn1_mask, "four_wp_cost"] = d4_1

df.loc[inn2_mask, "exp_wp_swing"] = sw2
df.loc[inn2_mask, "swing_range_6_W"] = rng2
df.loc[inn2_mask, "wkt_wp_drop"] = dw2
df.loc[inn2_mask, "six_wp_gain"] = u6_2
df.loc[inn2_mask, "four_wp_cost"] = d4_2

# Super over leverage
df.loc[df["innings"] >= 3, "exp_wp_swing"] = 0.12
df.loc[df["innings"] >= 3, "swing_range_6_W"] = 0.40
df.loc[df["innings"] >= 3, "wkt_wp_drop"] = 0.25
df.loc[df["innings"] >= 3, "six_wp_gain"] = 0.20
df.loc[df["innings"] >= 3, "four_wp_cost"] = 0.15

med_swing = df.loc[reg_mask, "exp_wp_swing"].median()
df["leverage_index"] = df["exp_wp_swing"] / med_swing

# Compute Actual Post-Delivery WP & Win Probability Added (WPA)
# Next row within same match & innings gives post-delivery WP; at terminal ball of Inn 2 use actual match result,
# at terminal ball of Inn 1 evaluate Inn 1 WP model at post-delivery state!
print("Computing actual delivery-level Win Probability Added (WPA)...")
df["wp_after"] = df.groupby(["match_id", "innings"])["wp_before"].shift(-1)

# Handle terminal ball of each innings
is_last_in_inn = df["wp_after"].isna()
# For Inn 2 last ball in decisive match -> exact outcome (1.0 if won else 0.0)
last_inn2_dec = is_last_in_inn & (df["innings"] == 2) & df["bat_team_won"].notna()
df.loc[last_inn2_dec, "wp_after"] = df.loc[last_inn2_dec, "bat_team_won"]

# For Inn 1 last ball (or non-decisive Inn 2 last ball) -> evaluate post-state WP
last_inn1 = is_last_in_inn & (df["innings"] == 1)
if last_inn1.sum() > 0:
    post1 = df.loc[last_inn1].copy()
    post1["runs_before"] = post1["runs_before"] + post1["total_runs"]
    post1["wickets_before"] = np.minimum(post1["wickets_before"] + post1["is_wicket"].astype(int), 10)
    post1["legal_balls_before"] = post1["legal_balls_before"] + post1["is_legal"].astype(int)
    post1["balls_remaining"] = np.maximum(post1["balls_remaining"] - post1["is_legal"].astype(int), 0.0)
    post1["crr_before"] = post1["runs_before"] / np.maximum(post1["legal_balls_before"] / 6.0, 1/6)
    df.loc[last_inn1, "wp_after"] = predict_wp_batch(post1, inn1_features, models1, iso1)

# Fallback for non-decisive last ball in Inn 2
last_inn2_nd = is_last_in_inn & (df["innings"] == 2) & df["bat_team_won"].isna()
if last_inn2_nd.sum() > 0:
    post2 = df.loc[last_inn2_nd].copy()
    post2["runs_before"] = post2["runs_before"] + post2["total_runs"]
    post2["runs_needed"] = np.maximum(post2["runs_needed"] - post2["total_runs"], 0.0)
    post2["wickets_before"] = np.minimum(post2["wickets_before"] + post2["is_wicket"].astype(int), 10)
    post2["legal_balls_before"] = post2["legal_balls_before"] + post2["is_legal"].astype(int)
    post2["balls_remaining"] = np.maximum(post2["balls_remaining"] - post2["is_legal"].astype(int), 0.0)
    post2["crr_before"] = post2["runs_before"] / np.maximum(post2["legal_balls_before"] / 6.0, 1/6)
    post2["rrr_before"] = np.where(post2["balls_remaining"] > 0, np.clip(post2["runs_needed"] / (post2["balls_remaining"] / 6.0), 0.0, 36.0), 36.0)
    post2["rrr_minus_crr"] = post2["rrr_before"] - post2["crr_before"]
    post2["rrr_ratio_par"] = post2["rrr_before"] / np.maximum(post2["prior_venue_rpo"], 5.0)
    df.loc[last_inn2_nd, "wp_after"] = predict_wp_batch(post2, inn2_features, models2, iso2)

# Super overs WPA fallback
df.loc[df["innings"] >= 3, "wp_after"] = df.loc[df["innings"] >= 3, "wp_after"].fillna(df.loc[df["innings"] >= 3, "bat_team_won"].fillna(0.5))

df["wpa_bat"] = df["wp_after"] - df["wp_before"]
df["wpa_bowl"] = -df["wpa_bat"]
df["abs_wpa"] = np.abs(df["wpa_bat"])

# ==============================================================================
# CONSTRUCT INTERPRETABLE MULTI-DIMENSIONAL PRESSURE INDICES (P_bat, P_bowl)
# ==============================================================================
# 1. Match Closeness / Uncertainty: U = 4 * WP * (1 - WP) in [0, 1]
wp = df["wp_before"].values
df["match_closeness"] = 4.0 * wp * (1.0 - wp)

# Non-garbage-time live contest gate: smoothly decays when WP < 0.04 or WP > 0.96
# (especially near the end of an already-decided chase/defense!)
min_wp_side = np.minimum(wp, 1.0 - wp)
df["live_gate"] = np.clip(min_wp_side / 0.06, 0.0, 1.0) ** 0.6

# 2. Normalized Leverage Score (log-compressed percentile in [0, 1])
li_log = np.log1p(df["leverage_index"].clip(0, 12.0))
df["comp_leverage"] = np.clip(li_log / np.quantile(li_log[reg_mask], 0.98), 0.0, 1.0) * df["live_gate"]

# 3. Chase / Target Acceleration & Par Deficit Stress
# Inn 2: RRR relative to par, amplified by wicket scarcity and endgame proximity
rrr_excess_2 = np.clip((df["rrr_ratio_par"].fillna(1.0) - 0.85) / 1.15, 0.0, 1.25)
wkt_scarcity = np.clip(np.sqrt(5.0 / np.maximum(df["wickets_in_hand"], 1.0)), 0.7, 2.0) / 1.5
prog = np.clip(1.0 - df["balls_remaining"] / np.maximum(df["max_balls_inn"], 6.0), 0.0, 1.0)
chase_stress_2 = rrr_excess_2 * (0.55 + 0.45 * prog) * wkt_scarcity * df["live_gate"]

# Inn 1: Par deficit / setting-target acceleration in live contest
expected_runs_now = df["prior_venue_rpo"] * (df["legal_balls_before"] / 6.0)
par_deficit_1 = np.clip((expected_runs_now - df["runs_before"]) / 35.0, 0.0, 1.0)
death_ramp_1 = np.where(df["overid"] >= 15, np.clip((df["overid"] - 14) / 6.0, 0.0, 1.0), 0.0)
inn1_bat_stress = (0.55 * par_deficit_1 + 0.45 * death_ramp_1 * (0.5 + 0.5 * df["match_closeness"])) * df["live_gate"]

df["comp_chase_target"] = np.where(df["innings"] == 2, np.clip(chase_stress_2, 0.0, 1.0), np.clip(inn1_bat_stress, 0.0, 1.0))

# 4. Collapse / Wicket-Fragility Stress (critical for Batters!)
# Excess wickets lost relative to innings progress (~1 wicket per 19 legal balls), plus recent 12-ball cluster
exp_wkts_at_ball = df["legal_balls_before"] / 19.0
excess_wkts = np.clip(df["wickets_before"] - exp_wkts_at_ball, 0.0, 5.0)
# Early top-order collapse (e.g., >=2 down in PP or >=3 down in first 10 overs)
early_collapse = np.where((df["overid"] <= 6) & (df["wickets_before"] >= 2), 0.35 * (df["wickets_before"] - 1), 0.0)
recent_shock = np.clip(df["wkts_last_12"] / 2.0, 0.0, 1.0) * 0.35
raw_collapse = np.clip(excess_wkts / 3.0 + early_collapse + recent_shock, 0.0, 1.2)
df["comp_collapse"] = np.clip(raw_collapse, 0.0, 1.0) * np.clip(min_wp_side / 0.03, 0.0, 1.0)

# 5. Bowling Defensive Containment & Execution Stress
# In Inn 2: High when chase is alive & batting team has enough wickets/close RRR (bowler cannot leak runs!)
# Especially captured by four_wp_cost / swing_range and endgame closeness!
inn2_bowl_ease_threat = np.clip(1.25 - np.abs(df["rrr_ratio_par"].fillna(1.0) - 1.05), 0.0, 1.0)
inn2_bowl_stress = (
    0.50 * df["match_closeness"] * (0.45 + 0.55 * prog) +
    0.50 * inn2_bowl_ease_threat * np.clip(df["wickets_in_hand"] / 6.0, 0.3, 1.0) * prog
) * df["live_gate"]
# In Inn 1: Death containment (16-20) + Powerplay new-ball pressure when batters are accelerating above par
par_surplus_1 = np.clip((df["runs_before"] - expected_runs_now) / 30.0, 0.0, 1.0)
pp_newball_threat = np.where(df["overid"] <= 6, 0.35 + 0.35 * par_surplus_1, 0.0)
death_def_1 = np.where(df["overid"] >= 16, np.clip((df["overid"] - 15) / 5.0, 0.2, 1.0) * (0.5 + 0.5 * df["match_closeness"]), 0.0)
inn1_bowl_stress = np.clip(np.maximum(pp_newball_threat, death_def_1), 0.0, 1.0) * df["live_gate"]

df["comp_bowl_execution"] = np.where(df["innings"] == 2, np.clip(inn2_bowl_stress, 0.0, 1.0), np.clip(inn1_bowl_stress, 0.0, 1.0))

# 6. Tournament / Knockout Stage Stakes Boost
stage_boost = np.where(df["is_high_stakes_stage"], 0.10, 0.0)

# Composite Raw Pressure Scores
raw_p_bat = (
    0.38 * df["comp_leverage"] +
    0.22 * df["match_closeness"] * df["live_gate"] +
    0.24 * df["comp_chase_target"] +
    0.16 * df["comp_collapse"] +
    stage_boost
)
raw_p_bowl = (
    0.40 * df["comp_leverage"] +
    0.24 * df["match_closeness"] * df["live_gate"] +
    0.36 * df["comp_bowl_execution"] +
    stage_boost
)

# Super over override -> maximum pressure (0.99)
raw_p_bat = np.where(df["innings"] >= 3, 0.99, raw_p_bat)
raw_p_bowl = np.where(df["innings"] >= 3, 0.99, raw_p_bowl)

# Percentile-rank normalize onto [0, 1] across regular deliveries so thresholds have exact sample meaning
# (e.g., P >= 0.70 corresponds cleanly to the top ~30% highest-pressure deliveries!)
def empirical_cdf_scale(arr, ref_mask):
    ref_sorted = np.sort(arr[ref_mask])
    ranks = np.searchsorted(ref_sorted, arr, side="right") / len(ref_sorted)
    return np.clip(ranks, 0.0, 1.0)

df["pressure_bat_raw"] = raw_p_bat
df["pressure_bowl_raw"] = raw_p_bowl
df["pressure_bat"] = empirical_cdf_scale(raw_p_bat, reg_mask.values)
df["pressure_bowl"] = empirical_cdf_scale(raw_p_bowl, reg_mask.values)

# ==============================================================================
# EMPIRICAL VALIDATION: DOES PRESSURE DISTINGUISH PRESSURE FROM ORDINARY PLAY?
# ==============================================================================
reg_df = df[reg_mask].copy()
reg_df["p_bat_decile"] = pd.qcut(reg_df["pressure_bat"], 10, labels=[f"D{i}" for i in range(1, 11)])
reg_df["p_bowl_decile"] = pd.qcut(reg_df["pressure_bowl"], 10, labels=[f"D{i}" for i in range(1, 11)])
reg_df["is_extra_err"] = (reg_df["is_wide"] | reg_df["is_noball"]).astype(int)
reg_df["is_dot_bat"] = (reg_df["runs"] == 0).astype(int)

print("\n=== EMPIRICAL PRESSURE VALIDATION (BATTING PRESSURE DECILES) ===")
bat_val = reg_df.groupby("p_bat_decile", observed=False).agg(
    balls=("runs", "count"),
    mean_li=("leverage_index", "mean"),
    mean_closeness=("match_closeness", "mean"),
    actual_abs_wpa=("abs_wpa", "mean"),
    rpo=("total_runs", lambda s: s.sum() / (reg_df.loc[s.index, "is_legal"].sum() / 6.0)),
    wkt_pct=("is_wicket", lambda s: s.mean() * 100),
    boundary_pct=("is_boundary", lambda s: s.mean() * 100),
    runs_std=("runs", "std"),
).reset_index()
print(bat_val.to_string(index=False))

print("\n=== EMPIRICAL PRESSURE VALIDATION (BOWLING PRESSURE DECILES) ===")
bowl_val = reg_df.groupby("p_bowl_decile", observed=False).agg(
    balls=("bowler_runs", "count"),
    mean_li=("leverage_index", "mean"),
    mean_closeness=("match_closeness", "mean"),
    actual_abs_wpa=("abs_wpa", "mean"),
    rpo_conceded=("bowler_runs", lambda s: s.sum() / (reg_df.loc[s.index, "is_legal"].sum() / 6.0)),
    bowler_wkt_pct=("is_bowler_wicket", lambda s: s.mean() * 100),
    extra_err_pct=("is_extra_err", lambda s: s.mean() * 100),
    boundary_pct=("is_boundary", lambda s: s.mean() * 100),
).reset_index()
print(bowl_val.to_string(index=False))

print("\n=== PRESSURE BY STAGE (LEAGUE VS HIGH-STAKES KNOCKOUTS) ===")
print(reg_df.groupby("is_high_stakes_stage")[["pressure_bat", "pressure_bowl", "abs_wpa"]].mean().to_string())

# Save validation tables & full dataset
bat_val.to_csv(os.path.join(OUT_DIR, "validation_bat_pressure_deciles.csv"), index=False)
bowl_val.to_csv(os.path.join(OUT_DIR, "validation_bowl_pressure_deciles.csv"), index=False)
df.to_pickle(r"d:\Rajasthan Royals\outputs\processed\deliveries_with_pressure.pkl")
print("\nSaved deliveries_with_pressure.pkl successfully!")

