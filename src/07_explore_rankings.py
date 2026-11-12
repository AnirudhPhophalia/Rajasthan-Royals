import os
import numpy as np
import pandas as pd

IN_PATH = r"d:\Rajasthan Royals\outputs\processed\deliveries_modeled.pkl"
OUT_DIR = r"d:\Rajasthan Royals\outputs\rankings"
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading deliveries_modeled.pkl...")
df = pd.read_pickle(IN_PATH)
id_map = pd.read_csv(r"d:\Rajasthan Royals\outputs\processed\player_id_map.csv").set_index("player_id")["player_name"].to_dict()

# Helper: compute cluster-robust player statistics under High Pressure vs Ordinary Baseline
def compute_player_pressure_stats(
    sub_df, id_col, ball_flag_col, p_col, rva_col, raa_col, waa_col, wpa_col,
    tau_high=0.70, tau_low=0.55, top_tier_only=False
):
    d = sub_df.copy()
    if top_tier_only:
        d = d[d["is_top_tier_match"]].copy()

    # Keep only valid deliveries (e.g. balls faced for batting, all deliveries for bowling)
    d = d[d[ball_flag_col] == True].copy() if ball_flag_col else d.copy()
    d["p_val"] = d[p_col]
    d["is_high"] = d["p_val"] >= tau_high
    d["is_low"] = d["p_val"] < tau_low

    # Overall summary per player
    overall = d.groupby(id_col).agg(
        total_matches=("match_id", "nunique"),
        top_tier_matches=("match_id", lambda s: s[d.loc[s.index, "is_top_tier_match"]].nunique()),
        ipl_matches=("match_id", lambda s: s[d.loc[s.index, "comp_group"] == "IPL"].nunique()),
        t20i_matches=("match_id", lambda s: s[d.loc[s.index, "comp_group"] == "T20I"].nunique()),
        total_balls=(rva_col, "count"),
        top_tier_balls=(rva_col, lambda s: d.loc[s.index, "is_top_tier_match"].sum()),
        mean_rva_all=(rva_col, "mean"),
        mean_wpa_all=(wpa_col, "mean"),
    ).reset_index().rename(columns={id_col: "player_id"})

    def summarize_regime(reg_df, prefix):
        # Aggregate first to match level (cluster by match_id!)
        m_agg = reg_df.groupby([id_col, "match_id"]).agg(
            n_balls=(rva_col, "count"),
            rva_sum=(rva_col, "sum"),
            raa_sum=(raa_col, "sum"),
            waa_sum=(waa_col, "sum"),
            wpa_sum=(wpa_col, "sum"),
        ).reset_index()
        m_agg["rva_mean"] = m_agg["rva_sum"] / m_agg["n_balls"]
        m_agg["wpa_mean"] = m_agg["wpa_sum"] / m_agg["n_balls"]

        # Player-level totals & cluster-robust SE (Liang-Zeger sandwich for weighted mean)
        rows = []
        for pid, grp in m_agg.groupby(id_col):
            M = len(grp)
            N = grp["n_balls"].sum()
            mu_rva = grp["rva_sum"].sum() / N
            mu_raa = grp["raa_sum"].sum() / N
            mu_waa = grp["waa_sum"].sum() / N
            mu_wpa = grp["wpa_sum"].sum() / N

            w = grp["n_balls"].values
            if M >= 2:
                diff_rva = grp["rva_mean"].values - mu_rva
                var_rva = (M / (M - 1)) * np.sum((w ** 2) * (diff_rva ** 2)) / (N ** 2)
                se_rva = np.sqrt(max(var_rva, 1e-8))

                diff_wpa = grp["wpa_mean"].values - mu_wpa
                var_wpa = (M / (M - 1)) * np.sum((w ** 2) * (diff_wpa ** 2)) / (N ** 2)
                se_wpa = np.sqrt(max(var_wpa, 1e-10))
            else:
                se_rva = 0.50
                se_wpa = 0.02

            # Match-level positive-impact rate (matches with >=3 balls in regime where RVA > 0)
            sig_m = grp[grp["n_balls"] >= 3]
            n_sig_m = len(sig_m)
            pos_m = (sig_m["rva_sum"] > 0).sum() if n_sig_m > 0 else 0

            rows.append({
                "player_id": pid,
                f"{prefix}_matches": M,
                f"{prefix}_balls": int(N),
                f"{prefix}_rva100": mu_rva * 100.0,
                f"{prefix}_raa100": mu_raa * 100.0,
                f"{prefix}_waa100": mu_waa * 100.0,
                f"{prefix}_wpa100": mu_wpa * 100.0,
                f"{prefix}_se_rva100": se_rva * 100.0,
                f"{prefix}_se_wpa100": se_wpa * 100.0,
                f"{prefix}_sig_matches": n_sig_m,
                f"{prefix}_pos_matches": pos_m,
            })
        return pd.DataFrame(rows)

    hi_df = summarize_regime(d[d["is_high"]], "hi")
    lo_df = summarize_regime(d[d["is_low"]], "lo")
    out = overall.merge(hi_df, on="player_id", how="left").merge(lo_df, on="player_id", how="left")
    out["player_name"] = out["player_id"].map(id_map)
    return out

print("Computing Batting & Bowling stats (All matches with Elo adjustment)...")
bat_all = compute_player_pressure_stats(
    df, "striker_id", "is_ball_faced", "pressure_bat",
    "rva_bat", "raa_bat", "waa_bat", "wpa_bat", tau_high=0.70, tau_low=0.55
)
bowl_all = compute_player_pressure_stats(
    df, "bowler_id", None, "pressure_bowl",
    "rva_bowl", "raa_bowl", "waa_bowl", "wpa_bowl", tau_high=0.70, tau_low=0.55
)

# Inspect eligibility threshold options
print("\n=== BATTER ELIGIBILITY POOL SIZES (Top-Tier Balls & High-Pressure Balls) ===")
for tb in [300, 400, 500]:
    for hb in [80, 100, 120, 150]:
        c = ((bat_all["top_tier_balls"] >= tb) & (bat_all["hi_balls"] >= hb) & (bat_all["hi_matches"] >= 12) & (bat_all["lo_balls"] >= 100)).sum()
        print(f"  top_tier_balls>={tb}, hi_balls>={hb}, hi_matches>=12 -> {c} batters")

print("\n=== BOWLER ELIGIBILITY POOL SIZES (Top-Tier Balls & High-Pressure Balls) ===")
for tb in [300, 400, 500]:
    for hb in [80, 100, 120, 150]:
        c = ((bowl_all["top_tier_balls"] >= tb) & (bowl_all["hi_balls"] >= hb) & (bowl_all["hi_matches"] >= 12) & (bowl_all["lo_balls"] >= 100)).sum()
        print(f"  top_tier_balls>={tb}, hi_balls>={hb}, hi_matches>=12 -> {c} bowlers")

# Quick preview with clean threshold: top_tier_balls >= 400, hi_balls >= 120, hi_matches >= 15, lo_balls >= 120
def preview_eb_rank(pool, label):
    p = pool.copy()
    # Empirical Bayes shrinkage on hi_rva100, lo_rva100, hi_wpa100, lo_wpa100, uplift
    for col, se_col in [("hi_rva100", "hi_se_rva100"), ("lo_rva100", "lo_se_rva100"),
                        ("hi_wpa100", "hi_se_wpa100"), ("lo_wpa100", "lo_se_wpa100")]:
        mu0 = np.average(p[col], weights=1.0 / (p[se_col] ** 2))
        tau2 = max(np.var(p[col]) - np.mean(p[se_col] ** 2), 1.0 if "rva" in col else 0.05)
        w_data = 1.0 / (p[se_col] ** 2)
        w_prior = 1.0 / tau2
        p[col + "_eb"] = (w_data * p[col] + w_prior * mu0) / (w_data + w_prior)
        p[col + "_post_se"] = np.sqrt(1.0 / (w_data + w_prior))

    # Uplift (High - Low) with joint EB shrinkage
    p["uplift_rva100"] = p["hi_rva100"] - p["lo_rva100"]
    p["uplift_se_rva100"] = np.sqrt(p["hi_se_rva100"] ** 2 + p["lo_se_rva100"] ** 2)
    mu0_u = np.average(p["uplift_rva100"], weights=1.0 / (p["uplift_se_rva100"] ** 2))
    tau2_u = max(np.var(p["uplift_rva100"]) - np.mean(p["uplift_se_rva100"] ** 2), 1.0)
    wu = 1.0 / (p["uplift_se_rva100"] ** 2)
    p["uplift_rva100_eb"] = (wu * p["uplift_rva100"] + (1.0 / tau2_u) * mu0_u) / (wu + 1.0 / tau2_u)

    p["uplift_wpa100"] = p["hi_wpa100"] - p["lo_wpa100"]
    p["uplift_se_wpa100"] = np.sqrt(p["hi_se_wpa100"] ** 2 + p["lo_se_wpa100"] ** 2)
    mu0_uw = np.average(p["uplift_wpa100"], weights=1.0 / (p["uplift_se_wpa100"] ** 2))
    tau2_uw = max(np.var(p["uplift_wpa100"]) - np.mean(p["uplift_se_wpa100"] ** 2), 0.05)
    wuw = 1.0 / (p["uplift_se_wpa100"] ** 2)
    p["uplift_wpa100_eb"] = (wuw * p["uplift_wpa100"] + (1.0 / tau2_uw) * mu0_uw) / (wuw + 1.0 / tau2_uw)

    # Positive match rate under pressure (Beta-Binomial EB with prior Beta(4, 4))
    p["pmr_eb"] = (p["hi_pos_matches"] + 4.0) / (p["hi_sig_matches"] + 8.0)
    # Lower Confidence Bound on hi_rva100_eb
    p["lcb_rva100"] = p["hi_rva100_eb"] - 0.5 * p["hi_rva100_post_se"]

    def z(s):
        return (s - s.mean()) / (s.std() + 1e-9)

    p["z_level"] = 0.65 * z(p["hi_rva100_eb"]) + 0.35 * z(p["hi_wpa100_eb"])
    p["z_uplift"] = 0.65 * z(p["uplift_rva100_eb"]) + 0.35 * z(p["uplift_wpa100_eb"])
    p["z_consist"] = 0.50 * z(p["pmr_eb"]) + 0.50 * z(p["lcb_rva100"])

    p["SPRI"] = 0.45 * p["z_level"] + 0.30 * p["z_uplift"] + 0.25 * p["z_consist"]
    p = p.sort_values("SPRI", ascending=False).reset_index(drop=True)
    p["rank"] = p.index + 1

    cols = [
        "rank", "player_name", "player_id", "ipl_matches", "t20i_matches",
        "hi_balls", "hi_matches", "lo_rva100_eb", "hi_rva100_eb", "uplift_rva100_eb",
        "hi_wpa100_eb", "pmr_eb", "SPRI"
    ]
    print(f"\n=== TOP 20 {label} (Pool N={len(p)}) ===")
    print(p[cols].head(20).round(3).to_string(index=False))
    return p

bat_pool = bat_all[(bat_all["top_tier_balls"] >= 400) & (bat_all["hi_balls"] >= 120) & (bat_all["hi_matches"] >= 15) & (bat_all["lo_balls"] >= 120)].copy()
bowl_pool = bowl_all[(bowl_all["top_tier_balls"] >= 400) & (bowl_all["hi_balls"] >= 120) & (bowl_all["hi_matches"] >= 15) & (bowl_all["lo_balls"] >= 120)].copy()

preview_eb_rank(bat_pool, "BATTERS")
preview_eb_rank(bowl_pool, "BOWLERS")

bat_all.to_csv(os.path.join(OUT_DIR, "bat_all_raw.csv"), index=False)
bowl_all.to_csv(os.path.join(OUT_DIR, "bowl_all_raw.csv"), index=False)

