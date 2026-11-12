import os
import numpy as np
import pandas as pd

IN_PATH = r"d:\Rajasthan Royals\outputs\processed\deliveries_modeled.pkl"
df = pd.read_pickle(IN_PATH)
id_map = pd.read_csv(r"d:\Rajasthan Royals\outputs\processed\player_id_map.csv").set_index("player_id")["player_name"].to_dict()

# Center WPA within pressure deciles so WPA_above_exp has zero mean at every pressure level
for p_col, wpa_col, out_col in [
    ("pressure_bat", "wpa_bat", "wpa_bat_adj"),
    ("pressure_bowl", "wpa_bowl", "wpa_bowl_adj"),
]:
    dec = pd.qcut(df[p_col], 20, labels=False)
    df[out_col] = df[wpa_col] - df.groupby(dec)[wpa_col].transform("mean")

def build_table(sub_df, id_col, ball_flag_col, p_col, rva_col, raa_col, waa_col, wpa_col,
                tau_high=0.70, tau_low=0.55):
    d = sub_df.copy()
    if ball_flag_col:
        d = d[d[ball_flag_col] == True].copy()
    d["p_val"] = d[p_col]
    d["is_high"] = d["p_val"] >= tau_high
    d["is_low"] = d["p_val"] < tau_low

    # Competition / tier weight? Check top-tier vs all
    overall = d.groupby(id_col).agg(
        total_matches=("match_id", "nunique"),
        top_tier_matches=("match_id", lambda s: s[d.loc[s.index, "is_top_tier_match"]].nunique()),
        ipl_matches=("match_id", lambda s: s[d.loc[s.index, "comp_group"] == "IPL"].nunique()),
        t20i_matches=("match_id", lambda s: s[d.loc[s.index, "comp_group"] == "T20I"].nunique()),
        total_balls=(rva_col, "count"),
        top_tier_balls=(rva_col, lambda s: d.loc[s.index, "is_top_tier_match"].sum()),
    ).reset_index().rename(columns={id_col: "player_id"})

    def summarize_regime(reg_df, prefix, weight_by_p=False):
        r = reg_df.copy()
        r["w"] = (r["p_val"] ** 2) if weight_by_p else 1.0
        r["wrva"] = r["w"] * r[rva_col]
        r["wraa"] = r["w"] * r[raa_col]
        r["wwaa"] = r["w"] * r[waa_col]
        r["wwpa"] = r["w"] * r[wpa_col]
        m_agg = r.groupby([id_col, "match_id"]).agg(
            n_balls=(rva_col, "count"),
            w_sum=("w", "sum"),
            wrva_sum=("wrva", "sum"),
            wraa_sum=("wraa", "sum"),
            wwaa_sum=("wwaa", "sum"),
            wwpa_sum=("wwpa", "sum"),
        ).reset_index()
        m_agg["rva_mean"] = m_agg["wrva_sum"] / m_agg["w_sum"]
        m_agg["wpa_mean"] = m_agg["wwpa_sum"] / m_agg["w_sum"]

        rows = []
        for pid, grp in m_agg.groupby(id_col):
            M = len(grp)
            N = grp["n_balls"].sum()
            W = grp["w_sum"].sum()
            mu_rva = grp["wrva_sum"].sum() / W
            mu_raa = grp["wraa_sum"].sum() / W
            mu_waa = grp["wwaa_sum"].sum() / W
            mu_wpa = grp["wwpa_sum"].sum() / W

            wm = grp["w_sum"].values
            if M >= 2:
                d_rva = grp["rva_mean"].values - mu_rva
                var_rva = (M / (M - 1)) * np.sum((wm ** 2) * (d_rva ** 2)) / (W ** 2)
                se_rva = np.sqrt(max(var_rva, 1e-8))

                d_wpa = grp["wpa_mean"].values - mu_wpa
                var_wpa = (M / (M - 1)) * np.sum((wm ** 2) * (d_wpa ** 2)) / (W ** 2)
                se_wpa = np.sqrt(max(var_wpa, 1e-10))
            else:
                se_rva = 0.50
                se_wpa = 0.02

            sig_m = grp[grp["n_balls"] >= 3]
            n_sig_m = len(sig_m)
            pos_m = (sig_m["wrva_sum"] > 0).sum() if n_sig_m > 0 else 0
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

    hi_df = summarize_regime(d[d["is_high"]], "hi", weight_by_p=True)
    lo_df = summarize_regime(d[d["is_low"]], "lo", weight_by_p=False)
    out = overall.merge(hi_df, on="player_id", how="left").merge(lo_df, on="player_id", how="left")
    out["player_name"] = out["player_id"].map(id_map)
    return out

def score_pool(pool, label):
    p = pool.copy()
    # Empirical Bayes shrinkage on hi_rva100, lo_rva100, hi_wpa100, lo_wpa100
    for col, se_col in [("hi_rva100", "hi_se_rva100"), ("lo_rva100", "lo_se_rva100"),
                        ("hi_wpa100", "hi_se_wpa100"), ("lo_wpa100", "lo_se_wpa100")]:
        iv = 1.0 / (p[se_col] ** 2)
        mu0 = np.average(p[col], weights=iv)
        # DerSimonian-Laird / EB between-player variance
        tau2 = max(np.var(p[col]) - np.mean(p[se_col] ** 2), 25.0 if "rva" in col else 0.03)
        w_prior = 1.0 / tau2
        p[col + "_eb"] = (iv * p[col] + w_prior * mu0) / (iv + w_prior)
        p[col + "_post_se"] = np.sqrt(1.0 / (iv + w_prior))

    # Weighted linear regression: E[hi_eb | lo_eb] to compute Regression-Adjusted Pressure Resilience
    w_fit = 1.0 / (p["hi_rva100_post_se"] ** 2)
    b1, b0 = np.polyfit(p["lo_rva100_eb"], p["hi_rva100_eb"], 1, w=w_fit)
    p["exp_hi_rva100"] = b0 + b1 * p["lo_rva100_eb"]
    p["res_rva100"] = p["hi_rva100_eb"] - p["exp_hi_rva100"]
    p["raw_uplift_eb"] = p["hi_rva100_eb"] - p["lo_rva100_eb"]

    w_fit_w = 1.0 / (p["hi_wpa100_post_se"] ** 2)
    bw1, bw0 = np.polyfit(p["lo_wpa100_eb"], p["hi_wpa100_eb"], 1, w=w_fit_w)
    p["res_wpa100"] = p["hi_wpa100_eb"] - (bw0 + bw1 * p["lo_wpa100_eb"])

    print(f"[{label}] N={len(p)} | RVA slope(hi|lo)={b1:.3f}, intercept={b0:.3f} | WPA slope={bw1:.3f}")

    # Consistency metrics:
    # 1. Beta-Binomial Positive-Impact Match Rate under pressure (prior Beta(6, 6))
    p["pmr_eb"] = (p["hi_pos_matches"] + 6.0) / (p["hi_sig_matches"] + 12.0)
    # 2. Lower Confidence Bound (84% LCB = mu_eb - 1.0 * post_se) on High-Pressure RVA
    p["lcb_rva100"] = p["hi_rva100_eb"] - 1.0 * p["hi_rva100_post_se"]

    def z(s):
        return (s - s.mean()) / (s.std() + 1e-9)

    p["z_level"] = 0.65 * z(p["hi_rva100_eb"]) + 0.35 * z(p["hi_wpa100_eb"])
    p["z_resilience"] = 0.65 * z(p["res_rva100"]) + 0.35 * z(p["res_wpa100"])
    p["z_consist"] = 0.50 * z(p["pmr_eb"]) + 0.50 * z(p["lcb_rva100"])

    # Primary SPRI: 45% High-Pressure Output, 30% Baseline-Relative Resilience, 25% Match Consistency
    p["SPRI"] = 0.45 * p["z_level"] + 0.30 * p["z_resilience"] + 0.25 * p["z_consist"]
    p = p.sort_values("SPRI", ascending=False).reset_index(drop=True)
    p["rank"] = p.index + 1

    cols = [
        "rank", "player_name", "ipl_matches", "t20i_matches", "hi_balls", "hi_matches",
        "lo_rva100_eb", "hi_rva100_eb", "res_rva100", "raw_uplift_eb", "hi_wpa100_eb", "pmr_eb", "SPRI"
    ]
    print(p[cols].head(15).round(3).to_string(index=False))
    return p

bat = build_table(df, "striker_id", "is_ball_faced", "pressure_bat", "rva_bat", "raa_bat", "waa_bat", "wpa_bat_adj")
bowl = build_table(df, "bowler_id", None, "pressure_bowl", "rva_bowl", "raa_bowl", "waa_bowl", "wpa_bowl_adj")

bat_pool = bat[(bat["top_tier_balls"] >= 500) & (bat["hi_balls"] >= 180) & (bat["hi_matches"] >= 20) & (bat["lo_balls"] >= 150)].copy()
bowl_pool = bowl[(bowl["top_tier_balls"] >= 500) & (bowl["hi_balls"] >= 180) & (bowl["hi_matches"] >= 20) & (bowl["lo_balls"] >= 150)].copy()

score_pool(bat_pool, "BATTERS")
print()
score_pool(bowl_pool, "BOWLERS")

