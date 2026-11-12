import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

IN_PATH = r"d:\Rajasthan Royals\outputs\processed\deliveries_modeled.pkl"
OUT_DIR = r"d:\Rajasthan Royals\outputs\validation"
RANK_DIR = r"d:\Rajasthan Royals\outputs\rankings"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(RANK_DIR, exist_ok=True)

print("Loading deliveries_modeled.pkl...")
df = pd.read_pickle(IN_PATH)
id_map = pd.read_csv(r"d:\Rajasthan Royals\outputs\processed\player_id_map.csv").set_index("player_id")["player_name"].to_dict()

# Center WPA within 20 pressure quantiles so WPA_above_exp has zero expectation at every pressure tier
for p_col, wpa_col, out_col in [
    ("pressure_bat", "wpa_bat", "wpa_bat_adj"),
    ("pressure_bowl", "wpa_bowl", "wpa_bowl_adj"),
]:
    dec = pd.qcut(df[p_col], 20, labels=False)
    df[out_col] = df[wpa_col] - df.groupby(dec)[wpa_col].transform("mean")

df["is_extra_err"] = (df["is_wide"] | df["is_noball"]).astype(int)
df["is_dot_bat"] = (df["runs"] == 0).astype(int)
df["is_dot_bowl"] = (df["bowler_runs"] == 0).astype(int)

def compute_full_player_metrics(
    sub_df, id_col, ball_flag_col, p_col, rva_col, raa_col, waa_col, wpa_col,
    tau_high=0.70, tau_low=0.55, min_top_tier_balls=500, min_hi_balls=180, min_hi_matches=20, min_lo_balls=150
):
    d = sub_df.copy()
    if ball_flag_col:
        d = d[d[ball_flag_col] == True].copy()
    d["p_val"] = d[p_col]
    d["is_high"] = d["p_val"] >= tau_high
    d["is_low"] = d["p_val"] < tau_low

    is_bat = (id_col == "striker_id")

    overall = d.groupby(id_col).agg(
        total_matches=("match_id", "nunique"),
        top_tier_matches=("match_id", lambda s: s[d.loc[s.index, "is_top_tier_match"]].nunique()),
        ipl_matches=("match_id", lambda s: s[d.loc[s.index, "comp_group"] == "IPL"].nunique()),
        t20i_matches=("match_id", lambda s: s[d.loc[s.index, "comp_group"] == "T20I"].nunique()),
        ko_hi_balls=(rva_col, lambda s: ((d.loc[s.index, "is_high"]) & (d.loc[s.index, "is_high_stakes_stage"])).sum()),
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

        # Also raw cricket metrics for interpretability
        raw_runs_col = "runs" if is_bat else "bowler_runs"
        raw_wkt_col = "striker_is_dismissed" if is_bat else "is_bowler_wicket"
        raw_dot_col = "is_dot_bat" if is_bat else "is_dot_bowl"

        m_agg = r.groupby([id_col, "match_id"]).agg(
            n_balls=(rva_col, "count"),
            n_legal=("is_legal", "sum"),
            w_sum=("w", "sum"),
            wrva_sum=("wrva", "sum"),
            wraa_sum=("wraa", "sum"),
            wwaa_sum=("wwaa", "sum"),
            wwpa_sum=("wwpa", "sum"),
            raw_runs=(raw_runs_col, "sum"),
            raw_wkts=(raw_wkt_col, "sum"),
            raw_dots=(raw_dot_col, "sum"),
            raw_bdy=("is_boundary", "sum"),
            raw_err=("is_extra_err", "sum"),
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

            n_leg = max(grp["n_legal"].sum(), 1)
            raw_runs_tot = grp["raw_runs"].sum()
            raw_wkts_tot = grp["raw_wkts"].sum()

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
                f"{prefix}_sr_or_econ": (raw_runs_tot / N * 100.0) if is_bat else (raw_runs_tot / (n_leg / 6.0)),
                f"{prefix}_wkt_rate": (raw_wkts_tot / N * 100.0),
                f"{prefix}_dot_pct": (grp["raw_dots"].sum() / N * 100.0),
                f"{prefix}_bdy_pct": (grp["raw_bdy"].sum() / N * 100.0),
                f"{prefix}_err_pct": (grp["raw_err"].sum() / N * 100.0),
            })
        return pd.DataFrame(rows)

    hi_df = summarize_regime(d[d["is_high"]], "hi", weight_by_p=True)
    lo_df = summarize_regime(d[d["is_low"]], "lo", weight_by_p=False)

    # Also format-specific high-pressure RVA (IPL vs T20I)
    hi_ipl = d[d["is_high"] & (d["comp_group"] == "IPL")].groupby(id_col).agg(
        hi_ipl_balls=(rva_col, "count"), hi_ipl_rva100=(rva_col, lambda s: s.mean() * 100.0)
    ).reset_index().rename(columns={id_col: "player_id"})
    hi_t20i = d[d["is_high"] & (d["comp_group"] == "T20I")].groupby(id_col).agg(
        hi_t20i_balls=(rva_col, "count"), hi_t20i_rva100=(rva_col, lambda s: s.mean() * 100.0)
    ).reset_index().rename(columns={id_col: "player_id"})

    out = overall.merge(hi_df, on="player_id", how="left").merge(lo_df, on="player_id", how="left")
    out = out.merge(hi_ipl, on="player_id", how="left").merge(hi_t20i, on="player_id", how="left")
    out["player_name"] = out["player_id"].map(id_map)

    pool = out[
        (out["top_tier_balls"] >= min_top_tier_balls) &
        (out["hi_balls"] >= min_hi_balls) &
        (out["hi_matches"] >= min_hi_matches) &
        (out["lo_balls"] >= min_lo_balls)
    ].copy()

    # Empirical Bayes Shrinkage
    for col, se_col in [("hi_rva100", "hi_se_rva100"), ("lo_rva100", "lo_se_rva100"),
                        ("hi_wpa100", "hi_se_wpa100"), ("lo_wpa100", "lo_se_wpa100")]:
        iv = 1.0 / (pool[se_col] ** 2)
        mu0 = np.average(pool[col], weights=iv)
        tau2 = max(np.var(pool[col]) - np.mean(pool[se_col] ** 2), 25.0 if "rva" in col else 0.03)
        w_prior = 1.0 / tau2
        pool[col + "_eb"] = (iv * pool[col] + w_prior * mu0) / (iv + w_prior)
        pool[col + "_post_se"] = np.sqrt(1.0 / (iv + w_prior))

    # Regression-Adjusted Pressure Resilience
    w_fit = 1.0 / (pool["hi_rva100_post_se"] ** 2)
    b1, b0 = np.polyfit(pool["lo_rva100_eb"], pool["hi_rva100_eb"], 1, w=w_fit)
    pool["exp_hi_rva100"] = b0 + b1 * pool["lo_rva100_eb"]
    pool["res_rva100"] = pool["hi_rva100_eb"] - pool["exp_hi_rva100"]
    pool["raw_uplift_eb"] = pool["hi_rva100_eb"] - pool["lo_rva100_eb"]

    w_fit_w = 1.0 / (pool["hi_wpa100_post_se"] ** 2)
    bw1, bw0 = np.polyfit(pool["lo_wpa100_eb"], pool["hi_wpa100_eb"], 1, w=w_fit_w)
    pool["res_wpa100"] = pool["hi_wpa100_eb"] - (bw0 + bw1 * pool["lo_wpa100_eb"])

    pool["pmr_eb"] = (pool["hi_pos_matches"] + 6.0) / (pool["hi_sig_matches"] + 12.0)
    pool["lcb_rva100"] = pool["hi_rva100_eb"] - 1.0 * pool["hi_rva100_post_se"]

    def z(s):
        return (s - s.mean()) / (s.std() + 1e-9)

    pool["z_level"] = 0.65 * z(pool["hi_rva100_eb"]) + 0.35 * z(pool["hi_wpa100_eb"])
    pool["z_resilience"] = 0.65 * z(pool["res_rva100"]) + 0.35 * z(pool["res_wpa100"])
    pool["z_consist"] = 0.50 * z(pool["pmr_eb"]) + 0.50 * z(pool["lcb_rva100"])
    pool["SPRI"] = 0.45 * pool["z_level"] + 0.30 * pool["z_resilience"] + 0.25 * pool["z_consist"]

    pool = pool.sort_values("SPRI", ascending=False).reset_index(drop=True)
    pool["rank"] = pool.index + 1
    return pool, d

print("Computing primary Batters & Bowlers rankings (tau_high=0.70)...")
bat_pool, bat_deliv = compute_full_player_metrics(
    df, "striker_id", "is_ball_faced", "pressure_bat", "rva_bat", "raa_bat", "waa_bat", "wpa_bat_adj", tau_high=0.70
)
bowl_pool, bowl_deliv = compute_full_player_metrics(
    df, "bowler_id", None, "pressure_bowl", "rva_bowl", "raa_bowl", "waa_bowl", "wpa_bowl_adj", tau_high=0.70
)

# ==============================================================================
# 1. SENSITIVITY ANALYSIS (Thresholds, Currencies, Weights, Top-Tier Only)
# ==============================================================================
def run_sensitivity(sub_df, base_pool, id_col, ball_flag, p_col, rva_col, raa_col, waa_col, wpa_col, label):
    base_top10 = set(base_pool.head(10)["player_id"])
    rows = []

    for label_var, kwargs in [
        ("Threshold tau=0.65", {"tau_high": 0.65, "min_hi_balls": 200}),
        ("Threshold tau=0.75", {"tau_high": 0.75, "min_hi_balls": 140, "min_hi_matches": 16}),
        ("Top-Tier Matches Only (IPL+FM T20I)", {"top_tier": True}),
    ]:
        s_df = sub_df[sub_df["is_top_tier_match"]] if kwargs.get("top_tier") else sub_df
        p_var, _ = compute_full_player_metrics(
            s_df, id_col, ball_flag, p_col, rva_col, raa_col, waa_col, wpa_col,
            tau_high=kwargs.get("tau_high", 0.70),
            min_hi_balls=kwargs.get("min_hi_balls", 180),
            min_hi_matches=kwargs.get("min_hi_matches", 20),
        )
        merged = base_pool[["player_id", "SPRI"]].merge(p_var[["player_id", "SPRI"]], on="player_id", suffixes=("_b", "_v"))
        rho, _ = spearmanr(merged["SPRI_b"], merged["SPRI_v"])
        ov = len(base_top10 & set(p_var.head(10)["player_id"]))
        rows.append({"category": label, "variant": label_var, "pool_n": len(p_var), "spearman_rho": round(rho, 4), "top10_overlap": ov})

    # Weighting variants on same pool
    def z(s): return (s - s.mean()) / (s.std() + 1e-9)
    for w_name, sc in [
        ("Pure High-Pressure Level (z_level)", base_pool["z_level"]),
        ("Pure RVA (hi_rva100_eb)", z(base_pool["hi_rva100_eb"])),
        ("Pure WPA (hi_wpa100_eb)", z(base_pool["hi_wpa100_eb"])),
        ("Equal Weights (33/33/33)", (base_pool["z_level"] + base_pool["z_resilience"] + base_pool["z_consist"]) / 3.0),
        ("High Uplift Weight (30/50/20)", 0.30 * base_pool["z_level"] + 0.50 * base_pool["z_resilience"] + 0.20 * base_pool["z_consist"]),
    ]:
        rho, _ = spearmanr(base_pool["SPRI"], sc)
        top10_v = set(base_pool.assign(tmp=sc).sort_values("tmp", ascending=False).head(10)["player_id"])
        rows.append({"category": label, "variant": w_name, "pool_n": len(base_pool), "spearman_rho": round(rho, 4), "top10_overlap": len(base_top10 & top10_v)})

    res = pd.DataFrame(rows)
    print(f"\n=== SENSITIVITY ANALYSIS ({label}) ===")
    print(res.to_string(index=False))
    return res

sens_bat = run_sensitivity(df, bat_pool, "striker_id", "is_ball_faced", "pressure_bat", "rva_bat", "raa_bat", "waa_bat", "wpa_bat_adj", "BATTERS")
sens_bowl = run_sensitivity(df, bowl_pool, "bowler_id", None, "pressure_bowl", "rva_bowl", "raa_bowl", "waa_bowl", "wpa_bowl_adj", "BOWLERS")

# ==============================================================================
# 2. MATCH-CLUSTERED BOOTSTRAP STABILITY (1,000 ITERATIONS ON ELIGIBLE POOL)
# ==============================================================================
def run_match_bootstrap(deliv_df, pool_df, id_col, p_col, rva_col, wpa_col, label, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    elig_ids = set(pool_df["player_id"])
    d = deliv_df[deliv_df[id_col].isin(elig_ids)].copy()
    d["is_high"] = d[p_col] >= 0.70
    d["is_low"] = d[p_col] < 0.55

    # Pre-aggregate per (player_id, match_id) for fast vectorized match-level bootstrap
    hi_d = d[d["is_high"]].copy()
    hi_d["w"] = hi_d[p_col] ** 2
    hi_d["wrva"] = hi_d["w"] * hi_d[rva_col]
    hi_d["wwpa"] = hi_d["w"] * hi_d[wpa_col]
    hi_m = hi_d.groupby([id_col, "match_id"]).agg(
        n=("w", "count"), w=("w", "sum"), wrva=("wrva", "sum"), wwpa=("wwpa", "sum")
    ).reset_index()

    lo_m = d[d["is_low"]].groupby([id_col, "match_id"]).agg(
        n=(rva_col, "count"), rva=(rva_col, "sum"), wpa=(wpa_col, "sum")
    ).reset_index()

    # Fast lookup arrays per player
    pids = pool_df["player_id"].values
    hi_by_p = {pid: grp for pid, grp in hi_m.groupby(id_col)}
    lo_by_p = {pid: grp for pid, grp in lo_m.groupby(id_col)}

    # Prior variances from full pool
    mu0_hr = pool_df["hi_rva100_eb"].mean()
    mu0_lr = pool_df["lo_rva100_eb"].mean()
    mu0_hw = pool_df["hi_wpa100_eb"].mean()
    mu0_lw = pool_df["lo_wpa100_eb"].mean()
    tau2_hr = max(np.var(pool_df["hi_rva100"]) - np.mean(pool_df["hi_se_rva100"] ** 2), 25.0)
    tau2_lr = max(np.var(pool_df["lo_rva100"]) - np.mean(pool_df["lo_se_rva100"] ** 2), 25.0)
    tau2_hw = max(np.var(pool_df["hi_wpa100"]) - np.mean(pool_df["hi_se_wpa100"] ** 2), 0.03)
    tau2_lw = max(np.var(pool_df["lo_wpa100"]) - np.mean(pool_df["lo_se_wpa100"] ** 2), 0.03)

    ranks_mat = np.zeros((n_boot, len(pids)), dtype=int)
    spri_mat = np.zeros((n_boot, len(pids)), dtype=float)

    # Pre-extract numpy arrays for speed
    hi_arrs = {}
    for pid in pids:
        g = hi_by_p[pid]
        hi_arrs[pid] = (g["n"].values, g["w"].values, g["wrva"].values, g["wwpa"].values)
    lo_arrs = {}
    for pid in pids:
        g = lo_by_p[pid]
        lo_arrs[pid] = (g["n"].values, g["rva"].values, g["wpa"].values)

    for b in range(n_boot):
        hr_eb = np.zeros(len(pids))
        hr_se = np.zeros(len(pids))
        lr_eb = np.zeros(len(pids))
        hw_eb = np.zeros(len(pids))
        lw_eb = np.zeros(len(pids))
        pmr = np.zeros(len(pids))

        for i, pid in enumerate(pids):
            n_a, w_a, wr_a, ww_a = hi_arrs[pid]
            m_h = len(n_a)
            idx_h = rng.integers(0, m_h, size=m_h)
            w_s = w_a[idx_h]
            W = w_s.sum()
            mu_r = (wr_a[idx_h].sum() / W) * 100.0
            mu_w = (ww_a[idx_h].sum() / W) * 100.0
            r_m = (wr_a[idx_h] / w_s) * 100.0
            w_m = (ww_a[idx_h] / w_s) * 100.0
            se_r = np.sqrt(max((m_h / (m_h - 1)) * np.sum((w_s ** 2) * ((r_m - mu_r) ** 2)) / (W ** 2), 1e-4))
            se_w = np.sqrt(max((m_h / (m_h - 1)) * np.sum((w_s ** 2) * ((w_m - mu_w) ** 2)) / (W ** 2), 1e-6))

            iv_r = 1.0 / (se_r ** 2)
            hr_eb[i] = (iv_r * mu_r + (1.0 / tau2_hr) * mu0_hr) / (iv_r + 1.0 / tau2_hr)
            hr_se[i] = np.sqrt(1.0 / (iv_r + 1.0 / tau2_hr))
            iv_w = 1.0 / (se_w ** 2)
            hw_eb[i] = (iv_w * mu_w + (1.0 / tau2_hw) * mu0_hw) / (iv_w + 1.0 / tau2_hw)

            sig = n_a[idx_h] >= 3
            n_sig = sig.sum()
            pos_sig = (wr_a[idx_h][sig] > 0).sum() if n_sig > 0 else 0
            pmr[i] = (pos_sig + 6.0) / (n_sig + 12.0)

            ln_a, lr_a, lw_a = lo_arrs[pid]
            m_l = len(ln_a)
            idx_l = rng.integers(0, m_l, size=m_l)
            ln_s = ln_a[idx_l]
            LN = ln_s.sum()
            mu_lr = (lr_a[idx_l].sum() / LN) * 100.0
            mu_lw = (lw_a[idx_l].sum() / LN) * 100.0
            lr_m = (lr_a[idx_l] / ln_s) * 100.0
            lw_m = (lw_a[idx_l] / ln_s) * 100.0
            se_lr = np.sqrt(max((m_l / (m_l - 1)) * np.sum((ln_s ** 2) * ((lr_m - mu_lr) ** 2)) / (LN ** 2), 1e-4))
            se_lw = np.sqrt(max((m_l / (m_l - 1)) * np.sum((ln_s ** 2) * ((lw_m - mu_lw) ** 2)) / (LN ** 2), 1e-6))
            iv_lr = 1.0 / (se_lr ** 2)
            lr_eb[i] = (iv_lr * mu_lr + (1.0 / tau2_lr) * mu0_lr) / (iv_lr + 1.0 / tau2_lr)
            iv_lw = 1.0 / (se_lw ** 2)
            lw_eb[i] = (iv_lw * mu_lw + (1.0 / tau2_lw) * mu0_lw) / (iv_lw + 1.0 / tau2_lw)

        b1_r, b0_r = np.polyfit(lr_eb, hr_eb, 1, w=1.0 / (hr_se ** 2))
        res_r = hr_eb - (b0_r + b1_r * lr_eb)
        b1_w, b0_w = np.polyfit(lw_eb, hw_eb, 1)
        res_w = hw_eb - (b0_w + b1_w * lw_eb)
        lcb = hr_eb - 1.0 * hr_se

        def z(a): return (a - a.mean()) / (a.std() + 1e-9)
        spri = 0.45 * (0.65 * z(hr_eb) + 0.35 * z(hw_eb)) + 0.30 * (0.65 * z(res_r) + 0.35 * z(res_w)) + 0.25 * (0.50 * z(pmr) + 0.50 * z(lcb))
        spri_mat[b] = spri
        order = np.argsort(-spri)
        r_vec = np.empty(len(pids), dtype=int)
        r_vec[order] = np.arange(1, len(pids) + 1)
        ranks_mat[b] = r_vec

    boot_summary = pd.DataFrame({
        "player_id": pids,
        "player_name": pool_df["player_name"].values,
        "orig_rank": pool_df["rank"].values,
        "orig_spri": pool_df["SPRI"].values,
        "spri_ci_low": np.quantile(spri_mat, 0.05, axis=0),
        "spri_ci_high": np.quantile(spri_mat, 0.95, axis=0),
        "median_boot_rank": np.median(ranks_mat, axis=0),
        "p_top10": (ranks_mat <= 10).mean(axis=0),
        "p_top15": (ranks_mat <= 15).mean(axis=0),
    })
    print(f"\n=== 1,000-RUN MATCH-CLUSTERED BOOTSTRAP STABILITY ({label}, Top 15) ===")
    print(boot_summary.head(15).round(3).to_string(index=False))
    return boot_summary

boot_bat = run_match_bootstrap(bat_deliv, bat_pool, "striker_id", "pressure_bat", "rva_bat", "wpa_bat_adj", "BATTERS")
boot_bowl = run_match_bootstrap(bowl_deliv, bowl_pool, "bowler_id", "pressure_bowl", "rva_bowl", "wpa_bowl_adj", "BOWLERS")

# ==============================================================================
# 3. OUT-OF-TIME CHRONOLOGICAL VALIDATION (2022-2024 -> 2025-2026)
# ==============================================================================
def temporal_validation(sub_df, id_col, ball_flag, p_col, rva_col, wpa_col, label):
    d1 = sub_df[sub_df["year"] <= 2024]
    d2 = sub_df[sub_df["year"] >= 2025]
    p1, _ = compute_full_player_metrics(d1, id_col, ball_flag, p_col, rva_col, rva_col, rva_col, wpa_col,
                                        min_top_tier_balls=200, min_hi_balls=75, min_hi_matches=8, min_lo_balls=60)
    p2, _ = compute_full_player_metrics(d2, id_col, ball_flag, p_col, rva_col, rva_col, rva_col, wpa_col,
                                        min_top_tier_balls=150, min_hi_balls=60, min_hi_matches=8, min_lo_balls=50)
    m = p1[["player_id", "player_name", "SPRI", "hi_rva100_eb", "lo_rva100_eb"]].merge(
        p2[["player_id", "hi_rva100", "hi_wpa100", "hi_balls", "SPRI"]], on="player_id", suffixes=("_22_24", "_25_26")
    )
    rho_spri, p_spri = spearmanr(m["SPRI_22_24"], m["hi_rva100"])
    rho_lo, p_lo = spearmanr(m["lo_rva100_eb"], m["hi_rva100"])
    rho_spri_spri, p_ss = spearmanr(m["SPRI_22_24"], m["SPRI_25_26"])
    print(f"\n=== OUT-OF-TIME VALIDATION 2022-24 -> 2025-26 ({label}, N={len(m)} players) ===")
    print(f"  2022-24 SPRI vs 2025-26 High-Pressure RVA: Spearman r = {rho_spri:.3f} (p = {p_spri:.4f})")
    print(f"  2022-24 Ordinary Baseline vs 2025-26 High-Pressure RVA: Spearman r = {rho_lo:.3f} (p = {p_lo:.4f})")
    print(f"  2022-24 SPRI vs 2025-26 SPRI: Spearman r = {rho_spri_spri:.3f} (p = {p_ss:.4f})")

temporal_validation(df, "striker_id", "is_ball_faced", "pressure_bat", "rva_bat", "wpa_bat_adj", "BATTERS")
temporal_validation(df, "bowler_id", None, "pressure_bowl", "rva_bowl", "wpa_bowl_adj", "BOWLERS")

# ==============================================================================
# 4. CONTRAST PROFILES: THRIVERS VS PRESSURE-DIPPERS (NEGATIVE RESILIENCE)
# ==============================================================================
print("\n=== BATTERS WHO DIP MOST UNDER PRESSURE (lo_rva100_eb > +2, lowest res_rva100) ===")
dip_bat = bat_pool[bat_pool["lo_rva100_eb"] > 2.0].sort_values("res_rva100").head(8)
print(dip_bat[["player_name", "hi_balls", "lo_rva100_eb", "hi_rva100_eb", "res_rva100", "raw_uplift_eb", "pmr_eb", "SPRI"]].round(3).to_string(index=False))

print("\n=== BOWLERS WHO DIP MOST UNDER PRESSURE (lo_rva100_eb > +2, lowest res_rva100) ===")
dip_bowl = bowl_pool[bowl_pool["lo_rva100_eb"] > 2.0].sort_values("res_rva100").head(8)
print(dip_bowl[["player_name", "hi_balls", "lo_rva100_eb", "hi_rva100_eb", "res_rva100", "raw_uplift_eb", "pmr_eb", "SPRI"]].round(3).to_string(index=False))

# Save all final tables
bat_pool.merge(boot_bat[["player_id", "spri_ci_low", "spri_ci_high", "median_boot_rank", "p_top10", "p_top15"]], on="player_id").to_csv(
    os.path.join(RANK_DIR, "final_batters_rankings.csv"), index=False
)
bowl_pool.merge(boot_bowl[["player_id", "spri_ci_low", "spri_ci_high", "median_boot_rank", "p_top10", "p_top15"]], on="player_id").to_csv(
    os.path.join(RANK_DIR, "final_bowlers_rankings.csv"), index=False
)
sens_bat.to_csv(os.path.join(OUT_DIR, "sensitivity_batters.csv"), index=False)
sens_bowl.to_csv(os.path.join(OUT_DIR, "sensitivity_bowlers.csv"), index=False)
print("\nSaved final rankings and validation files!")

