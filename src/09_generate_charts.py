import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

FIG_DIR = r"d:\Rajasthan Royals\outputs\figures"
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#FAFAFC",
    "axes.edgecolor": "#CCCCCC",
    "axes.grid": True,
    "grid.color": "#E6E6EA",
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
})

RR_PINK = "#E5197A"
RR_BLUE = "#004B8D"
RR_DARK = "#1B2138"
TEAL = "#0D9488"
SLATE = "#64748B"

bat_val = pd.read_csv(r"d:\Rajasthan Royals\outputs\pressure\validation_bat_pressure_deciles.csv")
bowl_val = pd.read_csv(r"d:\Rajasthan Royals\outputs\pressure\validation_bowl_pressure_deciles.csv")
bat_rank = pd.read_csv(r"d:\Rajasthan Royals\outputs\rankings\final_batters_rankings.csv")
bowl_rank = pd.read_csv(r"d:\Rajasthan Royals\outputs\rankings\final_bowlers_rankings.csv")
df = pd.read_pickle(r"d:\Rajasthan Royals\outputs\processed\deliveries_modeled.pkl")

# ==============================================================================
# FIGURE 1: Empirical Pressure Validation Across Deciles (D1 -> D10)
# ==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=220)
x = np.arange(1, 11)

ax1 = axes[0]
l1 = ax1.plot(x, bat_val["actual_abs_wpa"] * 100, marker="o", lw=2.4, color=RR_PINK, label="Batting Pressure: Actual |ΔWP| (% per ball)")
l2 = ax1.plot(x, bowl_val["actual_abs_wpa"] * 100, marker="s", lw=2.4, color=RR_BLUE, label="Bowling Pressure: Actual |ΔWP| (% per ball)")
ax1_r = ax1.twinx()
l3 = ax1_r.bar(x, bat_val["mean_li"], alpha=0.18, color=RR_DARK, width=0.55, label="Mean Counterfactual Leverage Index (LI)")
ax1_r.grid(False)
ax1.set_xticks(x)
ax1.set_xticklabels([f"D{i}" for i in x])
ax1.set_xlabel("Pressure Index Decile (D1 = Lowest Pressure, D10 = Highest Pressure)")
ax1.set_ylabel("Mean Actual Match Swing |ΔWP| (% per ball)", color=RR_DARK)
ax1_r.set_ylabel("Mean Leverage Index (LI)", color=SLATE)
ax1.set_title("A. Match Outcome Impact Scales 20–25× From D1 to D10")
lines = l1 + l2 + [l3]
ax1.legend(lines, [l.get_label() for l in lines], loc="upper left", fontsize=8.5, frameon=True, facecolor="white")

ax2 = axes[1]
b1 = ax2.plot(x, bat_val["wkt_pct"], marker="o", lw=2.4, color=RR_PINK, label="Batter Dismissal Rate (% per ball)")
b2 = ax2.plot(x, bowl_val["extra_err_pct"], marker="^", lw=2.4, color=TEAL, label="Bowler Wide/No-Ball Error Rate (% per ball)")
ax2_r = ax2.twinx()
b3 = ax2_r.plot(x, bat_val["rpo"], marker="D", lw=2.0, ls="--", color=RR_BLUE, label="Batting Scoring Rate (RPO, right axis)")
ax2_r.grid(False)
ax2.set_xticks(x)
ax2.set_xticklabels([f"D{i}" for i in x])
ax2.set_xlabel("Pressure Index Decile")
ax2.set_ylabel("Event Rate (% per delivery)")
ax2_r.set_ylabel("Scoring Rate (Runs per Over)", color=RR_BLUE)
ax2.set_title("B. Behavioral Signatures: Rising Wickets, Bowler Errors & RPO Squeeze")
lines2 = b1 + b2 + b3
ax2.legend(lines2, [l.get_label() for l in lines2], loc="lower left", fontsize=8.5, frameon=True, facecolor="white")

plt.suptitle("Empirical Validation of the Multi-Dimensional Pressure Index (200,961 Deliveries)", fontsize=13.5, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig1_pressure_validation.png"), bbox_inches="tight")
plt.close()

# ==============================================================================
# FIGURE 2: Win Probability Calibration & Smooth DLS Wicket Value Curve
# ==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=220)

ax1 = axes[0]
dec_df = df[df["bat_team_won"].notna()].copy()
for inn, col, lbl in [(1, RR_BLUE, "Innings 1 WP (AUC=0.706, ECE=0.030)"),
                      (2, RR_PINK, "Innings 2 WP (AUC=0.883, ECE=0.007)")]:
    sub = dec_df[dec_df["innings"] == inn]
    prob_true, prob_pred = calibration_curve(sub["bat_team_won"].astype(int), sub["wp_before"], n_bins=12, strategy="quantile")
    ax1.plot(prob_pred, prob_true, marker="o", lw=2.2, color=col, label=lbl)
ax1.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.6, label="Perfect Calibration")
ax1.set_xlabel("Predicted Win Probability (5-Fold Match-Grouped OOF)")
ax1.set_ylabel("Observed Match Win Frequency")
ax1.set_title("A. Win Probability Calibration Reliability Diagram")
ax1.legend(loc="upper left", fontsize=8.5, frameon=True, facecolor="white")

ax2 = axes[1]
balls_grid = np.linspace(6, 120, 40)
c, alpha, beta, par = 1.0041, 0.6952, 1.5686, 175.0
for w_hand, col in [(9, RR_DARK), (7, RR_BLUE), (5, RR_PINK), (3, TEAL)]:
    fb = balls_grid / 120.0
    r_w = c * par * (fb ** alpha) * (1.0 - (1.0 - w_hand / 10.0) ** beta)
    r_wm1 = c * par * (fb ** alpha) * (1.0 - (1.0 - (w_hand - 1) / 10.0) ** beta)
    vw = np.clip(r_w - r_wm1, 2.0, 13.5)
    ax2.plot(120 - balls_grid, vw, lw=2.3, color=col, label=f"{w_hand} Wickets in Hand (10-{w_hand} down)")
ax2.set_xlabel("Legal Balls Bowled in Innings (0 = Start, 120 = End)")
ax2.set_ylabel("Marginal Run Value of 1 Wicket (V_W, runs)")
ax2.set_title("B. Empirical State-Dependent Wicket Run Value (Par=175)")
ax2.legend(loc="upper right", fontsize=8.5, frameon=True, facecolor="white")

plt.suptitle("Core Quantitative Engines: Calibrated Win Probability & Dynamic Wicket Valuation", fontsize=13.5, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig2_wp_calibration_and_vw.png"), bbox_inches="tight")
plt.close()

# ==============================================================================
# FIGURE 3: Ordinary Baseline vs High-Pressure Scatter (Zero Overlapping Labels)
# ==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.8), dpi=220)

bat_offsets = {
    "Yashasvi Jaiswal": (-0.6, 0.75),
    "Suryakumar Yadav": (-1.2, 0.85),
    "Tilak Varma": (-0.3, 0.85),
    "Dinesh Karthik": (0.55, 0.55),
    "Nicholas Pooran": (0.75, -0.35),
    "Mark Chapman": (-2.9, 0.45),
    "Heinrich Klaasen": (-0.6, 0.85),
    "Towhid Hridoy": (-2.4, -0.75),
    "Tim David": (0.35, -0.80),
    "Andre Russell": (0.35, -0.95),
    "Rishabh Pant": (0.3, -0.75),
    "David Warner": (0.35, -0.60),
    "Jos Buttler": (0.35, 0.35),
    "Aiden Markram": (-1.5, -0.85),
}

bowl_offsets = {
    "Axar Patel": (-0.8, 1.15),
    "Matheesha Pathirana": (-2.2, 1.05),
    "Josh Hazlewood": (0.45, 1.25),
    "Mujeeb Ur Rahman": (1.1, 0.55),
    "Shaheen Afridi": (-2.9, 0.95),
    "Jasprit Bumrah": (-0.9, 1.10),
    "Bhuvneshwar Kumar": (0.55, -0.85),
    "Mitchell Santner": (-2.8, -1.25),
    "Jacob Duffy": (0.65, -0.95),
    "Rashid Khan": (0.65, -1.05),
    "Hardik Pandya": (-2.6, 0.45),
    "Tushar Deshpande": (-3.5, -0.75),
    "Ravi Bishnoi": (0.6, 0.55),
    "Azmatullah Omarzai": (0.65, -0.65),
}

for ax, pool, title, dip_names, offsets in [
    (axes[0], bat_rank, "A. Batters: Ordinary Baseline vs High-Pressure RVA (N=102)",
     ["Rishabh Pant", "David Warner", "Jos Buttler", "Aiden Markram"], bat_offsets),
    (axes[1], bowl_rank, "B. Bowlers: Ordinary Baseline vs High-Pressure RVA (N=102)",
     ["Hardik Pandya", "Tushar Deshpande", "Ravi Bishnoi", "Azmatullah Omarzai"], bowl_offsets),
]:
    top10 = pool.head(10)
    rest = pool.iloc[10:]
    ax.scatter(rest["lo_rva100_eb"], rest["hi_rva100_eb"], s=45, color=SLATE, alpha=0.45, edgecolors="none", label="Other Eligible Players")
    ax.scatter(top10["lo_rva100_eb"], top10["hi_rva100_eb"], s=95, color=RR_PINK, edgecolors=RR_DARK, lw=1.0, zorder=5, label="Top 10 SPRI Rank")

    dips = pool[pool["player_name"].isin(dip_names)]
    ax.scatter(dips["lo_rva100_eb"], dips["hi_rva100_eb"], s=75, color="#F59E0B", marker="v", edgecolors=RR_DARK, lw=0.8, zorder=4, label="Notable Pressure Dippers")

    xs = np.linspace(pool["lo_rva100_eb"].min() - 1, pool["lo_rva100_eb"].max() + 1, 100)
    w_fit = 1.0 / (pool["hi_rva100_post_se"] ** 2)
    b1, b0 = np.polyfit(pool["lo_rva100_eb"], pool["hi_rva100_eb"], 1, w=w_fit)
    ax.plot(xs, b0 + b1 * xs, color=RR_BLUE, lw=2.2, label=f"Expected Pressure Line (slope={b1:.2f})")
    ax.plot(xs, xs, color="gray", ls=":", lw=1.3, alpha=0.7, label="1:1 Baseline Parity Line")
    ax.axhline(0, color="#999999", lw=0.7, alpha=0.5)
    ax.axvline(0, color="#999999", lw=0.7, alpha=0.5)

    for _, r in top10.iterrows():
        sname = r["player_name"].replace("Namboori ", "")
        dx, dy = offsets.get(sname, (0.3, 0.3))
        ax.annotate(
            f"#{int(r['rank'])} {sname}",
            xy=(r["lo_rva100_eb"], r["hi_rva100_eb"]),
            xytext=(r["lo_rva100_eb"] + dx, r["hi_rva100_eb"] + dy),
            fontsize=8.0, fontweight="bold", color=RR_DARK,
            arrowprops=dict(arrowstyle="-", color="#999999", lw=0.6, shrinkA=0, shrinkB=4),
        )
    for _, r in dips.iterrows():
        sname = r["player_name"]
        dx, dy = offsets.get(sname, (0.3, -0.5))
        ax.annotate(
            sname,
            xy=(r["lo_rva100_eb"], r["hi_rva100_eb"]),
            xytext=(r["lo_rva100_eb"] + dx, r["hi_rva100_eb"] + dy),
            fontsize=7.8, color="#92400E",
            arrowprops=dict(arrowstyle="-", color="#D97706", lw=0.6, shrinkA=0, shrinkB=4),
        )

    ax.set_xlabel("Usual Ordinary Baseline RVA (EB runs per 100 balls, P < 0.55)")
    ax.set_ylabel("High-Pressure RVA (EB runs per 100 balls, P ≥ 0.70)")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8.0, frameon=True, facecolor="white")

axes[0].set_xlim(-3.5, 13.0)
axes[0].set_ylim(-2.5, 11.8)
axes[1].set_xlim(-5.5, 13.5)
axes[1].set_ylim(-11.8, 10.8)

plt.suptitle("High-Pressure Performance vs Usual Baseline: Identifying Risers & Elite Anchors", fontsize=13.5, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig3_baseline_vs_pressure_scatter.png"), bbox_inches="tight")
plt.close()

# ==============================================================================
# FIGURE 4: Top-10 Batters & Bowlers SPRI Decomposition + 90% Bootstrap CI
# ==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.2), dpi=220)

for ax, pool, title in [
    (axes[0], bat_rank.head(10).iloc[::-1], "A. Top 10 Batters Under Pressure (SPRI & 90% Bootstrap CI)"),
    (axes[1], bowl_rank.head(10).iloc[::-1], "B. Top 10 Bowlers Under Pressure (SPRI & 90% Bootstrap CI)"),
]:
    y = np.arange(len(pool))
    c_lvl = 0.45 * pool["z_level"]
    c_res = 0.30 * pool["z_resilience"]
    c_con = 0.25 * pool["z_consist"]

    ax.barh(y, c_lvl, color=RR_BLUE, height=0.62, label="High-Pressure Output (45%)")
    ax.barh(y, c_res, left=c_lvl, color=RR_PINK, height=0.62, label="Baseline-Relative Resilience (30%)")
    ax.barh(y, c_con, left=c_lvl + c_res, color=TEAL, height=0.62, label="Match Consistency & LCB (25%)")

    err_low = np.maximum(pool["SPRI"] - pool["spri_ci_low"], 0)
    err_high = np.maximum(pool["spri_ci_high"] - pool["SPRI"], 0)
    ax.errorbar(pool["SPRI"], y, xerr=[err_low, err_high], fmt="none", ecolor=RR_DARK, elinewidth=1.4, capsize=3.5, zorder=5, label="90% Match-Clustered Bootstrap CI")

    labels = [f"#{int(r['rank'])} {r['player_name'].replace('Namboori ', '')} (n={int(r['hi_balls'])})" for _, r in pool.iterrows()]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9, fontweight="bold")
    ax.set_xlabel("SupeRR Pressure Reliability Index (SPRI, z-score composite)")
    ax.set_title(title)
    ax.set_xlim(-0.2, 3.05)

handles, lbls = axes[0].get_legend_handles_labels()
fig.legend(handles, lbls, loc="lower center", ncol=4, fontsize=9.0, frameon=True, facecolor="white", bbox_to_anchor=(0.5, -0.04))
plt.suptitle("Final Round 1 Top-10 Selections: Component Breakdown & Uncertainty Bounds", fontsize=13.5, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig4_top10_profiles_and_ci.png"), bbox_inches="tight")
plt.close()

# ==============================================================================
# FIGURE 5: Multi-Format (IPL vs T20I) & Consistency Comparison for Top 10
# ==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.5), dpi=220)

for ax, pool, title in [
    (axes[0], bat_rank.head(10), "A. Top 10 Batters: High-Pressure Metrics & Multi-Format Split"),
    (axes[1], bowl_rank.head(10), "B. Top 10 Bowlers: High-Pressure Metrics & Multi-Format Split"),
]:
    p = pool.copy()
    names = [f"#{int(r['rank'])} {r['player_name'].replace('Namboori ', '')}" for _, r in p.iterrows()]

    # Display 5 clean metrics: Hi-P RVA(EB), Resilience, IPL Hi-P RVA (raw w/ balls), T20I Hi-P RVA (raw w/ balls), Pos Match %
    # Color strictly by standardized column percentile / centered RdYlGn with masked N/A!
    vals_num = np.column_stack([
        p["hi_rva100_eb"].values,
        p["res_rva100"].values,
        p["hi_ipl_rva100"].values,
        p["hi_t20i_rva100"].values,
        p["pmr_eb"].values * 100.0,
    ])
    # Normalized display matrix [0, 1] per column for clean color comparability
    norm_mat = np.zeros_like(vals_num)
    for j in range(5):
        col = vals_num[:, j]
        if j < 4:
            # 0 -> neutral yellow (0.5), positive -> green, negative -> red
            scale = 10.0 if j < 2 else 35.0
            norm_mat[:, j] = np.clip(0.5 + col / (2.0 * scale), 0.05, 0.95)
        else:
            norm_mat[:, j] = np.clip(0.5 + (col - 50.0) / 40.0, 0.05, 0.95)

    masked = np.ma.masked_invalid(norm_mat)
    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad(color="#E2E8F0")
    ax.imshow(masked, cmap=cmap, aspect="auto", vmin=0.0, vmax=1.0)

    ax.set_xticks(range(5))
    ax.set_xticklabels(["Hi-P RVA/100\n(EB)", "Baseline-Adj.\nResilience", "IPL Hi-P\nRVA/100 (balls)", "T20I Hi-P\nRVA/100 (balls)", "Pos. Match\nRate (EB)"], fontsize=8.3)
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names, fontsize=8.8, fontweight="bold")
    ax.set_title(title)
    ax.grid(False)

    ipl_b = p["hi_ipl_balls"].fillna(0).astype(int).values
    t20i_b = p["hi_t20i_balls"].fillna(0).astype(int).values
    for i in range(len(names)):
        for j in range(5):
            v = vals_num[i, j]
            if np.isnan(v):
                txt = "No Data"
                col_t = "#475569"
            elif j == 2:
                txt = f"{v:+.1f}\n(n={ipl_b[i]})"
                col_t = "black"
            elif j == 3:
                txt = f"{v:+.1f}\n(n={t20i_b[i]})"
                col_t = "black"
            elif j == 4:
                txt = f"{v:.1f}%"
                col_t = "black"
            else:
                txt = f"{v:+.1f}"
                col_t = "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.8, fontweight="bold", color=col_t)

plt.suptitle("Cross-Format Robustness & Match Consistency Across Top-10 Selections", fontsize=13.5, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig5_format_and_consistency.png"), bbox_inches="tight")
plt.close()

print("Regenerated all 5 polished figures in:", FIG_DIR)
