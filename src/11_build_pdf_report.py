import os
import pandas as pd
import numpy as np

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, HRFlowable
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY

# Load verified outputs dynamically
bat_rank = pd.read_csv(r"d:\Rajasthan Royals\outputs\rankings\final_batters_rankings.csv")
bowl_rank = pd.read_csv(r"d:\Rajasthan Royals\outputs\rankings\final_bowlers_rankings.csv")
bat_val = pd.read_csv(r"d:\Rajasthan Royals\outputs\pressure\validation_bat_pressure_deciles.csv")
bowl_val = pd.read_csv(r"d:\Rajasthan Royals\outputs\pressure\validation_bowl_pressure_deciles.csv")
sens_bat = pd.read_csv(r"d:\Rajasthan Royals\outputs\validation\sensitivity_batters.csv")
sens_bowl = pd.read_csv(r"d:\Rajasthan Royals\outputs\validation\sensitivity_bowlers.csv")

FIG_DIR = r"d:\Rajasthan Royals\outputs\figures"
OUT_PDF = r"d:\Rajasthan Royals\SupeRR_Selector_Round1_Methodology_Report.pdf"
OUT_MD = r"d:\Rajasthan Royals\SupeRR_Selector_Round1_Methodology_Report.md"
OUT_SUB = r"d:\Rajasthan Royals\outputs\rankings\SUBMISSION_TOP10.csv"

# Save clean Top-10 submission sheet
sub_rows = []
for _, r in bat_rank.head(10).iterrows():
    sub_rows.append({
        "Category": "Batter", "Rank": int(r["rank"]), "Player_Name": r["player_name"],
        "Player_ID": int(r["player_id"]), "SPRI_Score": round(r["SPRI"], 3),
        "Hi_Balls": int(r["hi_balls"]), "Hi_Matches": int(r["hi_matches"]),
        "Hi_RVA100_EB": round(r["hi_rva100_eb"], 2), "Baseline_RVA100_EB": round(r["lo_rva100_eb"], 2),
        "Resilience_RVA100": round(r["res_rva100"], 2), "Pos_Match_Rate": round(r["pmr_eb"], 3),
    })
for _, r in bowl_rank.head(10).iterrows():
    sub_rows.append({
        "Category": "Bowler", "Rank": int(r["rank"]), "Player_Name": r["player_name"],
        "Player_ID": int(r["player_id"]), "SPRI_Score": round(r["SPRI"], 3),
        "Hi_Balls": int(r["hi_balls"]), "Hi_Matches": int(r["hi_matches"]),
        "Hi_RVA100_EB": round(r["hi_rva100_eb"], 2), "Baseline_RVA100_EB": round(r["lo_rva100_eb"], 2),
        "Resilience_RVA100": round(r["res_rva100"], 2), "Pos_Match_Rate": round(r["pmr_eb"], 3),
    })
pd.DataFrame(sub_rows).to_csv(OUT_SUB, index=False)

# ==============================================================================
# 1. WRITE COMPREHENSIVE MARKDOWN METHODOLOGY REPORT
# ==============================================================================
bat_md_rows = []
for _, r in bat_rank.head(10).iterrows():
    bat_md_rows.append(
        f"| {int(r['rank'])} | **{r['player_name']}** | {int(r['ipl_matches'])} / {int(r['t20i_matches'])} | "
        f"{int(r['hi_balls'])} ({int(r['hi_matches'])}m) | {r['lo_rva100_eb']:+.2f} | {r['hi_rva100_eb']:+.2f} | "
        f"{r['res_rva100']:+.2f} | {r['hi_wpa100_eb']:+.3f}% | {r['pmr_eb']*100:.1f}% | **{r['SPRI']:.3f}** | [{r['spri_ci_low']:.2f}, {r['spri_ci_high']:.2f}] |"
    )

bowl_md_rows = []
for _, r in bowl_rank.head(10).iterrows():
    bowl_md_rows.append(
        f"| {int(r['rank'])} | **{r['player_name']}** | {int(r['ipl_matches'])} / {int(r['t20i_matches'])} | "
        f"{int(r['hi_balls'])} ({int(r['hi_matches'])}m) | {r['lo_rva100_eb']:+.2f} | {r['hi_rva100_eb']:+.2f} | "
        f"{r['res_rva100']:+.2f} | {r['hi_wpa100_eb']:+.3f}% | {r['pmr_eb']*100:.1f}% | **{r['SPRI']:.3f}** | [{r['spri_ci_low']:.2f}, {r['spri_ci_high']:.2f}] |"
    )

md_text = f"""# SupeRR Selector Season 2 — Round 1 Methodology Report
## Identifying the Most Reliable Batters and Bowlers Under Pressure (2022–2026)

---

### Executive Summary

This report presents an end-to-end, reproducible, game-theoretic and Empirical Bayes quantitative system — the **SupeRR Pressure Reliability Index (SPRI)** — designed to identify T20 batters and bowlers who deliver reliably in high-pressure match situations compared with their usual baseline performance across **200,999 ball-by-ball deliveries (867 played matches, 141 competitions, Jan 2022–Nov 2026)** spanning IPL and T20I cricket.

Rather than relying on arbitrary heuristics (such as "overs 16–20" or "required run rate > 10"), we construct a **calibrated Win Probability & Counterfactual Leverage engine** coupled with **asymmetric batting/bowling contextual stress indices**, evaluate every delivery via **5-fold match-grouped context-adjusted Expected Runs/Wickets (`xRuns`, `xWkt`) and smooth empirical DLS wicket values**, and rank players using **match-clustered Empirical Bayes shrinkage** across three pillars:
1. **High-Pressure Value Added (`45%`)** — Context-adjusted Net Run Value Added (`RVA/100`) and Win Probability Added (`WPA/100`) on high-pressure balls (`P >= 0.70`, top 30% of deliveries, weighted by `P^2`).
2. **Baseline-Adjusted Pressure Resilience (`30%`)** — Residual outperformance under pressure relative to the player's own ordinary-situation baseline (`P < 0.55`), eliminating both the "weak-baseline inflation" and "elite-baseline regression-to-the-mean" traps.
3. **Match-to-Match Consistency & Lower Confidence Bound (`25%`)** — Beta-Binomial shrunk Positive-Impact Match Rate under pressure and 1-SE cluster-robust Lower Confidence Bound (`LCB`).

#### Final Ordered Top-10 Submissions (Unstop Dropdown Lists)

**Top 10 Batters Under Pressure (Rank 1 to 10):**
1. Yashasvi Jaiswal
2. Suryakumar Yadav
3. Namboori Tilak Varma
4. Dinesh Karthik
5. Nicholas Pooran
6. Mark Chapman
7. Heinrich Klaasen
8. Towhid Hridoy
9. Tim David
10. Andre Russell

**Top 10 Bowlers Under Pressure (Rank 1 to 10):**
1. Axar Patel
2. Matheesha Pathirana
3. Josh Hazlewood
4. Mujeeb Ur Rahman
5. Shaheen Afridi
6. Jasprit Bumrah
7. Bhuvneshwar Kumar
8. Mitchell Santner
9. Jacob Duffy
10. Rashid Khan

---

### 1. Data Audit, Cleaning & Domain Accounting

Deep inspection of all six tables in `Data Pack/` uncovered several critical data-engineering subtleties that directly affect cricket metric integrity:

1. **Exact Runs & Extras Accounting (Verified to 1-Run Match Result Parity):**
   - `runs` records runs off the bat (`0` on all `7,714` wides). `extras_type == "other"` (`264` rows) behaves identically to `"Overstep"` (`576` rows): `extras1_runs = 1` (no-ball penalty), bat runs in `runs`, and occasional `extras_2_runs` (`1` or `4` byes/leg-byes off the no-ball).
   - **Legal delivery indicator**: `is_legal = ~(wide | Overstep | other)`. Verified maximum legal balls in any regular innings = `120` (zero anomalies). Crucially, `innings_ball_num` in the raw CSV increments on extras, so naive `120 - innings_ball_num` miscounts remaining legal balls; we reconstruct cumulative legal balls prior to every delivery.
   - **Batter vs. Bowler credit**: `is_ball_faced = ~wide` (no-balls count as balls faced for batters under official Law 21). `bowler_runs` includes bat runs + wide/no-ball extras, excluding `byes`, `legbyes`, `Penalty`, and `extras_2_runs` off no-balls.
   - **Dismissal & Non-Striker Run-Outs**: `275` wickets are `RunOut`s where `dismissed_player_id == non_striker_id`, and `12` rows have `is_wicket == False` with `dismissed_player_id` populated (retired hurt). Batting dismissals are therefore charged strictly when `is_wicket == True` and `dismissed_player_id == striker_id`, while bowler wickets exclude `RunOut`, `Retire Out`, and `Obstructing the field`.
2. **DLS / Revised-Overs Handling (`25` rain-affected matches):**
   - `first_inn_revised_overs` and `second_inn_revised_overs` store **allocated legal balls** (e.g., `114`, `90`, `66`, `48`, `30`), not overs. Using these directly with `revised_target` yields exact `balls_remaining` and `rrr_before` in shortened games.
3. **Player Identity Disambiguation:**
   - `7` player names map to multiple distinct `player_id`s representing different individuals across countries/teams (e.g., `Rashid Khan`: `4722` [Afghanistan/GT] vs `20172` [Nepal]; `Abdul Samad`: `7446` [SRH/LSG] vs `6922` [Pakistan]). All aggregations strictly group by `player_id`.
4. **Format, Era & Competition Tier Heterogeneity:**
   - The dataset comprises **IPL 2022–2026 (`371` played matches)**, **T20I (`470` played matches)**, and **2 Emerging/Rising Stars T20 tournaments (`30` matches)**, plus **16 abandoned matches** with no deliveries.
   - **Era scoring inflation**: IPL scoring rose +15.7% from **8.54 RPO (2022)** to **9.88 RPO (2026)** (`T20I`: 8.35 in 2022 to 9.15 in 2026).
   - **Opposition tier disparity**: `39` T20I matches are Full Member vs Associate and `15` are Associate vs Associate (including e.g. India vs Japan). We construct **chronological pre-match Dynamic Team Elo ratings** and **strictly causal expanding venue/competition-group par RPOs** so all downstream expectations control for opposition strength, venue scoring environment, and era inflation.

---

### 2. Pressure Definition & Empirical Validation

#### 2.1 Why Simple Heuristics Fail
A rule like "over >= 16 or RRR >= 10" fails on three fronts: (i) chasing 15 off 30 balls with 8 wickets left in over 16 carries almost zero match pressure; (ii) being 18/3 in over 3 of a 1st innings carries intense collapse pressure despite low RRR; and (iii) 1st-innings batting/bowling situations lack an explicit RRR target.

#### 2.2 Game-Theoretic Win Probability & Counterfactual Leverage Engine
We train separate **5-fold Match-Grouped (`GroupKFold` by `match_id`) LightGBM Win Probability (WP) models** with **domain monotonicity constraints** (`runs_before` +1, `wickets_before` -1, `runs_needed` -1, `rrr_before` -1, `elo_diff` +1) and nested **out-of-fold Isotonic Calibration** on decisive regular innings (`199,029` deliveries):
- **Innings 1 WP**: `LogLoss = 0.6338`, `Brier = 0.2176`, `ROC-AUC = 0.7064`, `ECE = 0.0303`.
- **Innings 2 WP**: `LogLoss = 0.4324`, `Brier = 0.1384`, `ROC-AUC = 0.8831`, `ECE = 0.0070`.

At every delivery state $s$, we evaluate the calibrated WP model on **5 counterfactual next-ball outcomes** (Dot, Single, Four, Six, Wicket) to compute **Expected Absolute WP Swing**:
$$\\text{{Swing}}(s) = \\sum_{{k \\in \\{{0,1,4,6,W\\}}}} p_k \\cdot |WP_k(s) - WP(s)|, \\qquad \\text{{LI}}(s) = \\frac{{\\text{{Swing}}(s)}}{{\\text{{median}}(\\text{{Swing}})}}$$

#### 2.3 Asymmetric Batting & Bowling Pressure Indices ($P_{{\\text{{bat}}}}, P_{{\\text{{bowl}}}} \\in [0, 1]$)
Using Leverage Index ($\\text{{LI}}$), Match Closeness $U(s) = 4 \\cdot WP(s) \\cdot (1 - WP(s))$, and a smooth non-garbage-time gate $g(s) = \\min(WP, 1-WP)/0.06$ (clipped to $[0,1]^{{0.6}}$, suppressing dead rubbers where $WP < 0.02$ or $> 0.98$), we construct asymmetric contextual stress composites and percentile-rank normalize them onto $[0, 1]$ across all `200,961` regular deliveries (Super Overs `innings >= 3` assigned `0.99`):

- **Batting Pressure ($P_{{\\text{{bat}}}}$)**:
  - `38%` Normalized Counterfactual Leverage ($C_{{\\text{{lev}}}}$)
  - `22%` Match Closeness $\\times$ Live Gate
  - `24%` Chase Acceleration / Innings-1 Par-Deficit Stress ($C_{{\\text{{chase}}}}$: `rrr_before / prior_venue_rpo` scaled by wicket scarcity $\\sqrt{{5 / \\text{{wkts\\_in\\_hand}}}}$ and innings progress in Inn 2; deficit vs venue par + live death ramp in Inn 1)
  - `16%` Collapse / Wicket-Fragility Stress ($C_{{\\text{{collapse}}}}$: excess wickets lost above `legal_balls / 19`, early top-order collapses $\\ge 2$ down in Powerplay, and 12-ball wicket clusters)
  - `+0.10` Knockout/Playoff Stage Stakes Boost (`Final`, `Qualifier`, `Eliminator`, `Semi Final`, `Super 4`, `knock_out == 1`)

- **Bowling Pressure ($P_{{\\text{{bowl}}}}$)**:
  - `40%` Normalized Counterfactual Leverage ($C_{{\\text{{lev}}}}$)
  - `24%` Match Closeness $\\times$ Live Gate
  - `36%` Defensive Containment & Execution Stress ($C_{{\\text{{bowl\\_exec}}}}$: live chase containment threat in Inn 2 when batting team has resources/closeness; death-overs containment and above-par Powerplay threats in Inn 1)
  - `+0.10` Knockout/Playoff Stage Stakes Boost

#### 2.4 Empirical Validation: Separating Pressure from Ordinary Play
Across pressure deciles (`D1` lowest 10% to `D10` highest 10%), our Pressure Index produces sharp, monotonically verified separation:
- **Match-Outcome Leverage (`|ΔWP|` per ball)**: Scales **19.7×** for Batting (`0.225%` in D1 → `4.445%` in D10) and **24.5×** for Bowling (`0.191%` in D1 → `4.675%` in D10).
- **Behavioral Stress Signatures**:
  - Under **Batting Pressure**, scoring rate is squeezed from **9.21–9.45 RPO (D1–D3)** down to **8.43–8.54 RPO (D9–D10)** while dismissal rate peaks at **6.15%** in D10.
  - Under **Bowling Pressure**, illegal-delivery discipline errors (`wide + no-ball` rate) spike from **3.63%–3.95% (D2–D4)** to **5.14%** in D10 (+35% error surge), while conceded RPO climbs to **8.99** and wicket rate peaks at **5.88%**.
- **High-Stakes Tournaments**: Knockout/playoff deliveries average `P_bat = 0.674` and `P_bowl = 0.635` vs `0.492` / `0.494` in league games.

We define **High Pressure** as $P \\ge 0.70$ (top 30% of deliveries, weighted continuously by $P^2$ so `D9/D10` crunch moments weigh more than borderline `D7` balls) and **Ordinary Baseline** as $P < 0.55$.

---

### 3. Performance Measurement & Rating Construction

#### 3.1 Context-Adjusted Expected Value Models (`xRuns`, `xWkt`)
To ensure a batter hitting on a 220-run 2025 Hyderabad pitch is compared fairly against a batter playing a 150-run 2022 T20I in Trinidad, we fit **5-fold GroupKFold LightGBM baseline models** predicting expected runs and dismissal probability on **context only** (competition group, year, causal venue par RPO, Impact Player rule flag, pre-match team & opponent Elo, innings, over, balls remaining, wickets lost, batting order, batter balls already faced in innings, bowler balls bowled, pace/spin, batter/bowler handedness, bowling sub-type):
- `xRuns_bat` (on `is_ball_faced`): OOF `RMSE = 1.7139`, Mean Pred `1.3952` vs Actual `1.3946`.
- `xWkt_bat` (on `is_ball_faced`): OOF `LogLoss = 0.2083`, `AUC = 0.5995`, Mean Pred `0.0549` vs Actual `0.0546`.
- `xRuns_bowl` (all deliveries): OOF `RMSE = 1.6907`, Mean Pred `1.3899` vs Actual `1.3909`.
- `xWkt_bowl` (all deliveries): OOF `LogLoss = 0.1989`, `AUC = 0.5924`, Mean Pred `0.0513` vs Actual `0.0511`.
- Residuals are unbiased across every phase and format (`Death RVA_bat = +0.0098`, `Middle = -0.0012`, `Powerplay = +0.0041`, `IPL = +0.0039`, `T20I = +0.0017`).

#### 3.2 Empirical Smooth DLS Marginal Wicket Value ($V_W$)
Rather than a naive constant wicket weight, we fit a smooth DLS-style resource surface on Innings 1 remaining runs:
$$\\widehat{{\\text{{RemRuns}}}}(b, w) = c \\cdot \\text{{Par}} \\cdot \\left(\\frac{{b}}{{B_{{\\max}}}}\\right)^\\alpha \\cdot \\left(1 - \\left(1 - \\frac{{w}}{{10}}\\right)^\\beta\\right)$$
yielding fitted parameters $c = 1.0041, \\alpha = 0.6952, \\beta = 1.5686$. The marginal value of a wicket $V_W(b, w) = \\text{{clip}}(\\widehat{{\\text{{RemRuns}}}}(b, w) - \\widehat{{\\text{{RemRuns}}}}(b, w-1), 2.0, 13.5)$ (`1.5` on final ball) smoothly values a Powerplay/Middle wicket at **4.4–12.9 runs** and a late Death wicket at **2.0–5.0 runs**.

Every delivery receives two complementary performance currencies:
1. **Context-Adjusted Net Run Value Added (`RVA`, scaled per 100 balls)**:
   - Batting: $\\text{{RVA}}_{{\\text{{bat}}}} = (\\text{{runs}} - \\widehat{{xR}}_{{\\text{{bat}}}}) + V_W \\cdot (\\widehat{{xW}}_{{\\text{{bat}}}} - \\mathbb{{I}}[\\text{{striker\\_out}}])$
   - Bowling: $\\text{{RVA}}_{{\\text{{bowl}}}} = (\\widehat{{xR}}_{{\\text{{bowl}}}} - \\text{{bowler\\_runs}}) + V_W \\cdot (\\mathbb{{I}}[\\text{{bowler\\_wkt}}] - \\widehat{{xW}}_{{\\text{{bowl}}}})$
2. **Expected-Centered Win Probability Added (`WPA_adj`, scaled per 100 balls)**: Actual $\\Delta WP$ minus the league mean $\\Delta WP$ in that pressure vigintile, ensuring players fighting in desperate chases aren't penalized for baseline negative drift.

#### 3.3 Match-Clustered Empirical Bayes Shrinkage & Baseline-Adjusted Resilience
Deliveries within the same innings are correlated. We aggregate to the `(player_id, match_id)` level and compute **Liang-Zeger cluster-robust Standard Errors** ($\text{{SE}}_{{\\text{{cluster}}}}$) for both High-Pressure (`P >= 0.70`, weighted by $P^2$) and Ordinary Baseline (`P < 0.55`) regimes, then apply **Empirical Bayes (James-Stein) shrinkage** toward the eligible pool inverse-variance mean using DerSimonian-Laird between-player variance $\\tau^2$.

**Crucial Methodological Discovery — Solving the "Elite Baseline Trap":**
In raw data, subtracting ordinary performance from high-pressure performance (`hi - lo`) adds two noisy estimates (`mean SE^2_diff = 447.4` vs `var = 441.5`) and penalizes elite normal-overs superstars (e.g., Jasprit Bumrah `+10.68`, Suryakumar Yadav `+9.70`, Heinrich Klaasen `+10.99`) due to statistical regression to the mean (`Corr(lo, hi) = 0.231` batters, `0.181` bowlers). Therefore, we measure **Baseline-Adjusted Pressure Resilience** via Weighted Least Squares regression of $\\hat{{\\theta}}_{{\\text{{hi}}}}^{{\\text{{EB}}}}$ on $\\hat{{\\theta}}_{{\\text{{lo}}}}^{{\\text{{EB}}}}$:
$$\\text{{Resilience}}(p) = \\hat{{\\theta}}_{{\\text{{hi}}}}^{{\\text{{EB}}}}(p) - \\left(\\hat{{\\alpha}} + \\hat{{\\beta}} \\cdot \\hat{{\\theta}}_{{\\text{{lo}}}}^{{\\text{{EB}}}}(p)\\right)$$
where fitted slopes are $\\hat{{\\beta}} = 0.182$ (Batters) and $\\hat{{\\beta}} = 0.337$ (Bowlers). This rewards *both* players who elevate dramatically from modest baselines (`Yashasvi Jaiswal`, `Axar Patel`) *and* elite superstars who sustain elite delivery under pressure (`Suryakumar Yadav`, `Matheesha Pathirana`, `Jasprit Bumrah`).

#### 3.4 Final Composite Score: SupeRR Pressure Reliability Index (SPRI)
Across the eligible pool ($N = 102$ batters, $N = 102$ bowlers), each pillar is z-scored and combined:
$$\\text{{SPRI}} = 0.45 \\cdot Z_{{\\text{{Level}}}} + 0.30 \\cdot Z_{{\\text{{Resilience}}}} + 0.25 \\cdot Z_{{\\text{{Consistency}}}}$$
- $Z_{{\\text{{Level}}}}$: `65%` High-Pressure `RVA_100 (EB)` + `35%` High-Pressure `WPA_100 (EB)`
- $Z_{{\\text{{Resilience}}}}$: `65%` Baseline-Adjusted `RVA` Resilience + `35%` Baseline-Adjusted `WPA` Resilience
- $Z_{{\\text{{Consistency}}}}$: `50%` Beta-Binomial Positive-Impact Match Rate (`(pos_matches + 6)/(sig_matches + 12)` on matches with $\\ge 3$ pressure balls) + `50%` 1-SE Lower Confidence Bound (`hi_RVA_EB - 1.0 * post_SE`)

---

### 4. Fair Comparisons & Player Eligibility

1. **Sample Size & Top-Tier Exposure Floors**: To eliminate small-sample noise and prevent Associate-only or Emerging-only stat-padding, players must meet **all four criteria**:
   - `>= 500` deliveries in Top-Tier matches (IPL + Full Member vs Full Member T20Is)
   - `>= 180` High-Pressure deliveries (`P >= 0.70`)
   - `>= 20` High-Pressure match appearances
   - `>= 150` Ordinary Baseline deliveries (`P < 0.55`)
   This yields competitive, balanced pools of **`N = 102` batters** and **`N = 102` bowlers**.
2. **IPL + T20I Integration**: Both formats are evaluated on a unified scale after controlling for `comp_group`, `year`, `prior_venue_rpo`, `match_has_impact_rule`, and dynamic pre-match `Elo`. Sensitivity tests restricting strictly to Top-Tier matches yield `Spearman rho = 0.9971` (Batters) and `0.9955` (Bowlers), with **9/10 Top-10 overlap** in both categories.

---

### 5. Evidence for Top 10 Batters & Top 10 Bowlers

#### 5.1 Top 10 Batters Under Pressure

| Rank | Player | IPL / T20I Matches | Hi-P Balls (Matches) | Baseline RVA/100 | Hi-P RVA/100 (EB) | Baseline-Adj. Resilience | Hi-P WPA/100 | Pos. Match Rate | **SPRI** | 90% Bootstrap CI |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(bat_md_rows)}

**Key Analytical Insights (Batters):**
- **#1 Yashasvi Jaiswal (`SPRI = 2.167`, 69.6% Bootstrap Top-10)**: Displays the largest baseline-to-pressure surge in the dataset (`+2.22` ordinary → **`+9.64` RVA/100 under pressure**, `+5.06` resilience). His raw strike rate actually *rises* under pressure (`156.3 → 161.7`) while his dismissal rate *falls* (`4.25% → 3.72%`), highlighted by a `+27.24` RVA clutch knock (`57` off `30` pressure balls) vs SRH in IPL 2024.
- **#2 Suryakumar Yadav (`SPRI = 1.985`, 67.1% Bootstrap Top-10)**: Combines the **largest high-pressure sample among Top-10 batters (`819` balls across `103` matches)** with elite retention (`+9.70` ordinary → `+9.37` high-pressure, `+3.43` resilience, `59.5%` positive match rate).
- **#3 Namboori Tilak Varma (`SPRI = 1.752`)**: Carries the **heaviest knockout pressure volume (`111` knockout pressure balls)**, elevating his strike rate from `141.3 → 150.0` with only `3.36%` dismissals (`+3.28` resilience, `+0.362%` WPA/100).
- **#4 Dinesh Karthik (`SPRI = 1.633`) & #5 Nicholas Pooran (`1.572`)**: Elite death finishers (`+21.70` and `+22.42` raw `RAA/100` under pressure, `64.9%` match consistency for Karthik; Pooran produced the single highest-RVA clutch cameo of the Top 5 with `51` off `14` pressure balls for `+38.80` RVA / `+0.678` WPA vs RCB in IPL 2023).
- **#6 Mark Chapman (`1.505`) & #8 Towhid Hridoy (`1.196`)**: T20I middle-order stabilizers who slash their dismissal rates under pressure (`5.24% → 2.93%` for Chapman, `4.64% → 2.54%` for Hridoy).
- **#7 Heinrich Klaasen (`1.451`), #9 Tim David (`1.153`) & #10 Andre Russell (`1.127`)**: High-impact power anchors—David posts the highest `hi_WPA/100` (`+0.414%`) and Russell the highest shrunk Positive Match Rate (`70.0%`).

#### 5.2 Top 10 Bowlers Under Pressure

| Rank | Player | IPL / T20I Matches | Hi-P Balls (Matches) | Baseline RVA/100 | Hi-P RVA/100 (EB) | Baseline-Adj. Resilience | Hi-P WPA/100 | Pos. Match Rate | **SPRI** | 90% Bootstrap CI |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(bowl_md_rows)}

**Key Analytical Insights (Bowlers):**
- **#1 Axar Patel (`SPRI = 2.186`, 76.3% Bootstrap Top-10)**: The clearest statistical pressure riser among all bowlers (`726` pressure balls across `90` matches). From a slightly negative ordinary baseline (`-1.56` RVA/100), Axar surges to **`+5.93` RVA/100 (`+6.80` resilience)**, dropping his economy from `8.11 → 7.40` while jumping his wicket rate (`3.86% → 5.37%`) and leading the pool in `hi_WPA/100` (`+0.557%`).
- **#2 Matheesha Pathirana (`SPRI = 1.674`, 59.4% Bootstrap Top-10)**: Owns the **highest absolute high-pressure `RVA/100` (`+8.15`)** and highest match-to-match consistency (`65.2%`), striking every `11.3` balls under pressure (`8.81%` wicket rate!).
- **#3 Josh Hazlewood (`SPRI = 1.619`)**: Elite across both formats (`+15.6` IPL, `+47.0` T20I raw `hi_RVA/100`, `66` knockout pressure balls), improving both economy (`8.33 → 7.94`) and strike threat (`6.16% → 7.23%`), anchored by his match-winning `+1.011` WPA (`4/25` in `20` pressure balls) vs RR in IPL 2025.
- **#4 Mujeeb Ur Rahman (`1.508`) & #5 Shaheen Afridi (`1.493`)**: Mujeeb tightens to `6.75` RPO under pressure, while Shaheen thrives on big-stage T20I crunch moments (`96` knockout pressure balls, `4.73% → 7.76%` wicket rate).
- **#6 Jasprit Bumrah (`1.460`)**: Commands the highest ordinary baseline in cricket (`+10.68`) and maintains an elite **`+6.02` high-pressure RVA/100** (`+29.70` raw `RAA/100`!).
- **#7 Bhuvneshwar Kumar (`1.365`)**, **#8 Mitchell Santner (`1.353`)**, **#9 Jacob Duffy (`1.220`)**, **#10 Rashid Khan (`1.051`)**: High-volume workhorses (`Bhuvneshwar`: dataset-leading `968` high-pressure balls across `92` matches; `Rashid`: `756` balls across `86` matches) and disciplined T20I strike bowlers (`Santner` drops economy to `7.06`; `Duffy` hits `8.65%` wicket rate).

#### 5.3 Contrasting Archetype: Players Who Dip Under Pressure
Our baseline-adjusted resilience diagnostic also identifies high-skill players whose performance drops sharply when stakes spike:
- **Batters**: `Rishabh Pant` (`+6.62` ordinary → `+1.38` pressure, `res = -4.00`), `Aiden Markram` (`+4.85 → +1.75`, `res = -3.30`), `David Warner` (`+7.70 → +2.50`, `res = -3.08`), `Jos Buttler` (`+8.46 → +3.09`, `res = -2.62`).
- **Bowlers**: `Hardik Pandya` (`+3.64` ordinary → `-10.36` pressure, `res = -11.25`), `Tushar Deshpande` (`+3.72 → -4.12`, `res = -5.04`), `Azmatullah Omarzai` (`+3.93 → -3.92`, `res = -4.90`), `Ravi Bishnoi` (`+3.75 → -3.46`, `res = -4.39`).

---

### 6. Reliability, Validation & Limitations

1. **Sensitivity Across Definitions & Weights:**
   - **Batters**: Varying threshold (`tau = 0.65` → `rho = 0.966`, 9/10 overlap; `tau = 0.75` → `rho = 0.948`, 8/10 overlap), restricting to Top-Tier matches only (`rho = 0.997`, 9/10 overlap), Equal 33/33/33 weights (`rho = 0.999`, **10/10 overlap**), and Pure High-Pressure Level (`rho = 0.990`, 9/10 overlap) all confirm high stability.
   - **Bowlers**: Top-Tier only (`rho = 0.996`, 9/10 overlap), Equal weights (`rho = 0.998`, 9/10 overlap), High-Uplift 30/50/20 (`rho = 0.997`, 9/10 overlap), `tau = 0.65` (`rho = 0.974`, 8/10 overlap).
2. **1,000-Iteration Match-Clustered Bootstrap**: All Top-7 batters and Top-6 bowlers exhibit strictly positive 90% SPRI confidence intervals (`spri_ci_low > 0`), with top-10 retention probabilities up to `69.6%` (`Jaiswal`), `67.1%` (`Suryakumar`), and `76.3%` (`Axar Patel`).
3. **Out-of-Time Signal-to-Noise Limitation (`2022–2024` → `2025–2026`)**:
   - Splitting the data chronologically reveals low 2-year-to-2-year rank persistence in isolated high-pressure sub-samples (`Spearman r = 0.038` for batters, `r = 0.136` for bowlers, while 2022–24 ordinary bowling baseline predicts 2025–26 pressure RVA at `r = 0.211, p = 0.069`).
   - **Why**: Variance decomposition proves that even over 5 years, sampling variance within high-pressure balls (`mean SE^2 ≈ 278` batters / `248` bowlers) is **4–6× larger** than true between-player pressure talent variance (`tau^2 ≈ 51` batters / `38` bowlers). Over a 2-year half-sample, noise overwhelms signal—proving why **5-year pooling, strict eligibility floors, and Empirical Bayes shrinkage** are essential, and why marginal rankings (`#8–#15`) carry wider uncertainty intervals.
4. **Additional Model Limitations**:
   - **Fielding & Captaincy Confounding**: Dropped catches, misfields, and tactical field placements are unobserved in delivery logs and partially flow into bowler/batter residuals.
   - **Super Over Sample Scarcity**: Only `38` deliveries across `6` Super Overs exist in the dataset—too sparse for standalone estimation, though naturally absorbed at `P = 0.99`.
   - **Tactical Role Selection**: Specialist death bowlers (`Pathirana`, `Bumrah`, `Bhuvneshwar`) face deliberately higher-risk phases; while `xRuns`/`xWkt` controls for over and phase, residual role specialization (e.g., bowling exclusively to set hitters in overs 18–20) remains a subtle factor.

---

*Reproducible pipeline scripts (`src/01_eda_audit.py` through `src/11_build_pdf_report.py`) and all intermediate datasets/tables are stored in `d:\\Rajasthan Royals\\` and run cleanly inside `d:\\Rajasthan Royals\\venv`.*
"""

with open(OUT_MD, "w", encoding="utf-8") as f:
    f.write(md_text)
print("Saved Markdown report:", OUT_MD)

# ==============================================================================
# 2. BUILD STYLED PUBLICATION PDF REPORT (REPORTLAB)
# ==============================================================================
doc = SimpleDocTemplate(
    OUT_PDF, pagesize=A4,
    leftMargin=1.5*cm, rightMargin=1.5*cm, topMargin=1.5*cm, bottomMargin=1.5*cm,
    title="SupeRR Selector Season 2 - Round 1 Methodology Report"
)

styles = getSampleStyleSheet()
RR_PINK = colors.HexColor("#E5197A")
RR_BLUE = colors.HexColor("#004B8D")
RR_DARK = colors.HexColor("#1B2138")
LIGHT_BG = colors.HexColor("#F8FAFC")
ALT_ROW = colors.HexColor("#F1F5F9")

title_style = ParagraphStyle("RRTitle", parent=styles["Title"], fontSize=20, leading=24, textColor=RR_DARK, spaceAfter=4, alignment=TA_LEFT)
sub_style = ParagraphStyle("RRSub", parent=styles["Normal"], fontSize=11, leading=14, textColor=RR_PINK, fontName="Helvetica-Bold", spaceAfter=10)
h1_style = ParagraphStyle("RRH1", parent=styles["Heading1"], fontSize=13.5, leading=17, textColor=RR_BLUE, spaceBefore=12, spaceAfter=5, fontName="Helvetica-Bold")
h2_style = ParagraphStyle("RRH2", parent=styles["Heading2"], fontSize=11, leading=14, textColor=RR_DARK, spaceBefore=8, spaceAfter=3, fontName="Helvetica-Bold")
body_style = ParagraphStyle("RRBody", parent=styles["Normal"], fontSize=9.3, leading=13, textColor=colors.HexColor("#1E293B"), alignment=TA_JUSTIFY, spaceAfter=5)
bullet_style = ParagraphStyle("RRBullet", parent=body_style, leftIndent=12, firstLineIndent=-8, spaceAfter=3)
cap_style = ParagraphStyle("RRCap", parent=styles["Normal"], fontSize=8.2, leading=10.5, textColor=colors.HexColor("#475569"), alignment=TA_CENTER, spaceAfter=8)
tbl_hdr_style = ParagraphStyle("TblHdr", parent=styles["Normal"], fontSize=7.0, leading=8.8, textColor=colors.white, fontName="Helvetica-Bold", alignment=TA_CENTER)
tbl_cell_style = ParagraphStyle("TblCell", parent=styles["Normal"], fontSize=7.2, leading=9.0, textColor=RR_DARK, alignment=TA_CENTER)
tbl_left_style = ParagraphStyle("TblLeft", parent=tbl_cell_style, alignment=TA_LEFT, fontName="Helvetica-Bold")

story = []

# Header banner
story.append(Paragraph("RAJASTHAN ROYALS | SUPERR SELECTOR SEASON 2 — ROUND 1", sub_style))
story.append(Paragraph("Identifying the Most Reliable Players Under Pressure", title_style))
story.append(Paragraph("A Game-Theoretic Win Probability, Context-Adjusted Expected Value & Empirical Bayes Framework (2022–2026)", ParagraphStyle("Lead", parent=body_style, fontSize=10, textColor=colors.HexColor("#475569"))))
story.append(HRFlowable(width="100%", thickness=2, color=RR_PINK, spaceAfter=10))

# Executive Summary
story.append(Paragraph("Executive Summary & Final Round 1 Submissions", h1_style))
story.append(Paragraph(
    "We analyze <b>200,999 ball-by-ball deliveries across 867 played matches (141 competitions, Jan 2022–Nov 2026)</b> spanning IPL and T20I cricket to measure how batters and bowlers perform in high-pressure situations compared with their usual baseline performance. Using 5-fold match-grouped LightGBM Win Probability models (Innings 2 AUC = <b>0.8831</b>, ECE = <b>0.0070</b>), counterfactual ball-level leverage, asymmetric contextual stressors, and match-clustered Empirical Bayes shrinkage, we construct the <b>SupeRR Pressure Reliability Index (SPRI)</b> combining High-Pressure Value Added (45%), Baseline-Adjusted Resilience (30%), and Match-to-Match Consistency (25%).",
    body_style
))

# Quick side-by-side summary table of Top 10 Batters & Bowlers
sum_data = [[Paragraph("<b>Rank</b>", tbl_hdr_style), Paragraph("<b>Top 10 Batters Under Pressure</b>", tbl_hdr_style), Paragraph("<b>SPRI</b>", tbl_hdr_style),
             Paragraph("<b>Rank</b>", tbl_hdr_style), Paragraph("<b>Top 10 Bowlers Under Pressure</b>", tbl_hdr_style), Paragraph("<b>SPRI</b>", tbl_hdr_style)]]
for i in range(10):
    rb = bat_rank.iloc[i]
    rw = bowl_rank.iloc[i]
    sum_data.append([
        Paragraph(f"#{i+1}", tbl_cell_style),
        Paragraph(rb["player_name"], tbl_left_style),
        Paragraph(f"{rb['SPRI']:.3f}", tbl_cell_style),
        Paragraph(f"#{i+1}", tbl_cell_style),
        Paragraph(rw["player_name"], tbl_left_style),
        Paragraph(f"{rw['SPRI']:.3f}", tbl_cell_style),
    ])
sum_tbl = Table(sum_data, colWidths=[1.1*cm, 5.6*cm, 1.5*cm, 1.1*cm, 5.6*cm, 1.5*cm])
sum_tbl.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), RR_DARK),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, ALT_ROW]),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 3.5),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
]))
story.append(sum_tbl)
story.append(Spacer(1, 8))

# Section 1: Data Audit
story.append(Paragraph("1. Data Engineering, Audit & Domain Accounting", h1_style))
for b in [
    "<b>Runs & Extras Parity:</b> Verified exact 1-run match-result parity via <font face='Courier'>total_runs = runs + extras1_runs + extras_2_runs</font>. Identified <font face='Courier'>extras_type == 'other'</font> (264 rows) as non-overstep no-balls alongside <font face='Courier'>Overstep</font> (576 rows) and <font face='Courier'>wide</font> (7,714 rows). Reconstructed exact cumulative legal balls prior to every delivery (max = 120), preventing <font face='Courier'>innings_ball_num</font> extra-ball drift.",
    "<b>Batter vs. Bowler Credit:</b> <font face='Courier'>is_ball_faced</font> excludes wides (per Law 21). <font face='Courier'>bowler_runs</font> excludes byes, leg-byes, penalty runs, and no-ball bye extras. Batting dismissals are charged strictly when <font face='Courier'>dismissed_player_id == striker_id</font> (fixing 275 non-striker run-outs and 12 retired-hurt rows); bowler wickets exclude RunOut, Retire Out, and Obstructing the field.",
    "<b>DLS & Player Disambiguation:</b> Handled 25 DLS/revised-overs matches (<font face='Courier'>first/second_inn_revised_overs</font> are stored in allocated legal balls, e.g., 90, 66, 30) and 38 Super Over deliveries. Grouped strictly by <font face='Courier'>player_id</font> after discovering 7 shared names across distinct players (e.g., Rashid Khan 4722 vs 20172 Nepal).",
    "<b>Era Inflation & Opposition Quality:</b> IPL scoring inflated +15.7% from 8.54 RPO (2022) to 9.88 RPO (2026), while 54 T20I matches involve Associate nations. Built chronological pre-match <b>Dynamic Team Elo</b> ratings and strictly causal expanding venue/format par RPOs."
]:
    story.append(Paragraph(f"• {b}", bullet_style))

# Section 2: Pressure Definition
story.append(Paragraph("2. Defining Pressure: Win Probability, Counterfactual Leverage & Asymmetric Stress", h1_style))
story.append(Paragraph(
    "Heuristic cutoffs (e.g., 'overs 16–20' or 'RRR > 10') fail in one-sided chases, 1st-innings collapses, and low-scoring defenses. We define pressure directly from match-outcome cruciality and domain stressors:",
    body_style
))
for b in [
    "<b>Calibrated Win Probability (WP):</b> 5-fold GroupKFold (by <font face='Courier'>match_id</font>) LightGBM models with monotonicity constraints (<font face='Courier'>runs_before +1, wickets_before -1, runs_needed -1, rrr_before -1, elo_diff +1</font>) and nested Isotonic Calibration. OOF performance: <b>Innings 1</b> LogLoss=0.6338, AUC=0.7064, ECE=0.0303; <b>Innings 2</b> LogLoss=0.4324, AUC=0.8831, ECE=0.0070.",
    "<b>Counterfactual Leverage Index (LI):</b> Evaluates how much a dot, single, four, six, or wicket changes WP at state <i>s</i>, yielding expected absolute WP swing normalized by the median delivery.",
    "<b>Asymmetric Pressure Indices (P<sub>bat</sub>, P<sub>bowl</sub> in [0,1]):</b> Combines Leverage (38–40%), Match Closeness 4·WP·(1-WP) gated against dead-rubber garbage time (22–24%), and role-specific stressors: <b>Batting</b> combines Chase Acceleration/Par Deficit (24%) + Collapse/Wicket-Fragility (16%); <b>Bowling</b> incorporates Defensive Containment & Execution Stress (36%), plus a +0.10 Knockout/Playoff Stakes boost. Percentile-normalized across all 200,961 regular deliveries; High Pressure defined as <b>P ≥ 0.70</b> (weighted by P²) vs Ordinary Baseline <b>P &lt; 0.55</b>."
]:
    story.append(Paragraph(f"• {b}", bullet_style))

story.append(Image(os.path.join(FIG_DIR, "fig1_pressure_validation.png"), width=17.6*cm, height=6.8*cm))
story.append(Paragraph("<i>Figure 1: Empirical validation across Pressure Deciles D1–D10. Actual match swing |ΔWP| increases 19.7× (batting) and 24.5× (bowling), while batting RPO drops, dismissal risk peaks at 6.15%, and bowler wide/no-ball errors jump +35% to 5.14% in D10.</i>", cap_style))

story.append(PageBreak())

# Section 3: Performance Measurement & Baseline Comparison
story.append(Paragraph("3. Context-Adjusted Performance Measurement & Empirical Bayes Rating Logic", h1_style))
story.append(Paragraph(
    "Raw strike rates and economy rates are severely confounded by venue, era (2022 vs 2026), phase, and opposition tier. We fit 5-fold match-grouped LightGBM baseline models (<font face='Courier'>xRuns</font>, <font face='Courier'>xWkt</font>) controlling for competition group, year, causal venue par RPO, Impact Player rule, pre-match team/opponent Elo, innings, over, balls remaining, wickets lost, batting order, batter balls faced, and pace/spin/hand matchups. Residuals show near-zero bias across all phases (Death +0.010, Middle -0.001, Powerplay +0.004) and formats (IPL +0.004, T20I +0.002).",
    body_style
))
story.append(Paragraph(
    "To value wickets dynamically, we fit a smooth empirical DLS resource curve on Innings 1 remaining runs (<i>c</i> = 1.0041, <i>α</i> = 0.6952, <i>β</i> = 1.5686), yielding state-dependent wicket run values <i>V<sub>W</sub></i> ranging smoothly from <b>4.4–12.9 runs</b> in Powerplay/Middle to <b>2.0–5.0 runs</b> at the Death. Each delivery yields <b>Net Run Value Added (RVA/100)</b> and <b>Expected-Centered Win Probability Added (WPA_adj/100)</b>.",
    body_style
))

story.append(Image(os.path.join(FIG_DIR, "fig2_wp_calibration_and_vw.png"), width=17.6*cm, height=6.8*cm))
story.append(Paragraph("<i>Figure 2: Out-of-fold Win Probability reliability diagrams (left) and smooth empirical DLS marginal wicket run values (right).</i>", cap_style))

story.append(Paragraph("Solving the 'Elite Baseline Trap' via Regression-Adjusted Resilience", h2_style))
story.append(Paragraph(
    "Because sampling variance in high-pressure balls is large and elite baselines regress toward the mean (Corr(lo, hi) = 0.231 batters, 0.181 bowlers), raw subtraction (<font face='Courier'>hi - lo</font>) is dominated by noise (mean SE²<sub>diff</sub> = 447 vs total variance 442) and unfairly penalizes superstars like Bumrah (+10.68 baseline) or Klaasen (+10.99 baseline). After aggregating to match level with <b>Liang-Zeger cluster-robust SEs</b> and applying <b>Empirical Bayes shrinkage</b>, we measure <b>Baseline-Adjusted Resilience</b> as the residual above the Weighted Least Squares expectation line: <i>Resilience(p) = μ<sub>hi</sub><sup>EB</sup> − (α + β · μ<sub>lo</sub><sup>EB</sup>)</i> (slopes 0.182 batters, 0.337 bowlers). Our composite <b>SPRI</b> blends High-Pressure Level (45%), Baseline-Adjusted Resilience (30%), and Match Consistency via Beta-Binomial Positive Match Rate + 1-SE Lower Confidence Bound (25%).",
    body_style
))

story.append(Image(os.path.join(FIG_DIR, "fig3_baseline_vs_pressure_scatter.png"), width=17.6*cm, height=7.7*cm))
story.append(Paragraph("<i>Figure 3: Ordinary baseline vs. high-pressure RVA/100 (EB shrunk). Top-10 selections sit well above the blue WLS expectation line, contrasting sharply with notable pressure-dippers below.</i>", cap_style))

story.append(PageBreak())

# Section 4: Eligibility & Top 10 Evidence
story.append(Paragraph("4. Fair Comparisons, Eligibility & Top-10 Evidence", h1_style))
story.append(Paragraph(
    "To guarantee fair cross-format comparison and eliminate small-sample or Associate-only stat-padding, players must record <b>≥ 500 Top-Tier deliveries (IPL + Full-Member T20Is)</b>, <b>≥ 180 High-Pressure deliveries</b> across <b>≥ 20 High-Pressure matches</b>, and <b>≥ 150 Ordinary Baseline deliveries</b>—producing balanced pools of <b>N = 102 Batters</b> and <b>N = 102 Bowlers</b>.",
    body_style
))

# Detailed Batters Table
story.append(Paragraph("Table 1: Final Top 10 Batters Under Pressure — Full Quantitative Evidence", h2_style))
b_hdr = [Paragraph(h, tbl_hdr_style) for h in ["Rk", "Batter", "IPL/T20I\nMatches", "Hi-P Balls\n(Matches)", "Base\nRVA/100", "Hi-P\nRVA/100", "Resil.\nRVA/100", "Hi-P\nWPA/100", "Pos.\nMatch%", "SPRI", "90% Boot\nCI"]]
b_rows = [b_hdr]
for _, r in bat_rank.head(10).iterrows():
    b_rows.append([
        Paragraph(f"{int(r['rank'])}", tbl_cell_style),
        Paragraph(r["player_name"].replace("Namboori ", ""), tbl_left_style),
        Paragraph(f"{int(r['ipl_matches'])}/{int(r['t20i_matches'])}", tbl_cell_style),
        Paragraph(f"{int(r['hi_balls'])} ({int(r['hi_matches'])})", tbl_cell_style),
        Paragraph(f"{r['lo_rva100_eb']:+.2f}", tbl_cell_style),
        Paragraph(f"{r['hi_rva100_eb']:+.2f}", tbl_cell_style),
        Paragraph(f"{r['res_rva100']:+.2f}", tbl_cell_style),
        Paragraph(f"{r['hi_wpa100_eb']:+.3f}%", tbl_cell_style),
        Paragraph(f"{r['pmr_eb']*100:.1f}%", tbl_cell_style),
        Paragraph(f"<b>{r['SPRI']:.3f}</b>", tbl_cell_style),
        Paragraph(f"[{r['spri_ci_low']:.2f}, {r['spri_ci_high']:.2f}]", tbl_cell_style),
    ])
t_bat = Table(b_rows, colWidths=[0.85*cm, 2.85*cm, 1.3*cm, 1.85*cm, 1.35*cm, 1.35*cm, 1.35*cm, 1.5*cm, 1.25*cm, 1.1*cm, 2.25*cm])
t_bat.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), RR_BLUE),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, ALT_ROW]),
    ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 3),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
]))
story.append(t_bat)
story.append(Spacer(1, 8))

# Detailed Bowlers Table
story.append(Paragraph("Table 2: Final Top 10 Bowlers Under Pressure — Full Quantitative Evidence", h2_style))
w_hdr = [Paragraph(h, tbl_hdr_style) for h in ["Rk", "Bowler", "IPL/T20I\nMatches", "Hi-P Balls\n(Matches)", "Base\nRVA/100", "Hi-P\nRVA/100", "Resil.\nRVA/100", "Hi-P\nWPA/100", "Pos.\nMatch%", "SPRI", "90% Boot\nCI"]]
w_rows = [w_hdr]
for _, r in bowl_rank.head(10).iterrows():
    w_rows.append([
        Paragraph(f"{int(r['rank'])}", tbl_cell_style),
        Paragraph(r["player_name"], tbl_left_style),
        Paragraph(f"{int(r['ipl_matches'])}/{int(r['t20i_matches'])}", tbl_cell_style),
        Paragraph(f"{int(r['hi_balls'])} ({int(r['hi_matches'])})", tbl_cell_style),
        Paragraph(f"{r['lo_rva100_eb']:+.2f}", tbl_cell_style),
        Paragraph(f"{r['hi_rva100_eb']:+.2f}", tbl_cell_style),
        Paragraph(f"{r['res_rva100']:+.2f}", tbl_cell_style),
        Paragraph(f"{r['hi_wpa100_eb']:+.3f}%", tbl_cell_style),
        Paragraph(f"{r['pmr_eb']*100:.1f}%", tbl_cell_style),
        Paragraph(f"<b>{r['SPRI']:.3f}</b>", tbl_cell_style),
        Paragraph(f"[{r['spri_ci_low']:.2f}, {r['spri_ci_high']:.2f}]", tbl_cell_style),
    ])
t_bowl = Table(w_rows, colWidths=[0.85*cm, 2.85*cm, 1.3*cm, 1.85*cm, 1.35*cm, 1.35*cm, 1.35*cm, 1.5*cm, 1.25*cm, 1.1*cm, 2.25*cm])
t_bowl.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), RR_PINK),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, ALT_ROW]),
    ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 3),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
]))
story.append(t_bowl)
story.append(Spacer(1, 8))

story.append(Image(os.path.join(FIG_DIR, "fig4_top10_profiles_and_ci.png"), width=16.8*cm, height=6.4*cm))
story.append(Paragraph("<i>Figure 4: SPRI pillar breakdown and 90% match-clustered bootstrap confidence intervals (1,000 resamples) for Top 10 Batters and Bowlers.</i>", cap_style))

# Section 5: Cross-Format & Case Studies
story.append(Paragraph("5. Cross-Format Robustness & Signature Pressure Moments", h1_style))
story.append(Image(os.path.join(FIG_DIR, "fig5_format_and_consistency.png"), width=16.8*cm, height=5.9*cm))
story.append(Paragraph("<i>Figure 5: High-pressure performance consistency across IPL, T20Is, and match-level positive-impact rates (EB shrunk).</i>", cap_style))

for b in [
    "<b>Batting Standouts:</b> <b>#1 Yashasvi Jaiswal</b> boosts his strike rate under pressure (156.3 → 161.7) while lowering his dismissal rate (4.25% → 3.72%), highlighted by +27.2 RVA (57 off 30 pressure balls) vs SRH (IPL 2024). <b>#2 Suryakumar Yadav</b> leads all Top-10 batters in sample size (819 pressure balls, 103 matches) with +9.37 RVA/100. <b>#3 Tilak Varma</b> carries 111 knockout pressure balls (+28.8 RVA in Qualifier 2 vs GT, IPL 2023). <b>#5 Nicholas Pooran</b> delivered the single highest-RVA cameo (+38.80 RVA, +0.678 WPA, 51 off 14 pressure balls vs RCB, IPL 2023).",
    "<b>Bowling Standouts:</b> <b>#1 Axar Patel</b> exhibits the sharpest pressure elevation (+7.50 raw uplift, +6.80 resilience, economy 8.11 → 7.40, wicket rate 3.86% → 5.37% across 90 pressure matches). <b>#2 Matheesha Pathirana</b> posts the highest absolute pressure RVA (+8.15/100, 8.81% wicket rate, 65.2% match consistency). <b>#3 Josh Hazlewood</b> dominates both IPL (+24.4 RVA, +1.011 WPA, 4/25 vs RR in IPL 2025) and T20Is (47.0 raw RVA/100), while <b>#6 Jasprit Bumrah</b> (+29.7 raw RAA/100) and <b>#7 Bhuvneshwar Kumar</b> (968 pressure balls) provide elite volume stability."
]:
    story.append(Paragraph(f"• {b}", bullet_style))

# Section 6: Reliability, Sensitivity & Limitations
story.append(Paragraph("6. Reliability, Sensitivity Checks & Honest Limitations", h1_style))

# Sensitivity summary table
s_hdr = [Paragraph(h, tbl_hdr_style) for h in ["Model / Sensitivity Variant", "Batters Spearman ρ", "Batters Top-10 Overlap", "Bowlers Spearman ρ", "Bowlers Top-10 Overlap"]]
s_rows = [s_hdr]
for i in range(len(sens_bat)):
    rb = sens_bat.iloc[i]
    rw = sens_bowl.iloc[i]
    s_rows.append([
        Paragraph(rb["variant"], tbl_left_style),
        Paragraph(f"{rb['spearman_rho']:.4f}", tbl_cell_style),
        Paragraph(f"{int(rb['top10_overlap'])} / 10", tbl_cell_style),
        Paragraph(f"{rw['spearman_rho']:.4f}", tbl_cell_style),
        Paragraph(f"{int(rw['top10_overlap'])} / 10", tbl_cell_style),
    ])
t_sens = Table(s_rows, colWidths=[6.4*cm, 2.5*cm, 2.5*cm, 2.5*cm, 2.5*cm])
t_sens.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), RR_DARK),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, ALT_ROW]),
    ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 3),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
]))
story.append(t_sens)
story.append(Spacer(1, 6))

for b in [
    "<b>Sensitivity & Bootstrap Robustness:</b> Rankings are highly robust across pressure thresholds (τ = 0.65, 0.75), Top-Tier-only filtering (ρ = 0.997 batters, 0.996 bowlers, 9/10 overlap), and pillar weights (Equal 33/33/33 yields 10/10 batter & 9/10 bowler overlap). In 1,000 match-clustered bootstraps, all Top-7 batters and Top-6 bowlers hold strictly positive 90% SPRI confidence intervals.",
    "<b>Signal-to-Noise Limitation (Chronological 2022–24 → 2025–26 Split):</b> Out-of-time persistence of isolated high-pressure skill across 2-year halves is modest (Spearman r = 0.038 batters, 0.136 bowlers; while 2022–24 ordinary bowling predicts 2025–26 pressure RVA at r = 0.211, p = 0.069). Variance decomposition explains why: within high-pressure balls, sampling variance (mean SE² ≈ 278 batters / 248 bowlers) is 4–6× larger than true between-player pressure variance (τ² ≈ 51 / 38). Pooling all 5 seasons with Empirical Bayes shrinkage is therefore essential, and rank ordering among #7–#12 retains genuine statistical uncertainty.",
    "<b>Unobserved Context & Role Selection:</b> Dropped catches, fielding quality, and tactical captaincy are unobserved in delivery logs. Death specialists (Pathirana, Bumrah, Bhuvneshwar) systematically bowl to set hitters in overs 18–20, which phase-level controls only partially capture. Super Overs (38 balls) are too sparse for separate inference."
]:
    story.append(Paragraph(f"• {b}", bullet_style))

doc.build(story)
print("Saved styled PDF report:", OUT_PDF)
