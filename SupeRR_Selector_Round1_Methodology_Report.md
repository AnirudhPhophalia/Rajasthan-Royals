# SupeRR Selector Season 2 — Round 1 Methodology Report
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
$$\text{Swing}(s) = \sum_{k \in \{0,1,4,6,W\}} p_k \cdot |WP_k(s) - WP(s)|, \qquad \text{LI}(s) = \frac{\text{Swing}(s)}{\text{median}(\text{Swing})}$$

#### 2.3 Asymmetric Batting & Bowling Pressure Indices ($P_{\text{bat}}, P_{\text{bowl}} \in [0, 1]$)
Using Leverage Index ($\text{LI}$), Match Closeness $U(s) = 4 \cdot WP(s) \cdot (1 - WP(s))$, and a smooth non-garbage-time gate $g(s) = \min(WP, 1-WP)/0.06$ (clipped to $[0,1]^{0.6}$, suppressing dead rubbers where $WP < 0.02$ or $> 0.98$), we construct asymmetric contextual stress composites and percentile-rank normalize them onto $[0, 1]$ across all `200,961` regular deliveries (Super Overs `innings >= 3` assigned `0.99`):

- **Batting Pressure ($P_{\text{bat}}$)**:
  - `38%` Normalized Counterfactual Leverage ($C_{\text{lev}}$)
  - `22%` Match Closeness $\times$ Live Gate
  - `24%` Chase Acceleration / Innings-1 Par-Deficit Stress ($C_{\text{chase}}$: `rrr_before / prior_venue_rpo` scaled by wicket scarcity $\sqrt{5 / \text{wkts\_in\_hand}}$ and innings progress in Inn 2; deficit vs venue par + live death ramp in Inn 1)
  - `16%` Collapse / Wicket-Fragility Stress ($C_{\text{collapse}}$: excess wickets lost above `legal_balls / 19`, early top-order collapses $\ge 2$ down in Powerplay, and 12-ball wicket clusters)
  - `+0.10` Knockout/Playoff Stage Stakes Boost (`Final`, `Qualifier`, `Eliminator`, `Semi Final`, `Super 4`, `knock_out == 1`)

- **Bowling Pressure ($P_{\text{bowl}}$)**:
  - `40%` Normalized Counterfactual Leverage ($C_{\text{lev}}$)
  - `24%` Match Closeness $\times$ Live Gate
  - `36%` Defensive Containment & Execution Stress ($C_{\text{bowl\_exec}}$: live chase containment threat in Inn 2 when batting team has resources/closeness; death-overs containment and above-par Powerplay threats in Inn 1)
  - `+0.10` Knockout/Playoff Stage Stakes Boost

#### 2.4 Empirical Validation: Separating Pressure from Ordinary Play
Across pressure deciles (`D1` lowest 10% to `D10` highest 10%), our Pressure Index produces sharp, monotonically verified separation:
- **Match-Outcome Leverage (`|ΔWP|` per ball)**: Scales **19.7×** for Batting (`0.225%` in D1 → `4.445%` in D10) and **24.5×** for Bowling (`0.191%` in D1 → `4.675%` in D10).
- **Behavioral Stress Signatures**:
  - Under **Batting Pressure**, scoring rate is squeezed from **9.21–9.45 RPO (D1–D3)** down to **8.43–8.54 RPO (D9–D10)** while dismissal rate peaks at **6.15%** in D10.
  - Under **Bowling Pressure**, illegal-delivery discipline errors (`wide + no-ball` rate) spike from **3.63%–3.95% (D2–D4)** to **5.14%** in D10 (+35% error surge), while conceded RPO climbs to **8.99** and wicket rate peaks at **5.88%**.
- **High-Stakes Tournaments**: Knockout/playoff deliveries average `P_bat = 0.674` and `P_bowl = 0.635` vs `0.492` / `0.494` in league games.

We define **High Pressure** as $P \ge 0.70$ (top 30% of deliveries, weighted continuously by $P^2$ so `D9/D10` crunch moments weigh more than borderline `D7` balls) and **Ordinary Baseline** as $P < 0.55$.

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
$$\widehat{\text{RemRuns}}(b, w) = c \cdot \text{Par} \cdot \left(\frac{b}{B_{\max}}\right)^\alpha \cdot \left(1 - \left(1 - \frac{w}{10}\right)^\beta\right)$$
yielding fitted parameters $c = 1.0041, \alpha = 0.6952, \beta = 1.5686$. The marginal value of a wicket $V_W(b, w) = \text{clip}(\widehat{\text{RemRuns}}(b, w) - \widehat{\text{RemRuns}}(b, w-1), 2.0, 13.5)$ (`1.5` on final ball) smoothly values a Powerplay/Middle wicket at **4.4–12.9 runs** and a late Death wicket at **2.0–5.0 runs**.

Every delivery receives two complementary performance currencies:
1. **Context-Adjusted Net Run Value Added (`RVA`, scaled per 100 balls)**:
   - Batting: $\text{RVA}_{\text{bat}} = (\text{runs} - \widehat{xR}_{\text{bat}}) + V_W \cdot (\widehat{xW}_{\text{bat}} - \mathbb{I}[\text{striker\_out}])$
   - Bowling: $\text{RVA}_{\text{bowl}} = (\widehat{xR}_{\text{bowl}} - \text{bowler\_runs}) + V_W \cdot (\mathbb{I}[\text{bowler\_wkt}] - \widehat{xW}_{\text{bowl}})$
2. **Expected-Centered Win Probability Added (`WPA_adj`, scaled per 100 balls)**: Actual $\Delta WP$ minus the league mean $\Delta WP$ in that pressure vigintile, ensuring players fighting in desperate chases aren't penalized for baseline negative drift.

#### 3.3 Match-Clustered Empirical Bayes Shrinkage & Baseline-Adjusted Resilience
Deliveries within the same innings are correlated. We aggregate to the `(player_id, match_id)` level and compute **Liang-Zeger cluster-robust Standard Errors** ($	ext{SE}_{\text{cluster}}$) for both High-Pressure (`P >= 0.70`, weighted by $P^2$) and Ordinary Baseline (`P < 0.55`) regimes, then apply **Empirical Bayes (James-Stein) shrinkage** toward the eligible pool inverse-variance mean using DerSimonian-Laird between-player variance $\tau^2$.

**Crucial Methodological Discovery — Solving the "Elite Baseline Trap":**
In raw data, subtracting ordinary performance from high-pressure performance (`hi - lo`) adds two noisy estimates (`mean SE^2_diff = 447.4` vs `var = 441.5`) and penalizes elite normal-overs superstars (e.g., Jasprit Bumrah `+10.68`, Suryakumar Yadav `+9.70`, Heinrich Klaasen `+10.99`) due to statistical regression to the mean (`Corr(lo, hi) = 0.231` batters, `0.181` bowlers). Therefore, we measure **Baseline-Adjusted Pressure Resilience** via Weighted Least Squares regression of $\hat{\theta}_{\text{hi}}^{\text{EB}}$ on $\hat{\theta}_{\text{lo}}^{\text{EB}}$:
$$\text{Resilience}(p) = \hat{\theta}_{\text{hi}}^{\text{EB}}(p) - \left(\hat{\alpha} + \hat{\beta} \cdot \hat{\theta}_{\text{lo}}^{\text{EB}}(p)\right)$$
where fitted slopes are $\hat{\beta} = 0.182$ (Batters) and $\hat{\beta} = 0.337$ (Bowlers). This rewards *both* players who elevate dramatically from modest baselines (`Yashasvi Jaiswal`, `Axar Patel`) *and* elite superstars who sustain elite delivery under pressure (`Suryakumar Yadav`, `Matheesha Pathirana`, `Jasprit Bumrah`).

#### 3.4 Final Composite Score: SupeRR Pressure Reliability Index (SPRI)
Across the eligible pool ($N = 102$ batters, $N = 102$ bowlers), each pillar is z-scored and combined:
$$\text{SPRI} = 0.45 \cdot Z_{\text{Level}} + 0.30 \cdot Z_{\text{Resilience}} + 0.25 \cdot Z_{\text{Consistency}}$$
- $Z_{\text{Level}}$: `65%` High-Pressure `RVA_100 (EB)` + `35%` High-Pressure `WPA_100 (EB)`
- $Z_{\text{Resilience}}$: `65%` Baseline-Adjusted `RVA` Resilience + `35%` Baseline-Adjusted `WPA` Resilience
- $Z_{\text{Consistency}}$: `50%` Beta-Binomial Positive-Impact Match Rate (`(pos_matches + 6)/(sig_matches + 12)` on matches with $\ge 3$ pressure balls) + `50%` 1-SE Lower Confidence Bound (`hi_RVA_EB - 1.0 * post_SE`)

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
| 1 | **Yashasvi Jaiswal** | 69 / 20 | 457 (78m) | +2.22 | +9.64 | +5.06 | +0.154% | 60.6% | **2.167** | [0.31, 2.79] |
| 2 | **Suryakumar Yadav** | 64 / 75 | 819 (103m) | +9.70 | +9.37 | +3.43 | +0.274% | 59.5% | **1.985** | [0.30, 2.71] |
| 3 | **Namboori Tilak Varma** | 65 / 58 | 684 (80m) | +4.80 | +8.32 | +3.28 | +0.362% | 56.8% | **1.752** | [0.14, 2.36] |
| 4 | **Dinesh Karthik** | 42 / 19 | 267 (40m) | +6.02 | +7.78 | +2.51 | +0.354% | 64.9% | **1.633** | [0.34, 2.29] |
| 5 | **Nicholas Pooran** | 70 / 45 | 627 (78m) | +6.18 | +8.04 | +2.75 | +0.319% | 58.0% | **1.572** | [0.22, 2.04] |
| 6 | **Mark Chapman** | 0 / 53 | 239 (36m) | +4.54 | +7.55 | +2.55 | +0.368% | 59.5% | **1.505** | [0.14, 2.63] |
| 7 | **Heinrich Klaasen** | 54 / 18 | 401 (48m) | +10.99 | +7.70 | +1.53 | +0.301% | 65.3% | **1.451** | [0.12, 2.17] |
| 8 | **Towhid Hridoy** | 0 / 50 | 276 (32m) | +2.53 | +6.88 | +2.25 | +0.267% | 58.5% | **1.196** | [-0.10, 1.93] |
| 9 | **Tim David** | 58 / 40 | 455 (59m) | +9.75 | +6.80 | +0.85 | +0.414% | 59.3% | **1.153** | [0.04, 1.70] |
| 10 | **Andre Russell** | 45 / 12 | 284 (41m) | +5.18 | +6.92 | +1.80 | +0.009% | 70.0% | **1.127** | [0.01, 2.08] |

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
| 1 | **Axar Patel** | 65 / 71 | 726 (90m) | -1.56 | +5.93 | +6.80 | +0.557% | 54.1% | **2.186** | [0.63, 2.57] |
| 2 | **Matheesha Pathirana** | 33 / 21 | 420 (39m) | +9.28 | +8.15 | +5.36 | -0.016% | 65.2% | **1.674** | [0.27, 2.51] |
| 3 | **Josh Hazlewood** | 40 / 25 | 595 (59m) | +5.06 | +6.13 | +4.76 | +0.236% | 60.0% | **1.619** | [0.27, 2.28] |
| 4 | **Mujeeb Ur Rahman** | 1 / 37 | 223 (26m) | +4.47 | +5.08 | +3.91 | +0.279% | 60.6% | **1.508** | [0.15, 2.19] |
| 5 | **Shaheen Afridi** | 0 / 47 | 438 (42m) | +1.83 | +4.85 | +4.57 | +0.377% | 52.1% | **1.493** | [0.16, 2.07] |
| 6 | **Jasprit Bumrah** | 52 / 23 | 580 (61m) | +10.68 | +6.02 | +2.76 | +0.275% | 56.1% | **1.460** | [0.25, 2.06] |
| 7 | **Bhuvneshwar Kumar** | 74 / 27 | 968 (92m) | +6.17 | +4.35 | +2.61 | +0.312% | 60.0% | **1.365** | [-0.05, 2.05] |
| 8 | **Mitchell Santner** | 29 / 54 | 392 (55m) | +2.97 | +4.66 | +4.00 | +0.280% | 56.6% | **1.353** | [0.17, 1.82] |
| 9 | **Jacob Duffy** | 6 / 36 | 266 (33m) | +4.01 | +4.96 | +3.94 | +0.235% | 52.5% | **1.220** | [0.20, 1.75] |
| 10 | **Rashid Khan** | 77 / 45 | 756 (86m) | +7.87 | +3.83 | +1.52 | +0.330% | 52.4% | **1.051** | [-0.14, 1.67] |

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

*Reproducible pipeline scripts (`src/01_eda_audit.py` through `src/11_build_pdf_report.py`) and all intermediate datasets/tables are stored in `d:\Rajasthan Royals\` and run cleanly inside `d:\Rajasthan Royals\venv`.*
