import os
import json
import numpy as np
import pandas as pd

DATA_DIR = r"d:\Rajasthan Royals\Data Pack"
OUT_DIR = r"d:\Rajasthan Royals\outputs\processed"
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading raw CSVs...")
comp = pd.read_csv(os.path.join(DATA_DIR, "competitions.csv"))
matches = pd.read_csv(os.path.join(DATA_DIR, "matches.csv"))
bbb = pd.read_csv(os.path.join(DATA_DIR, "ball_by_ball.csv"), low_memory=False)
pe = pd.read_csv(os.path.join(DATA_DIR, "playing_eleven.csv"))

# 1. Clean matches + competitions
m = matches.merge(comp, on="comp_id", how="left")
m["match_date"] = pd.to_datetime(m["match_date"])
m["year"] = m["match_date"].dt.year

def classify_comp(row):
    name = str(row["comp_name"])
    if "IPL" in name:
        return "IPL"
    elif row["match_format"] == "T20":
        return "Emerging_T20"
    else:
        return "T20I"

m["comp_group"] = m.apply(classify_comp, axis=1)

full_members = {
    "India", "Australia", "England", "South Africa", "New Zealand",
    "Pakistan", "West Indies", "Sri Lanka", "Bangladesh", "Afghanistan",
    "Ireland", "Zimbabwe"
}
m["home_is_fm"] = m["home_team"].isin(full_members) | (m["comp_group"] == "IPL")
m["away_is_fm"] = m["away_team"].isin(full_members) | (m["comp_group"] == "IPL")
m["is_top_tier_match"] = (
    (m["comp_group"] == "IPL") |
    ((m["comp_group"] == "T20I") & m["home_is_fm"] & m["away_is_fm"])
)

knockout_stages = {"Final", "Qualifier", "Eliminator", "Semi Final", "Super 4"}
m["is_high_stakes_stage"] = (m["knock_out"] == 1) | (m["stage"].isin(knockout_stages))

# Sort chronologically for Elo & expanding venue stats
m = m.sort_values(["match_date", "match_id"]).reset_index(drop=True)

# 2. Compute Dynamic Pre-Match Team Elo Ratings
# Separate Elo pools for IPL franchises vs National teams, or unified with initial seeds
elo = {}
def get_init_elo(team_name, comp_group):
    if comp_group == "IPL":
        return 1550.0
    elif team_name in full_members:
        if team_name in {"India", "Australia", "England", "South Africa", "New Zealand"}:
            return 1560.0
        elif team_name in {"Pakistan", "West Indies", "Sri Lanka", "Bangladesh", "Afghanistan"}:
            return 1500.0
        else:
            return 1420.0
    elif comp_group == "Emerging_T20":
        return 1360.0
    else:
        return 1300.0

home_elos = []
away_elos = []
for _, row in m.iterrows():
    ht = row["home_team_id"]
    at = row["away_team_id"]
    cg = row["comp_group"]
    if ht not in elo:
        elo[ht] = get_init_elo(row["home_team"], cg)
    if at not in elo:
        elo[at] = get_init_elo(row["away_team"], cg)

    eh, ea = elo[ht], elo[at]
    home_elos.append(eh)
    away_elos.append(ea)

    wt = row["win_team_id"]
    if pd.notna(wt) and wt in (ht, at):
        exp_h = 1.0 / (1.0 + 10.0 ** ((ea - eh) / 400.0))
        actual_h = 1.0 if wt == ht else 0.0
        k = 24.0 if cg in ("IPL", "T20I") else 16.0
        elo[ht] = eh + k * (actual_h - exp_h)
        elo[at] = ea + k * ((1.0 - actual_h) - (1.0 - exp_h))

m["home_elo_pre"] = home_elos
m["away_elo_pre"] = away_elos

# 3. Delivery-level cleaning and canonical accounting
bbb = bbb.sort_values(["match_id", "innings", "overid", "over_ball_num", "innings_ball_num"]).reset_index(drop=True)

bbb["is_wide"] = bbb["extras_type"] == "wide"
bbb["is_noball"] = bbb["extras_type"].isin(["Overstep", "other"])
bbb["is_legal"] = ~(bbb["is_wide"] | bbb["is_noball"])
bbb["is_ball_faced"] = ~bbb["is_wide"]
bbb["total_runs"] = bbb["runs"] + bbb["extras1_runs"] + bbb["extras_2_runs"]

# Bowler runs: runs + extras1_runs on wides/noballs (extras_2_runs on noballs are byes/legbyes)
bbb["bowler_runs"] = np.where(
    bbb["is_wide"] | bbb["is_noball"],
    bbb["runs"] + bbb["extras1_runs"],
    bbb["runs"]
)
non_bowler_dismissals = {"RunOut", "Retire Out", "Obstracting the field", "Obstructing the field"}
bbb["is_bowler_wicket"] = bbb["is_wicket"] & (~bbb["dismissal_type"].isin(non_bowler_dismissals))

# Canonical player name mapping by player_id (most frequent name per ID)
s_names = bbb[["striker_id", "striker_name"]].rename(columns={"striker_id": "player_id", "striker_name": "player_name"})
b_names = bbb[["bowler_id", "bowler_name"]].rename(columns={"bowler_id": "player_id", "bowler_name": "player_name"})
ns_names = bbb[["non_striker_id", "non_striker_name"]].rename(columns={"non_striker_id": "player_id", "non_striker_name": "player_name"})
all_names = pd.concat([s_names, b_names, ns_names], ignore_index=True)
id_to_name = all_names.groupby("player_id")["player_name"].agg(lambda s: s.value_counts().index[0]).to_dict()

# Merge match info
m_cols = [
    "match_id", "comp_id", "comp_name", "match_format", "season", "year", "comp_group",
    "match_date", "stage", "venue_id", "venue_country", "home_team_id", "away_team_id",
    "home_team", "away_team", "toss_win_team_id", "toss_opted",
    "first_inn_revised_overs", "second_inn_revised_overs", "revised_target",
    "knock_out", "is_high_stakes_stage", "win_team_id", "result",
    "is_top_tier_match", "home_elo_pre", "away_elo_pre"
]
df = bbb.merge(m[m_cols], on="match_id", how="left")

# Batting & Bowling team Elo
df["bat_team_elo"] = np.where(df["batting_team_id"] == df["home_team_id"], df["home_elo_pre"], df["away_elo_pre"])
df["bowl_team_elo"] = np.where(df["bowling_team_id"] == df["home_team_id"], df["home_elo_pre"], df["away_elo_pre"])
df["elo_diff"] = df["bat_team_elo"] - df["bowl_team_elo"]

# 4. Innings Totals & Targets (for Innings 1 & 2)
inn_agg = df[df["innings"].isin([1, 2])].groupby(["match_id", "innings"]).agg(
    inn_total_runs=("total_runs", "sum"),
    inn_legal_balls=("is_legal", "sum"),
    inn_wickets=("is_wicket", "sum"),
).reset_index()

inn1_totals = inn_agg[inn_agg["innings"] == 1].set_index("match_id")["inn_total_runs"].to_dict()
inn1_balls = inn_agg[inn_agg["innings"] == 1].set_index("match_id")["inn_legal_balls"].to_dict()

# Expanding Venue & CompGroup-Season Par RPO (strictly causal: prior matches only)
m_inn1 = m[["match_id", "match_date", "venue_id", "comp_group", "year"]].copy()
m_inn1["inn1_runs"] = m_inn1["match_id"].map(inn1_totals)
m_inn1["inn1_balls"] = m_inn1["match_id"].map(inn1_balls)
m_inn1 = m_inn1.dropna(subset=["inn1_runs", "inn1_balls"]).sort_values(["match_date", "match_id"]).reset_index(drop=True)
m_inn1["inn1_rpo"] = m_inn1["inn1_runs"] / np.maximum(m_inn1["inn1_balls"] / 6.0, 1.0)

# Prior comp_group expanding mean RPO (prior seeded with 8.8 for IPL, 8.4 for T20I, 7.9 for Emerging)
cg_prior_rpo = {"IPL": 8.8, "T20I": 8.4, "Emerging_T20": 7.9}
cg_cum_runs = {}
cg_cum_balls = {}
venue_cum_runs = {}
venue_cum_balls = {}

exp_cg_rpo = []
exp_venue_rpo = []
for _, r in m_inn1.iterrows():
    cg = r["comp_group"]
    vid = r["venue_id"]
    prior_cg = cg_prior_rpo.get(cg, 8.5)
    # Smoothed expanding comp_group RPO (weight=10 matches on prior)
    c_r = cg_cum_runs.get(cg, 0.0)
    c_b = cg_cum_balls.get(cg, 0.0)
    cg_rpo_val = (c_r + prior_cg * 200.0) / ((c_b + 1200.0) / 6.0)
    exp_cg_rpo.append(cg_rpo_val)

    # Smoothed expanding venue RPO (weight=6 matches on current comp_group RPO)
    v_r = venue_cum_runs.get(vid, 0.0)
    v_b = venue_cum_balls.get(vid, 0.0)
    v_rpo_val = (v_r + cg_rpo_val * (720.0 / 6.0)) / ((v_b + 720.0) / 6.0)
    exp_venue_rpo.append(v_rpo_val)

    cg_cum_runs[cg] = c_r + r["inn1_runs"]
    cg_cum_balls[cg] = c_b + r["inn1_balls"]
    venue_cum_runs[vid] = v_r + r["inn1_runs"]
    venue_cum_balls[vid] = v_b + r["inn1_balls"]

m_inn1["prior_cg_rpo"] = exp_cg_rpo
m_inn1["prior_venue_rpo"] = exp_venue_rpo

df = df.merge(m_inn1[["match_id", "prior_cg_rpo", "prior_venue_rpo", "inn1_runs"]], on="match_id", how="left")
df["prior_cg_rpo"] = df["prior_cg_rpo"].fillna(df["comp_group"].map(cg_prior_rpo))
df["prior_venue_rpo"] = df["prior_venue_rpo"].fillna(df["prior_cg_rpo"])

# 5. Cumulative Pre-Delivery Match State
g_inn = df.groupby(["match_id", "innings"], sort=False)

df["runs_before"] = g_inn["total_runs"].cumsum() - df["total_runs"]
df["wickets_before"] = g_inn["is_wicket"].cumsum() - df["is_wicket"].astype(int)
df["wickets_in_hand"] = np.clip(10 - df["wickets_before"], 0, 10)
df["legal_balls_before"] = g_inn["is_legal"].cumsum() - df["is_legal"].astype(int)

# Allocated max balls for innings (handle DLS revised overs, which are stored in balls!)
df["max_balls_inn"] = np.where(
    df["innings"] == 1,
    np.where(df["first_inn_revised_overs"].notna(), df["first_inn_revised_overs"], 120.0),
    np.where(
        df["innings"] == 2,
        np.where(df["second_inn_revised_overs"].notna(), df["second_inn_revised_overs"], 120.0),
        6.0  # Super over
    )
)
df["balls_remaining"] = np.clip(df["max_balls_inn"] - df["legal_balls_before"], 0.0, 120.0)

# Target & Required Run Rate (Innings 2)
df["target"] = np.where(
    df["innings"] == 2,
    np.where(df["revised_target"].notna(), df["revised_target"], df["inn1_runs"] + 1.0),
    np.nan
)
df["runs_needed"] = np.where(df["innings"] == 2, df["target"] - df["runs_before"], np.nan)
df["crr_before"] = np.where(
    df["legal_balls_before"] > 0,
    df["runs_before"] / (df["legal_balls_before"] / 6.0),
    df["prior_venue_rpo"]
)
df["rrr_before"] = np.where(
    (df["innings"] == 2) & (df["balls_remaining"] > 0),
    np.maximum(df["runs_needed"], 0.0) / (df["balls_remaining"] / 6.0),
    np.where(df["innings"] == 2, np.where(df["runs_needed"] > 0, 36.0, 0.0), np.nan)
)
df["rrr_before"] = np.clip(df["rrr_before"], 0.0, 36.0)
df["rrr_minus_crr"] = df["rrr_before"] - df["crr_before"]
df["rrr_ratio_par"] = df["rrr_before"] / np.maximum(df["prior_venue_rpo"], 5.0)

# Recent Momentum (rolling last 12 deliveries within innings, strictly BEFORE current ball)
df["is_dot_team"] = (df["total_runs"] == 0).astype(int)
df["is_boundary"] = (df["is_four"] | df["is_six"]).astype(int)

df["runs_last_12"] = (
    g_inn["total_runs"].rolling(12, min_periods=1).sum().reset_index(level=[0, 1], drop=True)
    - df["total_runs"]
)
df["wkts_last_12"] = (
    g_inn["is_wicket"].rolling(12, min_periods=1).sum().reset_index(level=[0, 1], drop=True)
    - df["is_wicket"].astype(int)
)
df["boundaries_last_12"] = (
    g_inn["is_boundary"].rolling(12, min_periods=1).sum().reset_index(level=[0, 1], drop=True)
    - df["is_boundary"]
)
df["dots_last_6"] = (
    g_inn["is_dot_team"].rolling(6, min_periods=1).sum().reset_index(level=[0, 1], drop=True)
    - df["is_dot_team"]
)

# Batter's own balls faced & runs in current innings PRIOR to delivery
g_bat_inn = df.groupby(["match_id", "innings", "striker_id"], sort=False)
df["batter_balls_in_inn"] = g_bat_inn["is_ball_faced"].cumsum() - df["is_ball_faced"].astype(int)
df["batter_runs_in_inn"] = g_bat_inn["runs"].cumsum() - df["runs"]

# Bowler's balls bowled & runs conceded in current innings PRIOR to delivery
g_bowl_inn = df.groupby(["match_id", "innings", "bowler_id"], sort=False)
df["bowler_balls_in_inn"] = g_bowl_inn["is_legal"].cumsum() - df["is_legal"].astype(int)
df["bowler_runs_in_inn"] = g_bowl_inn["bowler_runs"].cumsum() - df["bowler_runs"]

# Phase & Matchups
df["phase"] = np.where(df["overid"] <= 6, "Powerplay", np.where(df["overid"] <= 15, "Middle", "Death"))
df["is_pace"] = (df["pace_or_spin"] == "Pace").astype(int)
df["is_rhb"] = (df["batting_type"] == "RHB").astype(int)
df["bowling_hand"] = np.where(df["bowling_type"].astype(str).str.contains("Left|Slow left", case=False), "Left", "Right")

# Playing XI features (Impact player, captain, wk, team batting depth)
pe_clean = pe.drop_duplicates(subset=["match_id", "team_id", "player_id"])
df = df.merge(
    pe_clean[["match_id", "team_id", "player_id", "is_captain", "is_wk", "is_impact"]].rename(
        columns={"team_id": "batting_team_id", "player_id": "striker_id",
                 "is_captain": "striker_is_capt", "is_wk": "striker_is_wk", "is_impact": "striker_is_impact"}
    ),
    on=["match_id", "batting_team_id", "striker_id"], how="left"
)

# Match-has-impact-player flag (IPL 2023+ allows 12 batters effectively)
match_impact = pe_clean.groupby("match_id")["is_impact"].max().to_dict()
df["match_has_impact_rule"] = df["match_id"].map(match_impact).fillna(0).astype(int)

# Batting team won flag (target for Win Probability model)
df["bat_team_won"] = np.where(
    df["win_team_id"].notna(),
    (df["batting_team_id"] == df["win_team_id"]).astype(float),
    np.nan
)

# Save canonical player dictionary & engineered dataset
pd.DataFrame(list(id_to_name.items()), columns=["player_id", "player_name"]).to_csv(
    os.path.join(OUT_DIR, "player_id_map.csv"), index=False
)
df.to_pickle(os.path.join(OUT_DIR, "deliveries.pkl"))
m_inn1.to_csv(os.path.join(OUT_DIR, "match_inn1_par.csv"), index=False)

print(f"Saved deliveries.pkl with shape {df.shape}")
print("Summary checks:")
print("  Innings 1 & 2 rows:", df["innings"].isin([1, 2]).sum())
print("  Decisive matches rows (Inn 1 & 2):", df[df["innings"].isin([1, 2]) & df["bat_team_won"].notna()].shape[0])
print("  Top-tier matches:", df["is_top_tier_match"].mean() * 100, "% of deliveries")
print("  Mean prior_venue_rpo:", df["prior_venue_rpo"].describe().to_dict())
print("  Mean bat_team_elo:", df["bat_team_elo"].describe().to_dict())
