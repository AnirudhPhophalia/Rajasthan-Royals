import os
import pandas as pd
import numpy as np

DATA_DIR = r"d:\Rajasthan Royals\Data Pack"
comp = pd.read_csv(os.path.join(DATA_DIR, "competitions.csv"))
matches = pd.read_csv(os.path.join(DATA_DIR, "matches.csv"))
bbb = pd.read_csv(os.path.join(DATA_DIR, "ball_by_ball.csv"), low_memory=False)
pe = pd.read_csv(os.path.join(DATA_DIR, "playing_eleven.csv"))

m_comp = matches.merge(comp, on="comp_id", how="left")
m_comp["match_date"] = pd.to_datetime(m_comp["match_date"])
m_comp["year"] = m_comp["match_date"].dt.year

# Classify competition category
def classify_comp(row):
    name = str(row["comp_name"])
    if "IPL" in name:
        return "IPL"
    elif row["match_format"] == "T20":
        return "Other_T20_Emerging"
    else:
        return "T20I"

m_comp["comp_group"] = m_comp.apply(classify_comp, axis=1)

# Delivery-level accounting
bbb["is_wide"] = bbb["extras_type"] == "wide"
bbb["is_noball"] = bbb["extras_type"].isin(["Overstep", "other"])
bbb["is_legal"] = ~(bbb["is_wide"] | bbb["is_noball"])
bbb["total_runs"] = bbb["runs"] + bbb["extras1_runs"] + bbb["extras_2_runs"]
# Bat ball faced: legal + noball (not wide)
bbb["is_ball_faced"] = ~bbb["is_wide"]
# Bowler runs conceded: runs + extras if wide/noball (except extras_2_runs on noball which are byes/legbyes)
bbb["bowler_runs"] = np.where(
    bbb["is_wide"] | bbb["is_noball"],
    bbb["runs"] + bbb["extras1_runs"],
    bbb["runs"]
)
bbb["is_bowler_wicket"] = bbb["is_wicket"] & (~bbb["dismissal_type"].isin(["RunOut", "Retire Out", "Obstracting the field", "Obstructing the field"]))

bm = bbb.merge(m_comp[["match_id", "comp_group", "match_format", "season", "year", "knock_out", "stage", "win_team_id", "result"]], on="match_id", how="left")

print("=== 1. RUN RATES & BOUNDARY RATES BY COMP_GROUP & YEAR ===")
reg = bm[bm["innings"].isin([1, 2])]
grp = reg.groupby(["comp_group", "year"]).agg(
    matches=("match_id", "nunique"),
    balls=("total_runs", "count"),
    legal_balls=("is_legal", "sum"),
    total_runs=("total_runs", "sum"),
    wkts=("is_wicket", "sum"),
    fours=("is_four", "sum"),
    sixes=("is_six", "sum"),
).reset_index()
grp["rpo"] = grp["total_runs"] / (grp["legal_balls"] / 6.0)
grp["balls_per_wkt"] = grp["legal_balls"] / grp["wkts"]
grp["boundary_pct"] = (grp["fours"] + grp["sixes"]) / grp["legal_balls"] * 100
print(grp.to_string(index=False))

print("\n=== 2. INNINGS 1 VS INNINGS 2 BY COMP_GROUP ===")
inn_g = reg.groupby(["comp_group", "innings"]).agg(
    innings_cnt=("match_id", "nunique"),
    legal_balls=("is_legal", "sum"),
    total_runs=("total_runs", "sum"),
    wkts=("is_wicket", "sum"),
).reset_index()
inn_g["rpo"] = inn_g["total_runs"] / (inn_g["legal_balls"] / 6.0)
inn_g["avg_score"] = inn_g["total_runs"] / inn_g["innings_cnt"]
print(inn_g.to_string(index=False))

print("\n=== 3. FULL MEMBER VS ASSOCIATE TEAMS IN T20I ===")
full_members = {
    "India", "Australia", "England", "South Africa", "New Zealand",
    "Pakistan", "West Indies", "Sri Lanka", "Bangladesh", "Afghanistan",
    "Ireland", "Zimbabwe"
}
t20i_m = m_comp[m_comp["comp_group"] == "T20I"].copy()
t20i_m["home_fm"] = t20i_m["home_team"].isin(full_members)
t20i_m["away_fm"] = t20i_m["away_team"].isin(full_members)
t20i_m["tier"] = np.where(t20i_m["home_fm"] & t20i_m["away_fm"], "FM_vs_FM",
                 np.where(t20i_m["home_fm"] | t20i_m["away_fm"], "FM_vs_Assoc", "Assoc_vs_Assoc"))
print(t20i_m["tier"].value_counts())

print("\n=== 4. PLAYER SAMPLE SIZE DISTRIBUTIONS (BATTERS & BOWLERS) ===")
# Exclude Other_T20_Emerging? Let's check overlap first
bat_stats = reg.groupby(["striker_id", "striker_name"]).agg(
    matches=("match_id", "nunique"),
    ipl_matches=("match_id", lambda s: s[reg.loc[s.index, "comp_group"] == "IPL"].nunique()),
    t20i_matches=("match_id", lambda s: s[reg.loc[s.index, "comp_group"] == "T20I"].nunique()),
    emg_matches=("match_id", lambda s: s[reg.loc[s.index, "comp_group"] == "Other_T20_Emerging"].nunique()),
    balls_faced=("is_ball_faced", "sum"),
    runs=("runs", "sum"),
).reset_index()

print("Total unique batters:", len(bat_stats))
for thr in [100, 200, 300, 400, 500, 750, 1000]:
    print(f"  Batters with >= {thr} balls faced: {(bat_stats['balls_faced'] >= thr).sum()}")

bowl_stats = reg.groupby(["bowler_id", "bowler_name"]).agg(
    matches=("match_id", "nunique"),
    ipl_matches=("match_id", lambda s: s[reg.loc[s.index, "comp_group"] == "IPL"].nunique()),
    t20i_matches=("match_id", lambda s: s[reg.loc[s.index, "comp_group"] == "T20I"].nunique()),
    emg_matches=("match_id", lambda s: s[reg.loc[s.index, "comp_group"] == "Other_T20_Emerging"].nunique()),
    legal_balls=("is_legal", "sum"),
    runs_conceded=("bowler_runs", "sum"),
    wkts=("is_bowler_wicket", "sum"),
).reset_index()

print("\nTotal unique bowlers:", len(bowl_stats))
for thr in [100, 200, 300, 400, 500, 750, 1000]:
    print(f"  Bowlers with >= {thr} legal balls bowled: {(bowl_stats['legal_balls'] >= thr).sum()}")

print("\nTop 20 Batters by balls faced:")
print(bat_stats.sort_values("balls_faced", ascending=False).head(20).to_string(index=False))

print("\nTop 20 Bowlers by legal balls bowled:")
print(bowl_stats.sort_values("legal_balls", ascending=False).head(20).to_string(index=False))

print("\n=== 5. PHASE-WISE RUN RATE & WICKET RATE (OVER 1-20) ===")
phase_g = reg.groupby("overid").agg(
    balls=("is_legal", "sum"),
    runs=("total_runs", "sum"),
    wkts=("is_wicket", "sum"),
    wides=("is_wide", "sum"),
    noballs=("is_noball", "sum"),
).reset_index()
phase_g["rpo"] = phase_g["runs"] / (phase_g["balls"] / 6.0)
phase_g["wkt_pct"] = phase_g["wkts"] / phase_g["balls"] * 100
phase_g["extra_pct"] = (phase_g["wides"] + phase_g["noballs"]) / phase_g["balls"] * 100
print(phase_g.to_string(index=False))

print("\n=== 6. CHECK CHASE WIN RATE BY INNINGS 1 SCORE & TARGET ===")
m_scores = reg.groupby(["match_id", "innings"]).agg(
    runs=("total_runs", "sum"),
    legal_balls=("is_legal", "sum"),
    wkts=("is_wicket", "sum"),
    bat_team=("batting_team_id", "first"),
).reset_index()
i1 = m_scores[m_scores["innings"] == 1].set_index("match_id")
i2 = m_scores[m_scores["innings"] == 2].set_index("match_id")
comb = m_comp.set_index("match_id").join(i1[["runs", "bat_team"]].rename(columns={"runs": "i1_runs", "bat_team": "i1_team"}))
comb = comb.join(i2[["runs", "bat_team"]].rename(columns={"runs": "i2_runs", "bat_team": "i2_team"}))
valid = comb[comb["win_team_id"].notna() & comb["i1_runs"].notna() & comb["i2_runs"].notna() & comb["first_inn_revised_overs"].isna() & comb["second_inn_revised_overs"].isna()].copy()
valid["chase_won"] = (valid["win_team_id"] == valid["i2_team"]).astype(int)
print(f"Full 20-over completed matches with winner: {len(valid)}, Chase win rate: {valid['chase_won'].mean():.3f}")
print("Chase win rate by comp_group:")
print(valid.groupby("comp_group")["chase_won"].agg(["count", "mean"]))
print("Chase win rate by Innings 1 score buckets:")
valid["i1_bucket"] = pd.cut(valid["i1_runs"], bins=[0, 130, 155, 175, 195, 215, 350])
print(valid.groupby("i1_bucket", observed=False)["chase_won"].agg(["count", "mean"]))

