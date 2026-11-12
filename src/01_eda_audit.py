import os
import pandas as pd
import numpy as np

DATA_DIR = r"d:\Rajasthan Royals\Data Pack"
OUT_DIR = r"d:\Rajasthan Royals\outputs\eda"
os.makedirs(OUT_DIR, exist_ok=True)

comp = pd.read_csv(os.path.join(DATA_DIR, "competitions.csv"))
matches = pd.read_csv(os.path.join(DATA_DIR, "matches.csv"))
bbb = pd.read_csv(os.path.join(DATA_DIR, "ball_by_ball.csv"), low_memory=False)
pe = pd.read_csv(os.path.join(DATA_DIR, "playing_eleven.csv"))
ctpm = pd.read_csv(os.path.join(DATA_DIR, "competition_team_player_mapping.csv"))
teams = pd.read_csv(os.path.join(DATA_DIR, "teams.csv"))

print("=== EXTRAS & RUNS ACCOUNTING ===")
print("extras_2_runs value counts:")
print(bbb["extras_2_runs"].value_counts(dropna=False).sort_index())

bbb["et_filled"] = bbb["extras_type"].fillna("None")
print("\nCross-tab of extras_type with runs, extras1_runs, extras_2_runs (min/max/mean/count):")
print(bbb.groupby("et_filled").agg(
    count=("runs", "count"),
    runs_min=("runs", "min"),
    runs_max=("runs", "max"),
    runs_sum=("runs", "sum"),
    e1_min=("extras1_runs", "min"),
    e1_max=("extras1_runs", "max"),
    e1_sum=("extras1_runs", "sum"),
    e2_min=("extras_2_runs", "min"),
    e2_max=("extras_2_runs", "max"),
    e2_sum=("extras_2_runs", "sum"),
    is_four_sum=("is_four", "sum"),
    is_six_sum=("is_six", "sum"),
    is_wkt_sum=("is_wicket", "sum"),
).to_string())

print("\nSample rows for each extras_type (wide, Overstep, legbyes, byes, other, Penalty):")
for et in ["wide", "Overstep", "legbyes", "byes", "other", "Penalty"]:
    sub = bbb[bbb["extras_type"] == et][["match_id", "innings", "overid", "over_ball_num", "innings_ball_num", "runs", "is_four", "is_six", "extras_type", "extras1_runs", "extras_2_runs", "is_wicket", "dismissal_type"]].head(5)
    print(f"\n-- {et} --")
    print(sub.to_string(index=False))

print("\n=== CHECK INNINGS SCORE RECONSTRUCTION AGAINST RESULT ===")
# Check a few "Won by X runs" matches to verify total_runs formula
m_sub = matches[matches["result"].astype(str).str.contains("Won by", case=False, na=False)].head(10)
for _, r in m_sub.iterrows():
    mb = bbb[(bbb["match_id"] == r["match_id"]) & (bbb["innings"].isin([1, 2]))]
    s1 = mb[mb["innings"] == 1][["runs", "extras1_runs", "extras_2_runs"]].sum().to_dict()
    s2 = mb[mb["innings"] == 2][["runs", "extras1_runs", "extras_2_runs"]].sum().to_dict()
    t1 = s1["runs"] + s1["extras1_runs"] + s1["extras_2_runs"]
    t2 = s2["runs"] + s2["extras1_runs"] + s2["extras_2_runs"]
    print(f"Match {r['match_id']} | {r['result']} | Inn1: runs={s1['runs']} e1={s1['extras1_runs']} e2={s1['extras_2_runs']} tot={t1} | Inn2: runs={s2['runs']} e1={s2['extras1_runs']} e2={s2['extras_2_runs']} tot={t2} | diff={t1-t2}")

print("\n=== CHECK INNINGS_BALL_NUM INCREMENT ON EXTRAS ===")
# Take a match with wide and Overstep and inspect sequential rows
sample_m = bbb[bbb["extras_type"] == "Overstep"]["match_id"].iloc[0]
sb = bbb[bbb["match_id"] == sample_m].reset_index(drop=True)
idx = sb[sb["extras_type"] == "Overstep"].index[0]
print(sb.iloc[max(0, idx-2):min(len(sb), idx+4)][["innings", "overid", "over_ball_num", "innings_ball_num", "runs", "extras_type", "extras1_runs", "extras_2_runs"]].to_string(index=False))

sample_m2 = bbb[bbb["extras_type"] == "wide"]["match_id"].iloc[0]
sb2 = bbb[bbb["match_id"] == sample_m2].reset_index(drop=True)
idx2 = sb2[sb2["extras_type"] == "wide"].index[0]
print("\nWide sequence:")
print(sb2.iloc[max(0, idx2-2):min(len(sb2), idx2+4)][["innings", "overid", "over_ball_num", "innings_ball_num", "runs", "extras_type", "extras1_runs", "extras_2_runs"]].to_string(index=False))

print("\n=== INNINGS 3 & 4 (SUPER OVERS) ===")
print(bbb[bbb["innings"] >= 3][["match_id", "innings", "overid", "over_ball_num", "innings_ball_num", "batting_team_id", "striker_name", "bowler_name", "runs", "extras_type", "extras1_runs", "is_wicket"]].to_string(index=False))

print("\n=== REVISED OVERS / DLS MATCHES ===")
rev = matches[matches["first_inn_revised_overs"].notna() | matches["second_inn_revised_overs"].notna() | matches["revised_target"].notna()]
print(rev[["match_id", "match_date", "home_team", "away_team", "first_inn_revised_overs", "second_inn_revised_overs", "revised_target", "result"]].to_string(index=False))

print("\n=== T20 COMPETITIONS (NON-T20I) ===")
m_comp = matches.merge(comp, on="comp_id", how="left")
print(m_comp[m_comp["match_format"] == "T20"].groupby(["comp_id", "comp_name", "season"])["match_id"].count().to_string())

print("\n=== PLAYER INTEGRITY ===")
all_p = pd.concat([
    bbb[["striker_id", "striker_name"]].rename(columns={"striker_id": "player_id", "striker_name": "player_name"}),
    bbb[["bowler_id", "bowler_name"]].rename(columns={"bowler_id": "player_id", "bowler_name": "player_name"}),
    bbb[["non_striker_id", "non_striker_name"]].rename(columns={"non_striker_id": "player_id", "non_striker_name": "player_name"}),
    ctpm[["player_id", "player_name"]],
]).drop_duplicates()
dup_id = all_p.groupby("player_id")["player_name"].nunique()
print("Player IDs with >1 name:", len(dup_id[dup_id > 1]))
if len(dup_id[dup_id > 1]) > 0:
    print(all_p[all_p["player_id"].isin(dup_id[dup_id > 1].index)].sort_values("player_id").to_string(index=False))

dup_name = all_p.groupby("player_name")["player_id"].nunique()
print("Player names with >1 ID:", len(dup_name[dup_name > 1]))
if len(dup_name[dup_name > 1]) > 0:
    print(all_p[all_p["player_name"].isin(dup_name[dup_name > 1].index)].sort_values("player_name").to_string(index=False))

print("\n=== DISMISSAL TYPES & DISMISSED_PLAYER_ID ===")
wkt = bbb[bbb["is_wicket"] == True]
print(wkt["dismissal_type"].value_counts(dropna=False))
print("Wickets where dismissed_player_id != striker_id:", (wkt["dismissed_player_id"] != wkt["striker_id"]).sum())
print(wkt[wkt["dismissed_player_id"] != wkt["striker_id"]][["dismissal_type", "striker_id", "non_striker_id", "dismissed_player_id"]].head(10).to_string(index=False))
print("Rows where is_wicket == False but dismissed_player_id not null:", bbb[(bbb["is_wicket"] == False) & bbb["dismissed_player_id"].notna()].shape[0])

print("\n=== BOWL_SPEED & METADATA COVERAGE BY COMP FORMAT ===")
bbb_m = bbb.merge(m_comp[["match_id", "match_format", "comp_name", "season"]], on="match_id", how="left")
print(bbb_m.groupby("match_format").agg(
    balls=("match_id", "count"),
    matches=("match_id", "nunique"),
    speed_pct=("bowl_speed", lambda s: (s > 0).mean()*100),
    stroke_pct=("stroke", lambda s: s.notna().mean()*100),
    style_pct=("delivery_style", lambda s: s.notna().mean()*100),
))
