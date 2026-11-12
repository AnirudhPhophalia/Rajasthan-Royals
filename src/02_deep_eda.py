import os
import pandas as pd
import numpy as np

DATA_DIR = r"d:\Rajasthan Royals\Data Pack"
comp = pd.read_csv(os.path.join(DATA_DIR, "competitions.csv"))
matches = pd.read_csv(os.path.join(DATA_DIR, "matches.csv"))
bbb = pd.read_csv(os.path.join(DATA_DIR, "ball_by_ball.csv"), low_memory=False)
pe = pd.read_csv(os.path.join(DATA_DIR, "playing_eleven.csv"))
ctpm = pd.read_csv(os.path.join(DATA_DIR, "competition_team_player_mapping.csv"))

m_comp = matches.merge(comp, on="comp_id", how="left")

print("=== 1. CHECK 'other' EXTRAS_TYPE & 'extras_2_runs' > 0 & 'is_wicket==False & dismissed_player_id notna' ===")
print("Sample 'other' extras_type with full over:")
m_oth = bbb[bbb["extras_type"] == "other"].iloc[0]
print(bbb[(bbb["match_id"] == m_oth["match_id"]) & (bbb["innings"] == m_oth["innings"]) & (bbb["overid"] == m_oth["overid"])][
    ["overid", "over_ball_num", "innings_ball_num", "runs", "extras_type", "extras1_runs", "extras_2_runs", "is_wicket"]
].to_string(index=False))

print("\nRows with extras_2_runs > 0:")
print(bbb[bbb["extras_2_runs"] > 0][["match_id", "innings", "overid", "runs", "extras_type", "extras1_runs", "extras_2_runs"]].head(15).to_string(index=False))

print("\nRows with extras_type == None and extras1_runs > 0:")
print(bbb[bbb["extras_type"].isna() & (bbb["extras1_runs"] > 0)][["match_id", "innings", "overid", "runs", "extras_type", "extras1_runs", "extras_2_runs"]].to_string(index=False))

print("\nRows with is_wicket == False and dismissed_player_id notna:")
print(bbb[(bbb["is_wicket"] == False) & bbb["dismissed_player_id"].notna()][
    ["match_id", "innings", "overid", "striker_name", "is_wicket", "dismissal_type", "dismissed_player_id"]
].to_string(index=False))

print("\n=== 2. CHECK DUPLICATE-NAME PLAYER IDS ===")
dup_ids = [7446, 6922, 29833, 39479, 8638, 39444, 7338, 28756, 20229, 3083, 27373, 28978, 4722, 20172]
for pid in dup_ids:
    sub_b = bbb[(bbb["striker_id"] == pid) | (bbb["bowler_id"] == pid)]
    teams_b = ctpm[ctpm["player_id"] == pid][["team_name", "comp_name"]].drop_duplicates()
    pname = teams_b["player_name"].iloc[0] if "player_name" in teams_b else ""
    t_list = teams_b["team_name"].unique().tolist()
    print(f"ID {pid}: balls_bat={(bbb['striker_id']==pid).sum()}, balls_bowl={(bbb['bowler_id']==pid).sum()}, teams={t_list[:5]}")

print("\n=== 3. ALL T20I COMPETITIONS & TEAMS ===")
t20i_teams = pd.concat([
    m_comp[m_comp["match_format"] == "T20I"][["home_team_id", "home_team"]].rename(columns={"home_team_id": "team_id", "home_team": "team_name"}),
    m_comp[m_comp["match_format"] == "T20I"][["away_team_id", "away_team"]].rename(columns={"away_team_id": "team_id", "away_team": "team_name"}),
])
print(t20i_teams["team_name"].value_counts().to_string())

print("\n=== 4. CHECK FOR WOMEN'S / U19 / NON-SENIOR COMPETITIONS ===")
print(comp[~comp["comp_name"].str.contains("IPL", na=False)][["comp_id", "comp_name", "match_format", "season"]].sort_values(["season", "comp_name"]).to_string(index=False))

print("\n=== 5. CHECK BOWLING_TYPE & BATTING_TYPE & VENUE DISTRIBUTION ===")
print("Top 15 bowling_type:")
print(bbb["bowling_type"].value_counts(dropna=False).head(15).to_string())
print("\nMatches with >120 legal balls or >20 overs in innings 1 or 2:")
bbb["is_legal"] = ~bbb["extras_type"].isin(["wide", "Overstep", "other"])
inn_legal = bbb[bbb["innings"].isin([1, 2])].groupby(["match_id", "innings"])["is_legal"].sum().reset_index()
print("Max legal balls in an innings:", inn_legal["is_legal"].max())
print("Innings with >120 legal balls:", inn_legal[inn_legal["is_legal"] > 120].to_string(index=False))

print("\n=== 6. CHECK MATCH COVERAGE: MATCHES IN matches.csv WITHOUT bbb ===")
no_bbb = set(matches["match_id"]) - set(bbb["match_id"])
print(matches[matches["match_id"].isin(no_bbb)][["match_id", "match_date", "home_team", "away_team", "result"]].to_string(index=False))

