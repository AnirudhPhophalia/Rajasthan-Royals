import os
import pandas as pd
import numpy as np

bat = pd.read_csv(r"d:\Rajasthan Royals\outputs\rankings\final_batters_rankings.csv")
bowl = pd.read_csv(r"d:\Rajasthan Royals\outputs\rankings\final_bowlers_rankings.csv")
df = pd.read_pickle(r"d:\Rajasthan Royals\outputs\processed\deliveries_modeled.pkl")

print("=== TOP 10 BATTERS FULL PROFILE ===")
cols_b = [
    "rank", "player_name", "player_id", "ipl_matches", "t20i_matches",
    "total_balls", "hi_balls", "hi_matches", "ko_hi_balls",
    "lo_sr_or_econ", "hi_sr_or_econ", "lo_wkt_rate", "hi_wkt_rate",
    "lo_rva100_eb", "hi_rva100_eb", "hi_raa100", "hi_waa100", "res_rva100",
    "hi_wpa100_eb", "pmr_eb", "SPRI", "spri_ci_low", "spri_ci_high", "p_top10"
]
print(bat[cols_b].head(10).round(3).to_string(index=False))

print("\n=== TOP 10 BOWLERS FULL PROFILE ===")
cols_bw = [
    "rank", "player_name", "player_id", "ipl_matches", "t20i_matches",
    "total_balls", "hi_balls", "hi_matches", "ko_hi_balls",
    "lo_sr_or_econ", "hi_sr_or_econ", "lo_wkt_rate", "hi_wkt_rate",
    "lo_rva100_eb", "hi_rva100_eb", "hi_raa100", "hi_waa100", "res_rva100",
    "hi_wpa100_eb", "pmr_eb", "SPRI", "spri_ci_low", "spri_ci_high", "p_top10"
]
print(bowl[cols_bw].head(10).round(3).to_string(index=False))

# Top iconic clutch match performances by Top 5 batters and Top 5 bowlers
top5_bat = bat.head(5)["player_id"].tolist()
bat_d = df[df["is_ball_faced"] & df["striker_id"].isin(top5_bat) & (df["pressure_bat"] >= 0.70)]
bat_spells = bat_d.groupby(["striker_name", "match_id", "match_date", "comp_name", "home_team", "away_team", "stage"]).agg(
    hi_balls=("runs", "count"),
    hi_runs=("runs", "sum"),
    hi_rva=("rva_bat", "sum"),
    hi_wpa=("wpa_bat", "sum"),
    mean_p=("pressure_bat", "mean"),
).reset_index()
bat_spells = bat_spells[bat_spells["hi_balls"] >= 10].sort_values("hi_rva", ascending=False)
print("\n=== TOP CLUTCH INNINGS BY TOP 5 BATTERS ===")
print(bat_spells.head(12).round(3).to_string(index=False))

top5_bowl = bowl.head(5)["player_id"].tolist()
bowl_d = df[df["bowler_id"].isin(top5_bowl) & (df["pressure_bowl"] >= 0.70)]
bowl_spells = bowl_d.groupby(["bowler_name", "match_id", "match_date", "comp_name", "home_team", "away_team", "stage"]).agg(
    hi_balls=("is_legal", "sum"),
    hi_runs=("bowler_runs", "sum"),
    hi_wkts=("is_bowler_wicket", "sum"),
    hi_rva=("rva_bowl", "sum"),
    hi_wpa=("wpa_bowl", "sum"),
    mean_p=("pressure_bowl", "mean"),
).reset_index()
bowl_spells = bowl_spells[bowl_spells["hi_balls"] >= 10].sort_values("hi_rva", ascending=False)
print("\n=== TOP CLUTCH SPELLS BY TOP 5 BOWLERS ===")
print(bowl_spells.head(12).round(3).to_string(index=False))

