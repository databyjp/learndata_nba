# ========== (c) JP Hwang 25/8/20  ==========

import logging

# ===== START LOGGER =====
logger = logging.getLogger(__name__)
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
sh = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
sh.setFormatter(formatter)
root_logger.addHandler(sh)

import pandas as pd
import numpy as np
import plotly.express as px

desired_width = 320
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', desired_width)

totals_df = pd.read_csv("data/player_totals.csv", index_col=0).reset_index(drop=True)

keys = ['name', 'pos', 'age', 'team_id', 'g', 'mp',
        'fg', 'fga', 'fg3', 'fg3a', 'fg2', 'fg2a', 'ft', 'fta',
        # 'fg_pct', 'fg3_pct', 'fg2_pct', 'efg_pct', 'ft_pct',
        'link', 'season']

totals_df = totals_df[keys]

# show players' progression at different ages

# ===== Pre-processing -> aggregate players' stats if traded in-season
# for yr in totals_df.season.unique():
#     tmp_df = totals_df[totals_df.season == yr]
#     dup_df = tmp_df[tmp_df.name.duplicated()]
#     for name in dup_df.name.unique():
#         player_df = dup_df[dup_df.name == name]
#         agg_stat = player_df.sum(axis=0)
#         agg_stat[["name", "pos", "age", "season"]] = player_df.iloc[0][["name", "pos", "age", "season"]]
#         agg_stat["team_id"] = "NBA"
#         totals_df = totals_df[(totals_df.name != name) | (totals_df.season != yr)]
#         totals_df = totals_df.append(agg_stat, ignore_index=True)
for yr in totals_df.season.unique():
    tmp_df = totals_df[totals_df.season == yr]
    traded_df = tmp_df[tmp_df.team_id == 'TOT']
    for name in traded_df.name.unique():
        player_df = traded_df[traded_df.name == name]
        totals_df = totals_df[(totals_df.name != name) | (totals_df.season != yr)]
        totals_df = pd.concat([totals_df, player_df], axis=0)

totals_df = totals_df.assign(mp_pct=0)
for season in totals_df.season.unique():
    mp_sum = totals_df[totals_df.season==season].mp.sum()
    totals_df.loc[totals_df.season==season, "mp_pct"] = totals_df.loc[totals_df.season==season, "mp"]/mp_sum * 100

# ===== Filter low-usage players
# totals_sm_df = totals_df[totals_df.mp > 1000]

# ===== Is the average on-court age getting younger?
labels_dict = {"age": "Age", "season": "Season", "mp_pct": "% of <BR>minutes played", "age_pct": "% of players"}

# What does the age distribution of the league look like currently?
tmp_df = totals_df[(totals_df.season == 2020)]
grp_df = tmp_df.groupby(["season", "age"]).count()["name"].reset_index()
grp_df = grp_df.assign(age_pct=0)
for season in grp_df.season.unique():
    players = grp_df[grp_df.season==season].name.sum()
    grp_df.loc[grp_df.season==season, "age_pct"] = grp_df.loc[grp_df.season==season, "name"]/players * 100

fig = px.bar(
    grp_df, x="age", y="age_pct",
    title="NBA Player Age Distribution in 2020", color="age_pct",
    template="plotly_white", color_continuous_scale=px.colors.sequential.YlGnBu,
    category_orders={"season": list(np.sort(tmp_df.season.unique()))},
    labels=labels_dict,
    hover_data=["name"],
    height=400, width=1200
)
fig.update_layout(bargap=0.3)
fig.update_traces(marker={"line": {"width": 0.4, "color": "black"}})
fig.show()

# What about over time?
tmp_df = totals_df[
    (totals_df.season == 1989) | (totals_df.season == 1999) |
    (totals_df.season == 2009) | (totals_df.season == 2019)
    ]
grp_df = tmp_df.groupby(["season", "age"]).count()["name"].reset_index()
grp_df = grp_df.assign(age_pct=0)
for season in grp_df.season.unique():
    players = grp_df[grp_df.season==season].name.sum()
    grp_df.loc[grp_df.season==season, "age_pct"] = grp_df.loc[grp_df.season==season, "name"]/players * 100

fig = px.bar(
    grp_df, x="age", y="age_pct",
    title="NBA Player Age Distribution Over Time", color="age_pct",
    facet_row="season",
    template="plotly_white", color_continuous_scale=px.colors.sequential.YlGnBu,
    category_orders={"season": list(np.sort(tmp_df.season.unique()))},
    labels=labels_dict,
    hover_data=["name"],
    width=1200
)
fig.update_layout(bargap=0.3)
fig.update_traces(marker={"line": {"width": 0.4, "color": "black"}})
fig.show()

# What if we adjust for playing time?
fig = px.bar(
    tmp_df.sort_values("mp_pct"), x="age", y="mp_pct", color="mp_pct",
    title="Is the NBA getting any younger?",
    facet_row="season", template="plotly_white", color_continuous_scale=px.colors.sequential.YlGnBu,
    category_orders={"season": list(np.sort(tmp_df.season.unique()))},
    labels=labels_dict,
    hover_data=["name"]
)
fig.show()

# Let's look at the mean (or weight age of players on the court -
tmp_df = totals_df[
    (totals_df.season == 1989) | (totals_df.season == 1999) |
    (totals_df.season == 2009) | (totals_df.season == 2019)
    ]
grp_df = tmp_df.groupby(["season", "age"]).sum()["mp"].reset_index()
grp_df = grp_df.assign(age_pct=0)
for season in grp_df.season.unique():
    players = grp_df[grp_df.season==season].mp.sum()
    grp_df.loc[grp_df.season==season, "age_pct"] = grp_df.loc[grp_df.season==season, "mp"]/players * 100

fig = px.bar(
    grp_df, x="age", y="age_pct",
    title="NBA Playing Time Age Distribution Over Time", color="age_pct",
    facet_row="season",
    template="plotly_white", color_continuous_scale=px.colors.sequential.YlGnBu,
    category_orders={"season": list(np.sort(tmp_df.season.unique()))},
    labels=labels_dict,
    width=1200
)
fig.update_layout(bargap=0.3)
fig.update_traces(marker={"line": {"width": 0.4, "color": "black"}})
fig.show()

# Interesting - let's plot the annual average (mean) and typical (median) ages over time
totals_df = totals_df.assign(mp_age=totals_df.mp * totals_df.age)
totals_df = totals_df.assign(mp_age_norm=0)
for season in totals_df.season.unique():
    totals_df.loc[totals_df.season==season, "mp_age_norm"] = totals_df.loc[totals_df.season==season, "mp_age"] / totals_df[totals_df.season==season].mp.sum()
mean_ages = [totals_df[totals_df.season==yr].age.mean() for yr in totals_df.season.unique()]
mean_wt_ages = [totals_df[totals_df.season==yr].mp_age_norm.sum() for yr in totals_df.season.unique()]
ann_df = pd.DataFrame([mean_ages, mean_wt_ages, totals_df.season.unique()], index=["mean", "weighted mean", "season"]).transpose()
ann_df = ann_df.melt(id_vars=["season"])

fig = px.scatter(
    ann_df, x="season", y="value",
    title="NBA Player Age Distribution Over Time", color="variable",
    template="plotly_white", color_discrete_sequence=px.colors.qualitative.Safe,
    category_orders={"season": list(np.sort(tmp_df.season.unique()))},
    labels=labels_dict,
    height=600, width=1200
)
fig.update_layout(bargap=0.3)
fig.update_traces(mode="markers+lines", marker={"line": {"width": 0.4, "color": "black"}})
fig.show()

# Split by groups of playing time
# Assign players into one of 3 tiers
# Pick top 2 (players * n_teams)
# Top 3-7 (players * n_teams)
# rest
totals_df = totals_df.assign(pl_tier=3)
tier_players = [8, 2]
for season in totals_df.season.unique():
    n_teams = len(totals_df[(totals_df.season==season)].team_id.unique()) - 1
    for i in range(len(tier_players)):
        n_pl = tier_players[i]
        n_thresh = n_teams * n_pl
        thresh = totals_df[(totals_df.season==season)].mp.sort_values().iloc[-n_thresh]
        totals_df.loc[(totals_df.season==season) & (totals_df.mp > thresh), "pl_tier"] = len(tier_players) - i

tmp_df = totals_df[
    (totals_df.season == 2019)
    ]
fig = px.bar(
    tmp_df.sort_values("mp_pct"), x="age", y="mp_pct", color="mp_pct",
    title="Is the NBA getting any younger?",
    facet_row="pl_tier", template="plotly_white", color_continuous_scale=px.colors.sequential.YlGnBu,
    category_orders={"season": list(np.sort(tmp_df.season.unique()))},
    labels=labels_dict,
    hover_data=["name"]
)
fig.show()


# Get mean ages for each tier
mean_tier_ages = list()
ovr_mean_ages = [totals_df[totals_df.season==yr].age.mean() for yr in totals_df.season.unique()]
for pl_tier in [1, 2, 3]:
    tier_ages = [totals_df[(totals_df.season==yr) & (totals_df.pl_tier==pl_tier)].age.mean() for yr in totals_df.season.unique()]
    tier_ages = [tier_ages[i]-ovr_mean_ages[i] for i in range(len(tier_ages))]
    mean_tier_ages.append(tier_ages)
mean_tier_ages.append(totals_df.season.unique())
ann_df = pd.DataFrame(mean_tier_ages, index=["Tier 1", "Tier 2", "Tier 3", "season"]).transpose()
ann_df = ann_df.melt(id_vars=["season"])

fig = px.bar(
    ann_df, x="season", y="value",
    title="NBA Player Age Distribution Over Time", color="variable", facet_row="variable",
    # range_y=[24, 32],
    template="plotly_white", color_discrete_sequence=px.colors.qualitative.Safe,
    category_orders={"season": list(np.sort(tmp_df.season.unique()))},
    labels=labels_dict,
    height=600, width=1200
)
# fig.update_traces(mode="markers+lines", marker={"line": {"width": 0.4, "color": "black"}})
fig.show()
