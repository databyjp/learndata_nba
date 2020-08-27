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

# Pre-processing -> aggregate players' stats if traded in-season
for yr in totals_df.season.unique():
    tmp_df = totals_df[totals_df.season == yr]
    dup_df = tmp_df[tmp_df.name.duplicated()]
    for name in dup_df.name.unique():
        player_df = dup_df[dup_df.name == name]
        agg_stat = player_df.sum(axis=0)
        agg_stat[["name", "pos", "age", "season"]] = player_df.iloc[0][["name", "pos", "age", "season"]]
        agg_stat["team_id"] = "NBA"
        totals_df = totals_df[(totals_df.name != name) | (totals_df.season != yr)]
        totals_df = totals_df.append(agg_stat, ignore_index=True)

# Find players with stats stretching across multiple years
# Add feature of players' stats from 1, 2, ... N years before to each row as applicable
# Target value to predict ->  "fg_pct_corner3"

