# ========== (c) JP Hwang 6/9/20  ==========

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
import streamlit as st
from sklearn import model_selection
from sklearn import preprocessing

desired_width = 320
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', desired_width)

df = pd.read_csv("data/player_per_game.csv", index_col=0).reset_index(drop=True)

# Merge BPM data, create simple position,
df["mp"] = df["mp_per_g"] * df["g"]
df = df.assign(pos_simple=df.pos.apply(lambda x: x.split("-")[0]))
# TODO - Change position to one-hot vector, add columns to the data

# Plot minutes played
hist_fig = px.histogram(df, x="mp", nbins=30, title="Histogram of minutes played", template="plotly_white")
st.write(hist_fig)

# Filter out small samples
df = df[df["mp"] > 500].reset_index(drop=True)

# ===== Process categorical variables =====
# Convert positions to one-hot encoding
pos_enc = preprocessing.OneHotEncoder()
pos_enc.fit(df["pos_simple"].unique().reshape(-1, 1))
pos_oh = pos_enc.transform(df["pos_simple"].to_numpy().reshape(-1, 1))
pos_df = pd.DataFrame(pos_oh.toarray(), columns=pos_enc.get_feature_names(["POS"]))
cat_var_cols = list(pos_enc.get_feature_names(["POS"]))
X_cat = pos_df[cat_var_cols].values

# ===== Process continuous variables =====
# Only keep relevant features
cont_var_cols = ['g', 'mp_per_g', 'fg_per_g', 'fga_per_g', 'fg3_per_g', 'fg3a_per_g', 'fg2_per_g', 'fg2a_per_g', 'efg_pct', 'ft_per_g', 'fta_per_g',
                 'orb_per_g', 'drb_per_g', 'trb_per_g', 'ast_per_g', 'stl_per_g', 'blk_per_g', 'tov_per_g', 'pf_per_g', 'pts_per_g', 'mp']
cont_df = df[cont_var_cols]

# Plot mean & st. dev of various features
feat_desc = cont_df.describe()[cont_var_cols].transpose().reset_index().rename({'index': "var"}, axis=1)
feat_fig = px.bar(feat_desc[['var', 'mean', 'std']].melt(id_vars=['var']), x="var", y="value", color="variable", barmode="group", template="plotly_white",
                  labels={"var": "Variable", "value": "Value", "variable": "Statistic"}, color_discrete_sequence=px.colors.qualitative.Safe)
st.write(feat_fig)

# Scale the features
scaler = preprocessing.StandardScaler().fit(cont_df)
X_cont = scaler.transform(cont_df)
# Prove that mean = 0, st deviation = 1
feat_desc = pd.DataFrame(X_cont).describe().transpose().reset_index().rename({'index': "var"}, axis=1)
feat_fig = px.bar(feat_desc[['var', 'mean', 'std']].melt(id_vars=['var']), x="var", y="value", color="variable", barmode="group", template="plotly_white",
                  labels={"var": "Variable", "value": "Value", "variable": "Statistic"}, color_discrete_sequence=px.colors.qualitative.Safe)
st.write(feat_fig)

# ========== COMBINE BOTH ==========
X = np.concatenate([X_cont, X_cat], axis=1)

# Split date into train/test set
X_train, X_test = model_selection.train_test_split(X, train_size=0.8, random_state=42, shuffle=True)

