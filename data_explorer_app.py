# ========== (c) JP Hwang 2/9/20  ==========

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
import streamlit as st
import plotly.express as px

df = pd.read_csv("data/player_per_game.csv", index_col=0).reset_index(drop=True)
df = df.assign(pos_simple=df.pos.apply(lambda x: x.split("-")[0]))

# Header
st.title("Data exploration demo app - NBA data")
st.write(f"Data from {df.season.min()} to {df.season.max()} seasons.")

st.subheader("Raw Dataset")
st.write(df)
st.write(df.describe())
# For univariate distributions
# histogram to better understand
st.header("Histogram")
hist_x = st.selectbox("Histogram variable", options=df.columns, index=df.columns.get_loc("mp_per_g"))
hist_bins = st.slider(label="Histogram bins", min_value=5, max_value=50, value=25, step=1)
hist_cats = df[hist_x].sort_values().unique()
hist_fig = px.histogram(df, x=hist_x, nbins=hist_bins, title="Histogram of " + hist_x,
                        template="plotly_white", category_orders={hist_x: hist_cats})
st.write(hist_fig)

# boxplots
st.header("Boxplot")
st.subheader("With a categorical variable - position, age or season")
box_x = st.selectbox("Boxplot variable", options=df.columns, index=df.columns.get_loc("pts_per_g"))
box_cat = st.selectbox("Categorical variable", ["pos_simple", "age", "season"], 0)
st.write("Hint - try correlating 3 point attempts per game ('fg3a_per_g') with season, or points per game ('pts_per_g') with age.")
box_fig = px.box(df, x=box_cat, y=box_x, title="Box plot of " + box_x,
                        template="plotly_white", category_orders={"pos_simple": ["PG", "SG", "SF", "PF", "C"]})
st.write(box_fig)

# min filter
st.header("Correlations")
corr_x = st.selectbox("Correlation - X variable", options=df.columns, index=df.columns.get_loc("fg3a_per_g"))
corr_y = st.selectbox("Correlation - Y variable", options=df.columns, index=df.columns.get_loc("efg_pct"))
corr_col = st.radio("Correlation - color variable", options=["age", "season", "pos_simple"], index=1)
corr_filt = st.selectbox("Filter variable", options=df.columns, index=df.columns.get_loc("fg3a_per_g"))
min_filt = st.number_input("Minimum value", value=6, min_value=0)
tmp_df = df[df[corr_filt] > min_filt]
fig = px.scatter(tmp_df, x=corr_x, y=corr_y, template="plotly_white", render_mode='webgl',
                 color=corr_col, hover_data=['name', 'pos', 'age', 'season'], color_continuous_scale=px.colors.sequential.OrRd,
                 category_orders={"pos_simple": ["PG", "SG", "SF", "PF", "C"]})
fig.update_traces(mode="markers", marker={"line": {"width": 0.4, "color": "slategrey"}})
st.subheader("Filtered scatterplot and dataframe")
st.write(fig)
st.write(tmp_df)

# correlation heatmap
hmap_params = st.multiselect("Select parameters to include on heatmap", options=list(df.columns), default=[p for p in df.columns if "fg" in p])
hmap_fig = px.imshow(df[hmap_params].corr())
st.write(hmap_fig)
