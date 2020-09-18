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
from sklearn import linear_model
from sklearn import svm
from sklearn import metrics

desired_width = 320
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', desired_width)

df = pd.read_csv("data/predict_stat.csv", index_col=0).reset_index(drop=True)
st.title("Simple Linear Regression")

# ========================================
# Prelim data viz
# ========================================
st.header("Data exploration")

# Plot minutes played
hist_fig = px.histogram(df, x="mp", nbins=30, title="Histogram of minutes played", template="plotly_white")
st.write(hist_fig)

# Filter out small samples
df = df[df["mp"] > 500].reset_index(drop=True)

st.subheader("Correlations")
corr_x = st.selectbox("Correlation - X variable", options=df.columns, index=df.columns.get_loc("pts_per_g"))
corr_y = st.selectbox("Correlation - Y variable", options=["bpm", "per"], index=0)
corr_col = st.radio("Correlation - color variable", options=["age", "season"], index=1)
fig = px.scatter(df, x=corr_x, y=corr_y, title=f"Correlation between {corr_x} & {corr_y}",
                 template="plotly_white", render_mode='webgl',
                 color=corr_col, hover_data=['name', 'pos', 'age', 'season'], color_continuous_scale=px.colors.sequential.OrRd)
fig.update_traces(mode="markers", marker={"line": {"width": 0.4, "color": "slategrey"}})
st.write(fig)

# ========================================
# Preprocessing
# ========================================
# Only keep relevant features
cont_var_cols = ['g', 'mp_per_g', 'fg_per_g', 'fga_per_g', 'fg3_per_g', 'fg3a_per_g', 'fg2_per_g', 'fg2a_per_g', 'efg_pct', 'ft_per_g', 'fta_per_g',
                 'orb_per_g', 'drb_per_g', 'trb_per_g', 'ast_per_g', 'stl_per_g', 'blk_per_g', 'tov_per_g', 'pf_per_g', 'pts_per_g', 'mp']
cont_df = df[cont_var_cols]

# Plot mean & st. dev of various features
st.header("Feature scaling")
st.subheader("What does the current scale look like?")
feat_desc = cont_df.describe()[cont_var_cols].transpose().reset_index().rename({'index': "var"}, axis=1)
feat_fig = px.bar(feat_desc[['var', 'mean', 'std']].melt(id_vars=['var']), x="var", y="value", color="variable", barmode="group", template="plotly_white",
                  labels={"var": "Variable", "value": "Value", "variable": "Statistic"}, color_discrete_sequence=px.colors.qualitative.Safe)
st.write(feat_fig)
st.subheader("Now in log scale")
feat_fig = px.bar(feat_desc[['var', 'mean', 'std']].melt(id_vars=['var']), x="var", y="value", color="variable", barmode="group", template="plotly_white",
                  labels={"var": "Variable", "value": "Value", "variable": "Statistic"}, color_discrete_sequence=px.colors.qualitative.Safe, log_y=True)
st.write(feat_fig)
st.write("Wow, that's quite a significant discrepancy - let's scale these to a mean of zero and a standard deviation of 1")

# Scale the features
scaler = preprocessing.StandardScaler().fit(cont_df)
X = scaler.transform(cont_df)
# Prove that mean = 0, st deviation = 1
feat_desc = pd.DataFrame(X).describe().transpose().reset_index().rename({'index': "var"}, axis=1)
feat_fig = px.bar(feat_desc[['var', 'mean', 'std']].melt(id_vars=['var']), x="var", y="value", color="variable", barmode="group", template="plotly_white",
                  labels={"var": "Variable", "value": "Value", "variable": "Statistic"}, color_discrete_sequence=px.colors.qualitative.Safe)
st.write(feat_fig)

# Split date into train/test set
X_train, X_test = model_selection.train_test_split(X, train_size=0.8, random_state=42, shuffle=True)

y_stat = st.selectbox("Select Y value to predict:", ["bpm", "per"], index=0)
Y = df[y_stat].values
Y_train, Y_test = model_selection.train_test_split(Y, train_size=0.8, random_state=42, shuffle=True)

# ========================================
# Build models
# ========================================

mdl_names = {
    "Stochastic Gradient Descent": "sdg", "Ridge Regression": "ridge",
    "Support Vector Regression": "svr",
}
reg_name = st.selectbox("Choose regressor model", list(mdl_names.keys()), index=0)
reg = mdl_names[reg_name]

if reg == 'sdg':
    mdl = linear_model.SGDRegressor(loss="squared_loss", penalty="l2", max_iter=1000)
    mdl.fit(X_train, Y_train)
elif reg == 'ridge':
    mdl = linear_model.Ridge(alpha=.5)
    mdl.fit(X_train, Y_train)
elif reg == 'svr':
    mdl = svm.SVR(kernel='rbf', degree=3)
    mdl.fit(X_train, Y_train)

# Test prediction
Y_test_hat = mdl.predict(X_test)
test_out = pd.DataFrame([Y_test_hat, Y_test], index=["Prediction", "Actual"]).transpose()
_, df_test = model_selection.train_test_split(df, train_size=0.8, random_state=42, shuffle=True)
test_out = test_out.assign(player=df_test["name"].values)
test_out = test_out.assign(season=df_test["season"].values)
val_fig = px.scatter(test_out, x="Prediction", y="Actual", title=f"{reg_name} model prediction of {y_stat.upper()} vs ground truths", template="plotly_white",
                     color_discrete_sequence=px.colors.qualitative.Safe, hover_data=["player", "season"]
                     )
st.write(val_fig)

st.header("Evaluations")
st.subheader("Errors")
mse = metrics.mean_squared_error(Y_test, Y_test_hat)
st.write(f"Mean square error with {reg_name}: {round(mse, 2)}")

# ========== CAN WE USE THIS TO SEE WHO HAS PRODUCED EXTRA VALUE ==========
st.header("Basketball learnings")
# Who are the outliers?
Y_hat = mdl.predict(X)
out_df = pd.DataFrame([Y_hat, Y], index=["Prediction", "Actual"]).transpose()
out_df = out_df.assign(player=df["name"].values)
out_df = out_df.assign(season=df["season"].values)
out_df = out_df.assign(pos=df["pos_simple"].values)
out_df = out_df.assign(rel_value=out_df["Actual"]-out_df["Prediction"])
out_df.sort_values("Actual", inplace=True, ascending=False)
out_df = out_df[:200]
out_df.sort_values("rel_value", inplace=True, ascending=True)

era_yrs = [out_df["season"].quantile(qnt) for qnt in [0, 0.33, 0.67, 1]]
# out_df.assign(era=f'{int(out_df["season"].min())}-{int(era_yrs[0])}')

for i in range(len(era_yrs)-1):
    out_df.loc[(out_df.season >= era_yrs[i]), "era"] = f"{int(era_yrs[i])}-{int(era_yrs[i+1])}"

# By correlations
rel_fig = px.scatter(out_df, x="Actual", y="rel_value", title=f"Top seasons by players and their relative {y_stat.upper()}",
                     color="rel_value", color_continuous_scale=px.colors.sequential.OrRd, facet_col="pos", facet_row="era",
                     category_orders={"pos": ["PG", "SG", "SF", "PF", "C"], "era": list(np.sort(out_df.era.unique()))},
                     template="plotly_white", hover_data=["player", "season"],
                     labels={"rel_value": "vs Prediction", "Actual": "Performance"}, height=600
                     )
rel_fig.update_traces(mode="markers", marker={"line": {"width": 0.4, "color": "slategrey"}})
st.write(rel_fig)

# Who has the highest values:
st.subheader("Players who most overperformed against the model")
st.table(out_df.sort_values("rel_value", ascending=False)[:10])

st.subheader("Players who most underperformed against the model")
st.table(out_df.sort_values("rel_value", ascending=True)[:10])
