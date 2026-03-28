import pandas as pd
import streamlit as st
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Crime AI Dashboard", layout="wide")

st.title("🚔 AI Crime Intelligence System")
st.markdown("### 🌍 Advanced Crime Analytics • ML Prediction • Interactive Dashboard")
st.markdown("---")

# ---------------- FILE UPLOAD ----------------
st.sidebar.header("📁 Upload Dataset")

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Custom dataset loaded ✅")
else:
    df = pd.read_csv("USArrests.csv")

# ---------------- BASIC VALIDATION ----------------
required_cols = ["Murder", "Assault", "Rape", "UrbanPop"]

if not all(col in df.columns for col in required_cols):
    st.error("Dataset must contain: Murder, Assault, Rape, UrbanPop")
    st.stop()

# ---------------- FAKE GEO (FOR MAP) ----------------
np.random.seed(42)
df["lat"] = np.random.uniform(25, 50, size=len(df))
df["lon"] = np.random.uniform(-120, -70, size=len(df))

# ---------------- SIDEBAR ----------------
st.sidebar.header("🔍 Filters")

if "State" in df.columns:
    state = st.sidebar.selectbox("Select State", df["State"])
    filtered = df[df["State"] == state]
else:
    filtered = df

crime_type = st.sidebar.selectbox("Crime Type", ["Murder", "Assault", "Rape"])

# ---------------- KPI ----------------
st.subheader("📊 Key Insights")

col1, col2, col3 = st.columns(3)

col1.metric("Avg Murder", round(df["Murder"].mean(), 2))
col2.metric("Avg Assault", round(df["Assault"].mean(), 2))
col3.metric("Avg Rape", round(df["Rape"].mean(), 2))

# ---------------- DATA ----------------
st.subheader("📋 Data Preview")
st.dataframe(filtered.head())

# ---------------- BAR CHART ----------------
st.subheader("📊 Crime Comparison")

crimes = ["Murder", "Assault", "Rape"]
values = filtered[crimes].iloc[0].values

chart_data = pd.DataFrame({
    "Crime": crimes,
    "Value": values
})

st.bar_chart(chart_data.set_index("Crime"))

# ---------------- HEATMAP ----------------
st.subheader("🔥 Crime Heatmap")

df["intensity"] = df["Murder"] + df["Assault"] + df["Rape"]

map_df = df.rename(columns={"lat": "latitude", "lon": "longitude"})
st.map(map_df)

# ---------------- ML MODELS ----------------
st.subheader("🤖 Machine Learning Models")

X = df[["Assault", "UrbanPop", "Rape"]]
y = df["Murder"]

lr = LinearRegression()
dt = DecisionTreeRegressor()

lr.fit(X, y)
dt.fit(X, y)

# Model accuracy
lr_score = r2_score(y, lr.predict(X))
dt_score = r2_score(y, dt.predict(X))

col1, col2 = st.columns(2)

col1.metric("Linear Regression R²", round(lr_score, 2))
col2.metric("Decision Tree R²", round(dt_score, 2))

# ---------------- USER INPUT ----------------
st.sidebar.header("🎯 Prediction Input")

assault = st.sidebar.slider("Assault", int(df["Assault"].min()), int(df["Assault"].max()))
urban = st.sidebar.slider("UrbanPop", int(df["UrbanPop"].min()), int(df["UrbanPop"].max()))
rape = st.sidebar.slider("Rape", int(df["Rape"].min()), int(df["Rape"].max()))

input_data = pd.DataFrame({
    "Assault": [assault],
    "UrbanPop": [urban],
    "Rape": [rape]
})

# Predictions
lr_pred = lr.predict(input_data)[0]
dt_pred = dt.predict(input_data)[0]

# ---------------- PREDICTIONS ----------------
st.subheader("🔮 Predictions")

col1, col2 = st.columns(2)

col1.metric("Linear Regression", round(lr_pred, 2))
col2.metric("Decision Tree", round(dt_pred, 2))

# ---------------- TOP STATES ----------------
if "State" in df.columns:
    st.subheader(f"🏆 Top 10 States by {crime_type}")
    top_states = df.sort_values(by=crime_type, ascending=False).head(10)
    st.bar_chart(top_states.set_index("State")[crime_type])

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("🚀 Built with Streamlit • Machine Learning • Data Analytics")