import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
import numpy as np

# Page config
st.set_page_config(page_title="Crime Dashboard", layout="wide")

# Title
st.title("🚔 Smart Crime Analysis & Prediction System")
st.markdown("## 📊 Interactive Crime Intelligence Dashboard")
st.markdown("---")

# Load dataset
df = pd.read_csv("USArrests.csv")

# Add coordinates for map visualization
np.random.seed(42)
df["lat"] = np.random.uniform(25, 50, size=len(df))
df["lon"] = np.random.uniform(-120, -70, size=len(df))

# Sidebar filters
st.sidebar.header("Filters")

state = st.sidebar.selectbox("Select State", df["State"])

crime_type = st.sidebar.selectbox(
    "Select Crime Type",
    ["Murder", "Assault", "Rape"]
)

# Filter data
filtered = df[df["State"] == state]

# KPI Section
st.subheader("📌 Key Insights")

col1, col2, col3 = st.columns(3)

col1.metric("Avg Murder Rate", round(df["Murder"].mean(), 2))
col2.metric("Avg Assault Rate", round(df["Assault"].mean(), 2))
col3.metric("Avg Rape Rate", round(df["Rape"].mean(), 2))

# Show selected state data
st.subheader(f"📍 Crime Data for {state}")
st.dataframe(filtered)

# Selected metric
value = filtered[crime_type].values[0]
st.metric(label=f"{crime_type} Rate in {state}", value=value)

# Layout
col1, col2 = st.columns(2)

# Chart (Streamlit native)
with col1:
    st.subheader("📊 Crime Comparison")

    crimes = ["Murder", "Assault", "Rape"]
    values = filtered[crimes].values.flatten()

    chart_data = pd.DataFrame({
        "Crime": crimes,
        "Value": values
    })

    st.bar_chart(chart_data.set_index("Crime"))

# Prediction
with col2:
    st.subheader("🔮 Prediction Result")

    X = df[["Assault", "UrbanPop", "Rape"]]
    y = df["Murder"]

    model = LinearRegression()
    model.fit(X, y)

    st.sidebar.header("Prediction Input")

    assault = st.sidebar.slider("Assault", int(df["Assault"].min()), int(df["Assault"].max()))
    urban = st.sidebar.slider("Urban Population", int(df["UrbanPop"].min()), int(df["UrbanPop"].max()))
    rape = st.sidebar.slider("Rape", int(df["Rape"].min()), int(df["Rape"].max()))

    input_data = pd.DataFrame({
        "Assault": [assault],
        "UrbanPop": [urban],
        "Rape": [rape]
    })

    prediction = model.predict(input_data)

    st.metric("Predicted Murder Rate", round(prediction[0], 2))

# Top states
st.subheader(f"📊 Top 10 States by {crime_type}")

top_states = df.sort_values(by=crime_type, ascending=False).head(10)
st.bar_chart(top_states.set_index("State")[crime_type])

# Map visualization
st.subheader("📍 Crime Map Visualization")

st.map(df[["lat", "lon"]])