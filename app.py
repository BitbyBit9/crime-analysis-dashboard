import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression

# Page config
st.set_page_config(page_title="Crime Dashboard", layout="wide")

# Title
st.title("🚔 Smart Crime Analysis & Prediction System")

st.markdown("""
This dashboard analyzes crime data across states and predicts crime rates using Machine Learning.
""")

# Load dataset
df = pd.read_csv("USArrests.csv")

# Sidebar filters
st.sidebar.header("Filters")

state = st.sidebar.selectbox("Select State", df["State"])

crime_type = st.sidebar.selectbox(
    "Select Crime Type",
    ["Murder", "Assault", "Rape"]
)

# Filter data
filtered = df[df["State"] == state]

# Show selected state data
st.subheader(f"Crime Data for {state}")
st.dataframe(filtered)

# Show selected crime value
value = filtered[crime_type].values[0]
st.metric(label=f"{crime_type} Rate", value=value)

# Layout columns
col1, col2 = st.columns(2)

# Crime comparison (NO matplotlib)
with col1:
    st.subheader("Crime Comparison")

    crimes = ["Murder", "Assault", "Rape"]
    values = filtered[crimes].values.flatten()

    chart_data = pd.DataFrame({
        "Crime": crimes,
        "Value": values
    })

    st.bar_chart(chart_data.set_index("Crime"))

# Machine Learning Prediction
with col2:
    st.subheader("🔮 Prediction Result")

    # Prepare data
    X = df[["Assault", "UrbanPop", "Rape"]]
    y = df["Murder"]

    model = LinearRegression()
    model.fit(X, y)

    # Inputs
    st.sidebar.header("Prediction Input")

    assault = st.sidebar.slider("Assault", int(df["Assault"].min()), int(df["Assault"].max()))
    urban = st.sidebar.slider("Urban Population", int(df["UrbanPop"].min()), int(df["UrbanPop"].max()))
    rape = st.sidebar.slider("Rape", int(df["Rape"].min()), int(df["Rape"].max()))

    # Prediction input
    input_data = pd.DataFrame({
        "Assault": [assault],
        "UrbanPop": [urban],
        "Rape": [rape]
    })

    prediction = model.predict(input_data)

    st.metric("Predicted Murder Rate", round(prediction[0], 2))

# Top states analysis
st.subheader(f"Top 5 States by {crime_type}")

top_states = df.sort_values(by=crime_type, ascending=False).head(5)
st.dataframe(top_states[["State", crime_type]])