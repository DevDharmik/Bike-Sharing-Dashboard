import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# ------------------ Page config ------------------
st.set_page_config(page_title="Bike Sharing Dashboard", layout="wide")

# ------------------ Load data ------------------
df = pd.read_csv("train.csv")
df["datetime"] = pd.to_datetime(df["datetime"])

# Feature engineering
df["year"] = df["datetime"].dt.year
df["month"] = df["datetime"].dt.month
df["hour"] = df["datetime"].dt.hour
df["day"] = df["datetime"].dt.day_name()

season_map = {1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"}
df["season"] = df["season"].map(season_map)

# ------------------ Title ------------------
st.title("🚲 Washington D.C. Bike Sharing Dashboard")
st.markdown("Interactive analysis of bike rental patterns (2011–2012)")

# ------------------ Sidebar filters ------------------
st.sidebar.header("Filters")

year_filter = st.sidebar.multiselect(
    "Select Year",
    options=df["year"].unique(),
    default=df["year"].unique()
)

season_filter = st.sidebar.multiselect(
    "Select Season",
    options=df["season"].unique(),
    default=df["season"].unique()
)

weather_filter = st.sidebar.multiselect(
    "Select Weather",
    options=df["weather"].unique(),
    default=df["weather"].unique()
)

filtered_df = df[
    (df["year"].isin(year_filter)) &
    (df["season"].isin(season_filter)) &
    (df["weather"].isin(weather_filter))
]

# ------------------ Layout ------------------
col1, col2 = st.columns(2)

# -------- Plot 1: Hourly rentals --------
with col1:
    st.subheader("Average Rentals by Hour")
    st.line_chart(filtered_df.groupby("hour")["count"].mean())

# -------- Plot 2: Daily rentals --------
with col2:
    st.subheader("Average Rentals by Day")
    st.bar_chart(filtered_df.groupby("day")["count"].mean())

# -------- Plot 3: Seasonal distribution --------
st.subheader("Bike Rentals by Season")
fig1, ax1 = plt.subplots()
sns.boxplot(x="season", y="count", data=filtered_df, ax=ax1)
st.pyplot(fig1)

# -------- Plot 4: Casual vs Registered --------
st.subheader("Casual vs Registered Users")
st.bar_chart(filtered_df[["casual", "registered"]].mean())

# -------- Plot 5: Weather impact --------
st.subheader("Weather Impact on Rentals")
st.bar_chart(filtered_df.groupby("weather")["count"].mean())

# -------- Plot 6: Heatmap --------
st.subheader("Heatmap: Rentals by Day & Hour")
heatmap = filtered_df.pivot_table(
    values="count",
    index="day",
    columns="hour",
    aggfunc="mean"
)
fig2 = px.imshow(
    heatmap,
    labels=dict(x="Hour", y="Day", color="Avg Rentals"),
    aspect="auto"
)
st.plotly_chart(fig2)

# ------------------ Insights ------------------
st.markdown("""
### 📌 Key Insights
- Bike demand peaks during morning and evening commute hours  
- Registered users account for most rentals  
- Summer shows the highest rental activity  
- Adverse weather conditions significantly reduce demand  
""")

