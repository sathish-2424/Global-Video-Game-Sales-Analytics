# ======================================================
# 🎮 Video Game Revenue Analytics - Streamlit App
# ======================================================

import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# ------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------
st.set_page_config(
    page_title="Video Game Revenue Analytics",
    page_icon="🎮",
    layout="wide"
)

st.title("🎮 Video Game Revenue Analytics Dashboard")

# ------------------------------------------------------
# DATA LOADING & CLEANING (MATCHES YOUR DATASET)
# ------------------------------------------------------
@st.cache_data
def load_and_clean_data():
    df = pd.read_csv(
        "Game.csv",
        dtype={
            "Year": "int16",
            "Units Sold (M)": "float32",
            "Total Revenue (M USD)_USD_M": "float32"
        }
    )

    # Remove invalid rows
    df = df[df["Total Revenue (M USD)"] > 0].copy()

    # Rename columns for clarity
    df.rename(columns={
        "Name": "Game",
        "Units Sold (M)": "Units Sold (M)",
        "Total Revenue (M USD)": "Revenue (M USD)"
    }, inplace=True)

    # Optimize memory
    for col in ["Platform", "Genre", "Publisher", "Developer"]:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df


df = load_and_clean_data()

# ------------------------------------------------------
# KPI METRICS
# ------------------------------------------------------
st.subheader("📌 Key Business Metrics")

c1, c2, c3, c4 = st.columns(4)

c1.metric("🎮 Total Games", f"{len(df):,}")
c2.metric("💰 Total Revenue (M USD)", f"{df['Revenue (M USD)'].sum():,.0f}")
c3.metric("📊 Avg Revenue / Name", f"{df['Revenue (M USD)'].mean():.2f}")
c4.metric("📈 Avg Units Sold (M)", f"{df['Units Sold (M)'].mean():.2f}")

st.divider()

# ------------------------------------------------------
# AGGREGATIONS (CACHED)
# ------------------------------------------------------
@st.cache_data
def compute_aggregates(df):
    return {
        "year": df.groupby("Year")["Revenue (M USD)"]
                  .sum().reset_index().sort_values("Year"),

        "platform": df.groupby("Platform", observed=True)["Revenue (M USD)"]
                      .sum().reset_index().sort_values("Revenue (M USD)", ascending=False),

        "genre": df.groupby("Genre", observed=True)["Revenue (M USD)"]
                   .sum().reset_index().sort_values("Revenue (M USD)", ascending=False),

        "publisher": df.groupby("Publisher", observed=True)["Revenue (M USD)"]
                       .sum().reset_index().sort_values("Revenue (M USD)", ascending=False)
    }


aggs = compute_aggregates(df)

# ------------------------------------------------------
# VISUAL ANALYTICS
# ------------------------------------------------------
st.subheader("📊 Revenue Insights")

COLOR = "#4EA8DE"

# Year-wise Revenue
fig_year = px.line(
    aggs["year"],
    x="Year",
    y="Revenue (M USD)",
    markers=True,
    title="📈 Year-wise Revenue Trend"
)
fig_year.update_traces(line=dict(width=3))
fig_year.update_yaxes(tickformat=",")

# Platform Revenue
fig_platform = px.bar(
    aggs["platform"],
    x="Platform",
    y="Revenue (M USD)",
    title="🎮 Platform-wise Revenue",
    text_auto=".1f",
    color_discrete_sequence=[COLOR]
)

# Genre Revenue
fig_genre = px.bar(
    aggs["genre"],
    x="Revenue (M USD)",
    y="Genre",
    orientation="h",
    title="🧩 Genre-wise Revenue",
    text_auto=".1f",
    color_discrete_sequence=[COLOR]
)

c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(fig_platform, use_container_width=True)
with c2:
    st.plotly_chart(fig_genre, use_container_width=True)

st.plotly_chart(fig_year, use_container_width=True)

st.divider()

# ------------------------------------------------------
# ADVANCED ANALYTICS
# ------------------------------------------------------
st.subheader("🔍 Publisher & Game Analysis")

tab1, tab2 = st.tabs(["Top Publishers", "Top Games"])

with tab1:
    fig_pub = px.bar(
        aggs["publisher"].head(10),
        x="Revenue (M USD)",
        y="Publisher",
        orientation="h",
        title="🏆 Top 10 Publishers by Revenue",
        text_auto=".0f",
        color_discrete_sequence=[COLOR]
    )
    st.plotly_chart(fig_pub, use_container_width=True)

with tab2:
    top_games = (
        df.sort_values("Revenue (M USD)", ascending=False)
        .head(15)[["Game", "Platform", "Genre", "Revenue (M USD)", "Year"]]
    )
    st.dataframe(top_games, use_container_width=True, hide_index=True)

st.divider()

# ------------------------------------------------------
# MACHINE LEARNING (REVENUE PREDICTION)
# ------------------------------------------------------
@st.cache_resource
def train_model(df):
    data = df[["Platform", "Genre", "Revenue (M USD)"]].dropna()

    X = data[["Platform", "Genre"]]
    y = data["Revenue (M USD)"]

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         ["Platform", "Genre"])
    ])

    model = Pipeline([
        ("prep", preprocessor),
        ("rf", RandomForestRegressor(
            n_estimators=80,
            max_depth=12,
            random_state=42,
            n_jobs=-1
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)

    return model, model.score(X_train, y_train), model.score(X_test, y_test)


with st.spinner("Training revenue prediction model..."):
    model, train_score, test_score = train_model(df)

# ------------------------------------------------------
# PREDICTION UI
# ------------------------------------------------------
st.subheader("🎯 Revenue Prediction Simulator")

with st.form("predict"):
    c1, c2 = st.columns(2)
    platform = c1.selectbox("Select Platform", sorted(df["Platform"].unique()))
    genre = c2.selectbox("Select Genre", sorted(df["Genre"].unique()))
    submit = st.form_submit_button("🔮 Predict Revenue")

if submit:
    input_df = pd.DataFrame({"Platform": [platform], "Genre": [genre]})
    pred = model.predict(input_df)[0]

    confidence = "High" if test_score > 0.7 else "Medium" if test_score > 0.5 else "Low"

    st.success(f"💰 Predicted Revenue: **{pred:.2f} Million USD**")
    st.info(f"📊 Model Confidence: **{confidence}**")

st.divider()

# ------------------------------------------------------
# FOOTER
# ------------------------------------------------------
with st.expander("📋 Dataset Info"):
    c1, c2, c3 = st.columns(3)
    c1.write(f"**Records:** {len(df):,}")
    c2.write(f"**Years:** {df['Year'].min()} – {df['Year'].max()}")
    c3.write(f"**Platforms:** {df['Platform'].nunique()}")

st.markdown("---\n🎮 *Video Game Revenue Analytics Project*")
