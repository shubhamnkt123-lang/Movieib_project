import ast
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="MovieIQ", page_icon="🎬", layout="wide")
st.title("🎬 MovieIQ — Predictive Analytics on Film Success")
st.caption("Analyse movie performance and predict whether a movie is likely to be successful.")

@st.cache_data
def load_data():
    df = pd.read_csv("movies.csv")
    for c in ["budget","revenue","popularity","runtime","vote_average"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.drop_duplicates()
    df = df[(df["budget"] > 0) & (df["revenue"] > 0)]
    df = df.dropna(subset=["budget","revenue","popularity","runtime","vote_average"]).copy()
    df["success"] = (df["revenue"] > df["budget"]).astype(int)
    df["genres"] = df["genres"].fillna("").astype(str)
    return df

def parse_genres(x):
    # `genres` is a stringified list of TMDB dicts, e.g. "[{'id': 18, 'name': 'Drama'}]".
    # A comma/pipe split shreds this; ast.literal_eval parses it correctly.
    try:
        parsed = ast.literal_eval(x)
        names = [g["name"] for g in parsed]
        return names if names else ["Unknown"]
    except (ValueError, SyntaxError, TypeError):
        return ["Unknown"]

df = load_data()
df["genre_list"] = df["genres"].apply(parse_genres)

features = ["budget","popularity","runtime","vote_average"]
model = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
model.fit(df[features], df["success"])

st.sidebar.header("Filters")
genres = sorted({g for row in df["genre_list"] for g in row})
selected = st.sidebar.multiselect("Genre", genres)
min_vote = st.sidebar.slider("Minimum Vote Average", 0.0, 10.0, 0.0, 0.1)

filtered = df[df["vote_average"] >= min_vote].copy()
if selected:
    filtered = filtered[filtered["genre_list"].apply(lambda x: any(g in selected for g in x))]

c1,c2,c3,c4 = st.columns(4)
c1.metric("Movies", f"{len(filtered):,}")
c2.metric("Success Rate", f"{filtered.success.mean()*100:.1f}%" if len(filtered) else "N/A")
c3.metric("Avg Popularity", f"{filtered.popularity.mean():.2f}" if len(filtered) else "N/A")
c4.metric("Avg Vote", f"{filtered.vote_average.mean():.2f}" if len(filtered) else "N/A")

st.header("Exploratory Analysis")
if len(filtered):
    fig, ax = plt.subplots(figsize=(8,5))
    sns.scatterplot(data=filtered, x="budget", y="revenue", hue="success", alpha=.6, ax=ax)
    ax.set_title("Budget vs Revenue")
    st.pyplot(fig)
    plt.close(fig)

    genre_df = filtered.explode("genre_list").rename(columns={"genre_list":"genre"})
    counts = genre_df.genre.value_counts().head(10).sort_values()
    fig, ax = plt.subplots(figsize=(8,5))
    counts.plot(kind="barh", ax=ax)
    ax.set_title("Top Genres")
    ax.set_xlabel("Movie Count")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Statistical Test")
    s = filtered.loc[filtered.success==1, "popularity"]
    u = filtered.loc[filtered.success==0, "popularity"]
    if len(s)>1 and len(u)>1:
        t,p = stats.ttest_ind(s,u,equal_var=False,nan_policy="omit")
        st.write(f"Popularity T-test: t = {t:.4f}, p = {p:.6g}")

st.header("Movie Success Predictor")
col1,col2 = st.columns(2)
with col1:
    budget = st.number_input("Budget", min_value=1.0, value=10_000_000.0, step=1_000_000.0)
    popularity = st.number_input("Popularity", min_value=0.0, value=10.0, step=.5)
with col2:
    runtime = st.number_input("Runtime (minutes)", min_value=1.0, value=120.0, step=5.0)
    vote_average = st.number_input("Vote Average", min_value=0.0, max_value=10.0, value=7.0, step=.1)

if st.button("Predict Success"):
    x = pd.DataFrame([[budget,popularity,runtime,vote_average]], columns=features)
    prediction = int(model.predict(x)[0])
    probability = float(model.predict_proba(x)[0,1])
    if prediction:
        st.success(f"Predicted SUCCESS — success probability: {probability:.1%}")
    else:
        st.warning(f"Predicted NOT SUCCESS — success probability: {probability:.1%}")

st.caption("Decision-support model only; predictions are not guarantees of future film performance.")
