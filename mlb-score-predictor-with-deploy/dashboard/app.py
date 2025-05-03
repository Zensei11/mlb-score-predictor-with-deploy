import streamlit as st
from pipeline.forecast import simulate, load_trace
from pipeline.preprocess import preprocess
import pandas as pd
import matplotlib.pyplot as plt

st.title("MLB Score Forecast")

df = pd.read_csv("data/games.csv")
df, team_idx = preprocess(df)
trace = load_trace()

teamA = st.selectbox("Team A", team_idx.keys())
teamB = st.selectbox("Team B", team_idx.keys())
home_team = st.radio("Who is the home team?", [teamA, teamB])

if st.button("Simulate Game"):
    sA, sB = simulate(trace, team_idx, teamA, teamB, home_team)

    fig, ax = plt.subplots()
    ax.hist2d(sA, sB, bins=(range(0,11), range(0,11)), cmap='Blues')
    ax.set_xlabel(f"{teamB} Runs")
    ax.set_ylabel(f"{teamA} Runs")
    ax.set_title(f"Predicted Score Distribution: {teamA} vs {teamB}")
    st.pyplot(fig)

    win_pct = (sA > sB).mean()
    st.markdown(f"**{teamA} Win Probability: {win_pct:.1%}**")