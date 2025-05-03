from pipeline.fetch_data import fetch_recent_games
from pipeline.preprocess import preprocess
from pipeline.train_model import train_model
import pandas as pd

fetch_recent_games(days=60)
df = pd.read_csv("data/games.csv")
df, _ = preprocess(df)
train_model(df)