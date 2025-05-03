import pymc as pm
import arviz as az
import pandas as pd
import pickle

def train_model(df):
    n_teams = len(pd.unique(df[['team1_id', 'team2_id']].values.ravel()))

    with pm.Model() as model:
        mu_att = pm.Normal("mu_att", mu=0, sigma=1)
        mu_def = pm.Normal("mu_def", mu=0, sigma=1)
        attack = pm.Normal("attack", mu=mu_att, sigma=1, shape=n_teams)
        defense = pm.Normal("defense", mu=mu_def, sigma=1, shape=n_teams)
        home_adv = pm.Normal("home_adv", mu=0.1, sigma=0.5)

        team1_theta = pm.math.exp(
            attack[df['team1_id'].values] -
            defense[df['team2_id'].values] +
            home_adv * df['home'].values
        )
        team2_theta = pm.math.exp(
            attack[df['team2_id'].values] -
            defense[df['team1_id'].values] +
            home_adv * (1 - df['home'].values)
        )

        pm.Poisson("team1_runs", mu=team1_theta, observed=df['team1_runs'].values)
        pm.Poisson("team2_runs", mu=team2_theta, observed=df['team2_runs'].values)

        trace = pm.sample(1000, tune=1000, return_inferencedata=True)

    with open("models/trace_latest.pkl", "wb") as f:
        pickle.dump(trace, f)