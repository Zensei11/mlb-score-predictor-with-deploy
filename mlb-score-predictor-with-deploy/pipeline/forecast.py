import numpy as np
import pickle

def simulate(trace, team_idx, teamA, teamB, home_team="Yankees", n_sim=10000):
    a_id = team_idx[teamA]
    b_id = team_idx[teamB]

    attack = trace.posterior['attack'].stack(samples=("chain", "draw")).values
    defense = trace.posterior['defense'].stack(samples=("chain", "draw")).values
    home_adv = trace.posterior['home_adv'].stack(samples=("chain", "draw")).values

    if home_team == teamA:
        muA = np.exp(attack[:, a_id] - defense[:, b_id] + home_adv)
        muB = np.exp(attack[:, b_id] - defense[:, a_id])
    else:
        muA = np.exp(attack[:, a_id] - defense[:, b_id])
        muB = np.exp(attack[:, b_id] - defense[:, a_id] + home_adv)

    return np.random.poisson(muA, n_sim), np.random.poisson(muB, n_sim)

def load_trace():
    with open("models/trace_latest.pkl", "rb") as f:
        return pickle.load(f)