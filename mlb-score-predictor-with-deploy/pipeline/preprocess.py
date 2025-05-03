import pandas as pd

def preprocess(df):
    teams = pd.unique(df[['team1', 'team2']].values.ravel())
    team_idx = {team: i for i, team in enumerate(sorted(teams))}
    df['team1_id'] = df['team1'].map(team_idx)
    df['team2_id'] = df['team2'].map(team_idx)
    df['home'] = (df['home_team'] == df['team1']).astype(int)
    return df, team_idx