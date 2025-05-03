from statsapi import schedule, boxscore
import pandas as pd
from datetime import date, timedelta

def fetch_recent_games(days=7):
    start = (date.today() - timedelta(days=days)).isoformat()
    end = date.today().isoformat()
    games = schedule(start_date=start, end_date=end, sportId=1)

    rows = []
    for g in games:
        if g['status'] == 'Final':
            box = boxscore(g['game_id'])
            try:
                home = g['home_name']
                away = g['away_name']
                home_runs = box['teams']['home']['teamStats']['batting']['runs']
                away_runs = box['teams']['away']['teamStats']['batting']['runs']
                rows.append([home, away, home_runs, away_runs, home])
            except KeyError:
                continue

    df = pd.DataFrame(rows, columns=['team1', 'team2', 'team1_runs', 'team2_runs', 'home_team'])
    df.to_csv('data/games.csv', index=False)