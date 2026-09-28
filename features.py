from collections import defaultdict, deque
import pandas as pd
from fetch_data import get_games

RECENT_GAMES = 7

# Take raw games from the API and turns them into ML-ready rows
def create_features(games):
    games = sorted(games, key=lambda g: g["startDate"])

    # Track each team's record before every game
    team_records = defaultdict(
        lambda: {"wins": 0, "games": 0}
    )

    # Track each team's recent results
    recent_results = defaultdict(
        lambda: deque(maxlen=RECENT_GAMES)
    )

    rows = []

    # Group by kickoff time to avoid simultaneous-game leakage
    games_by_date = defaultdict(list)

    for game in games:
        games_by_date[game["startDate"]].append(game)

    for start_date in sorted(games_by_date):
        current_games = games_by_date[start_date]
        completed_games = []

        for game in current_games:
            if not game.get("completed"):
                continue

            home = game["homeTeam"]
            away = game["awayTeam"]

            home_points = game.get("homePoints")
            away_points = game.get("awayPoints")

            home_elo = game.get("homePregameElo")
            away_elo = game.get("awayPregameElo")

            # Skip games with missing data or ties
            if home_points is None or away_points is None or home_elo is None or away_elo is None or home_points == away_points:
                continue

            home_record = team_records[home]
            away_record = team_records[away]

            # Neutral prior of 50% for teams without games
            if home_record["games"] > 0:
                home_win_pct = home_record["wins"] / home_record["games"]
            else:
                home_win_pct = 0.5

            if away_record["games"] > 0:
                away_win_pct = away_record["wins"] / away_record["games"]
            else:
                away_win_pct = 0.5

            # Recent form
            if len(recent_results[home]) > 0:
                home_recent_win_pct = (
                    sum(recent_results[home])/ len(recent_results[home]))
            else:
                home_recent_win_pct = 0.5

            if len(recent_results[away]) > 0:
                away_recent_win_pct = (sum(recent_results[away])/ len(recent_results[away]))
            else:
                away_recent_win_pct = 0.5

            home_won = int(home_points > away_points)

            rows.append({
                "season": game["season"],
                "date": game["startDate"],
                "home_team": home,
                "away_team": away,
                "elo_diff": home_elo - away_elo,
                "win_pct_diff": home_win_pct - away_win_pct,
                "recent_form_diff": home_recent_win_pct - away_recent_win_pct,
                "home_field": int(not game.get("neutralSite", False)),
                "home_win": home_won
            })

            completed_games.append((home, away, home_won))

        # Update records only AFTER generating features
        for home, away, home_won in completed_games:
            team_records[home]["games"] += 1
            team_records[away]["games"] += 1

            team_records[home]["wins"] += home_won
            team_records[away]["wins"] += 1 - home_won

            recent_results[home].append(home_won)
            recent_results[away].append(1 - home_won)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    all_games = []

    for year in range(2022, 2026):
        print(f"Fetching {year} games...")
        all_games.extend(get_games(year))

    # Calculate features for each season
    datasets = []

    for year in range(2022, 2026):
        season_games = [
            g for g in all_games
            if g["season"] == year
        ]

        datasets.append(create_features(season_games))

    df = pd.concat(datasets, ignore_index=True)

    df.to_csv("training_data.csv", index=False)

    print("\nDataset created!")
    print(f"Recent form window: {RECENT_GAMES} games")
    print(f"Total games: {len(df)}")

    print(df.head().to_string(index=False))

    print("\nMissing values:")
    print(df.isnull().sum())