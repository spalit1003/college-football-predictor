import joblib
import pandas as pd
from fetch_data import get_games


SEASON = 2026
model = joblib.load("football_model.pkl")


def get_team_stats(team, before_date=None):
    games = get_games(SEASON, team)
    completed_games = []

    for game in games:
        if not game.get("completed"):
            continue

        # Only use games before the target game
        if before_date and game["startDate"] >= before_date:
            continue

        completed_games.append(game)

    completed_games.sort(key=lambda g: g["startDate"])

    wins = 0
    latest_elo = None
    results = []

    for game in completed_games:
        if game["homeTeam"] == team:
            team_points = game["homePoints"]
            opponent_points = game["awayPoints"]
            latest_elo = game.get("homePostgameElo")
        else:
            team_points = game["awayPoints"]
            opponent_points = game["homePoints"]
            latest_elo = game.get("awayPostgameElo")

        if team_points > opponent_points:
            wins += 1
            results.append(1)
        else:
            results.append(0)

    # Use neutral 50% if the team hasn't played yet
    win_pct = wins / len(completed_games) if completed_games else 0.5

    # Win percentage over the last 7 games
    recent_results = results[-7:]
    recent_win_pct = sum(recent_results) / len(recent_results) if recent_results else 0.5

    return {
        "team": team,
        "wins": wins,
        "games": len(completed_games),
        "win_pct": win_pct,
        "recent_win_pct": recent_win_pct,
        "elo": latest_elo
    }


def find_matchup(team_1, team_2):
    games = get_games(SEASON, team_1)
    matching_games = []

    for game in games:
        teams = {game["homeTeam"], game["awayTeam"]}

        if teams == {team_1, team_2}:
            matching_games.append(game)

    matching_games.sort(key=lambda g: g["startDate"], reverse=True)

    return matching_games


def predict_game(
    home_team,
    away_team,
    neutral=False,
    before_date=None,
    pregame_home_elo=None,
    pregame_away_elo=None
):
    home = get_team_stats(home_team, before_date)
    away = get_team_stats(away_team, before_date)

    if pregame_home_elo is not None:
        home["elo"] = pregame_home_elo

    if pregame_away_elo is not None:
        away["elo"] = pregame_away_elo

    if home["elo"] is None or away["elo"] is None:
        missing_team = home_team if home["elo"] is None else away_team
        raise ValueError(f"Elo rating unavailable for {missing_team}.")

    elo_diff = home["elo"] - away["elo"]
    win_pct_diff = home["win_pct"] - away["win_pct"]
    recent_form_diff = home["recent_win_pct"] - away["recent_win_pct"]
    home_field = 0 if neutral else 1

    features = pd.DataFrame([{
        "elo_diff": elo_diff,
        "win_pct_diff": win_pct_diff,
        "recent_form_diff": recent_form_diff,
        "home_field": home_field
    }])

    probabilities = model.predict_proba(features)[0]

    away_probability = probabilities[0]
    home_probability = probabilities[1]

    winner = home_team if home_probability > away_probability else away_team

    return {
        "home_team": home_team,
        "away_team": away_team,
        "home_probability": home_probability,
        "away_probability": away_probability,
        "winner": winner,
        "home_stats": home,
        "away_stats": away
    }


def predict_upcoming_game(team_1, team_2):
    matches = find_matchup(team_1, team_2)
    upcoming_matches = [game for game in matches if not game.get("completed")]

    if not upcoming_matches:
        raise ValueError("No upcoming matchup found between these teams.")

    upcoming_matches.sort(key=lambda g: g["startDate"])
    game = upcoming_matches[0]

    actual_home = game["homeTeam"]
    actual_away = game["awayTeam"]

    prediction = predict_game(
        home_team=actual_home,
        away_team=actual_away,
        neutral=game.get("neutralSite", False),
        before_date=game["startDate"],
        pregame_home_elo=game.get("homePregameElo"),
        pregame_away_elo=game.get("awayPregameElo")
    )

    prediction["date"] = game["startDate"]
    prediction["neutral"] = game.get("neutralSite", False)

    return prediction


def backtest_game(team_1, team_2):
    matches = find_matchup(team_1, team_2)
    completed_matches = [game for game in matches if game.get("completed")]

    if not completed_matches:
        raise ValueError("No completed matchup found between these teams.")

    game = completed_matches[0]

    actual_home = game["homeTeam"]
    actual_away = game["awayTeam"]

    prediction = predict_game(
        home_team=actual_home,
        away_team=actual_away,
        neutral=game.get("neutralSite", False),
        before_date=game["startDate"],
        pregame_home_elo=game.get("homePregameElo"),
        pregame_away_elo=game.get("awayPregameElo")
    )

    prediction["date"] = game["startDate"]
    prediction["home_points"] = game["homePoints"]
    prediction["away_points"] = game["awayPoints"]

    return prediction


# Terminal version
if __name__ == "__main__":
    print("\n🏈 College Football Predictor")
    print("\n1 - Predict upcoming game")
    print("2 - Backtest completed game")

    mode = input("\nChoose mode: ").strip()

    if mode == "1":
        team_1 = input("Team 1: ").strip()
        team_2 = input("Team 2: ").strip()

        try:
            result = predict_upcoming_game(team_1, team_2)

            print(f"\n{result['home_team']}: {result['home_probability']:.1%}")
            print(f"{result['away_team']}: {result['away_probability']:.1%}")
            print(f"\n🏆 Predicted winner: {result['winner']}")

        except ValueError as error:
            print(f"\n⚠️ {error}")

    elif mode == "2":
        team_1 = input("Team 1: ").strip()
        team_2 = input("Team 2: ").strip()

        try:
            result = backtest_game(team_1, team_2)

            print(
                f"\nActual result: "
                f"{result['home_team']} {result['home_points']} - "
                f"{result['away_team']} {result['away_points']}"
            )

            print(f"\n{result['home_team']}: {result['home_probability']:.1%}")
            print(f"{result['away_team']}: {result['away_probability']:.1%}")
            print(f"\n🏆 Predicted winner: {result['winner']}")

        except ValueError as error:
            print(f"\n⚠️ {error}")

    else:
        print("Invalid option!")