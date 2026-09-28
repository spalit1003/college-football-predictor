import os
import requests
from dotenv import load_dotenv


load_dotenv()

API_KEY = os.getenv("CFBD_API_KEY")
BASE_URL = "https://api.collegefootballdata.com"


def get_games(year, team=None):
    url = f"{BASE_URL}/games"

    headers = {
        "Authorization": f"Bearer {API_KEY}"
    }

    params = {
        "year": year,
        "seasonType": "regular",
        "classification": "fbs"
    }

    if team:
        params["team"] = team

    response = requests.get(
        url,
        headers=headers,
        params=params
    )

    response.raise_for_status()
    return response.json()


def get_team_game_stats(year, week):
    url = f"{BASE_URL}/games/teams"

    headers = {
        "Authorization": f"Bearer {API_KEY}"
    }

    params = {
        "year": year,
        "week": week,
        "seasonType": "regular",
        "classification": "fbs"
    }

    response = requests.get(
        url,
        headers=headers,
        params=params
    )

    response.raise_for_status()
    return response.json()


if __name__ == "__main__":
    stats = get_team_game_stats(
        2025,
        1
    )

    print(f"Game stats received: {len(stats)}")

    print("\nExample game stats:")
    print(stats[0])