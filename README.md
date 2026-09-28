# 🏈 College Football Predictor

A machine learning app that predicts the winner of college football games.

The model uses pre-game Elo ratings, season win percentage, recent form, and home-field advantage to estimate each team's win probability.

The final Logistic Regression model achieved **74.88% accuracy** on 215 unseen 2026 games to date, after training on games from 2022–2025.

## Features

- Predict upcoming college football games
- Display win probabilities for both teams
- Identify the predicted winner
- Backtest predictions on completed games
- Streamlit web interface
- College Football Data API integration

## Tech Stack

Python, pandas, scikit-learn, Streamlit and the College Football Data API.

## Run Locally

Clone the repository:

```bash
git clone https://github.com/spalit1003/college-football-predictor.git
cd college-football-predictor
```

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root and add your College Football Data API key from https://collegefootballdata.com/key:

```text
CFBD_API_KEY=your_api_key_here
```

Train the model:

```bash
python train.py
```

Then launch the web app:

```bash
streamlit run app.py
```

## Model

The final model is a Logistic Regression classifier using:

- Elo rating difference
- Season win percentage difference
- Recent 7-game form difference
- Home-field advantage

Training data covers the **2022–2025 college football seasons**.

## Data

Game data is retrieved from the College Football Data API.
