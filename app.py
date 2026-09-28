import streamlit as st
from predict import predict_upcoming_game
import pandas as pd


st.set_page_config(
    page_title="College Football Predictor",
    page_icon="🏈",
    layout="centered"
)

st.title("🏈 College Football Predictor")
st.write(
    "Predict the outcome of an upcoming college football game using team Elo, "
    "season performance, recent form, and home-field advantage."
)
st.divider()

team_1 = st.text_input("Team 1", placeholder="e.g. Oregon")
team_2 = st.text_input("Team 2", placeholder="e.g. Washington")


if st.button("Predict Winner", type="primary", use_container_width=True):

    if not team_1 or not team_2:
        st.warning("Please enter both teams.")

    elif team_1.strip().lower() == team_2.strip().lower():
        st.warning("Please enter two different teams.")

    else:
        try:
            with st.spinner("Analyzing matchup..."):
                result = predict_upcoming_game(team_1.strip(), team_2.strip())

            st.divider()

            st.subheader(f"{result['away_team']} @ {result['home_team']}")
            game_date = pd.to_datetime(result["date"])
            st.caption(f"Game date: {game_date.strftime('%B %d, %Y • %I:%M %p')}")

            if result["neutral"]:
                st.caption("Neutral-site game")

            home_col, away_col = st.columns(2)

            with home_col:
                st.metric(
                    result["home_team"],
                    f"{result['home_probability']:.1%}"
                )
                st.progress(float(result["home_probability"]))

            with away_col:
                st.metric(
                    result["away_team"],
                    f"{result['away_probability']:.1%}"
                )
                st.progress(float(result["away_probability"]))

            st.success(f"🏆 Predicted winner: **{result['winner']}**")

        except ValueError as error:
            st.error(str(error))

        except Exception as error:
            st.error("Something went wrong while generating the prediction.")
            st.exception(error)


st.divider()
st.caption("Trained on 2022–2025 games")