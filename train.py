import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss


df = pd.read_csv("training_data.csv")


FEATURES = [
    "elo_diff",
    "win_pct_diff",
    "recent_form_diff",
    "home_field"
]


# --------------------------------
# Evaluate on completed 2026 games
# --------------------------------

train_df = df[df["season"] <= 2025]
test_df = df[df["season"] == 2026]

X_train = train_df[FEATURES]
y_train = train_df["home_win"]

X_test = test_df[FEATURES]
y_test = test_df["home_win"]


evaluation_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(C=0.01)
)

evaluation_model.fit(X_train, y_train)

predictions = evaluation_model.predict(X_test)
probabilities = evaluation_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, predictions)
loss = log_loss(y_test, probabilities)


print("\n🏈 College Football Model Evaluation")
print("------------------------------------")

print(f"Training games: {len(train_df)}")
print("Training seasons: 2022-2025")

print(f"\n2026 completed test games: {len(test_df)}")

print(f"\nAccuracy: {accuracy:.2%}")
print(f"Log Loss: {loss:.4f}")

print("\nFeatures:")

for feature in FEATURES:
    print(f"- {feature}")


# --------------------------------
# Train production model
# --------------------------------

# IMPORTANT: Do not include 2026.
# 2026 remains unseen evaluation data.

X = train_df[FEATURES]
y = train_df["home_win"]


model = make_pipeline(
    StandardScaler(),
    LogisticRegression(C=0.01)
)

model.fit(X, y)

joblib.dump(model, "football_model.pkl")


print("\n------------------------------------")
print("Final model trained on 2022-2025")
print("2026 reserved for ongoing evaluation")
print("Final model saved to football_model.pkl")