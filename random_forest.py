from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path.cwd().joinpath("data")


def format_predictions(y_model):
    """
    Back to original structure (team_A/B_scoring = 0/1 instead of team_socring=[-1, 1])
    undo: df["team_scoring_within_10sec"] = -df["team_A_scoring_within_10sec"] + df["team_B_scoring_within_10sec"]
    v < 0: team A scoring
    v > 0: team B scoring
    """
    y = pd.DataFrame(y_model)
    y.columns = ["p"]
    y.reset_index(inplace=True, drop=True)

    y["team_A_scoring_within_10sec"] = y["p"].apply(lambda v: abs(max(v, 0)))
    y["team_B_scoring_within_10sec"] = y["p"].apply(lambda v: min(v, 0))
    return y[["team_A_scoring_within_10sec", "team_B_scoring_within_10sec"]]


def my_log_loss(y_test_model, y_pred_model):
    clip = lambda p: max(min(p, 1 - 10 ** (-15)), 10 ** (-15))
    y_test = format_predictions(y_test_model)
    y_pred = format_predictions(y_pred_model)
    score = 0
    N = len(y_pred)
    for team in ["team_A_scoring_within_10sec", "team_B_scoring_within_10sec"]:
        y_test_t = y_test[team].apply(lambda p: clip(p))
        y_pred_t = y_pred[team].apply(lambda p: clip(p))
        team_score = 1 / N * sum(y_test_t * np.log(y_pred_t) + (1 - y_test_t) * np.log(1 - y_pred_t))
        score += team_score
    return -1 / 2 * score


if __name__ == "__main__":
    FEATURES = ["ball_pos_x", "ball_pos_y", "ball_pos_z", "ball_vel_x", "ball_vel_y", "ball_vel_z"]
    TARGET = "team_scoring_within_10sec"
    subset = "train_0"

    # Load Data and split
    print("\nRead File")
    df = pd.read_pickle(DATA_DIR.joinpath("processed", f"{subset}_final.pkl"))
    X_train, X_test, y_train, y_test = train_test_split(df[FEATURES], df[TARGET], test_size=0.33, random_state=42)

    # Model
    print("\nTrain Model...")
    score = make_scorer(my_log_loss, greater_is_better=False)
    model = RandomForestRegressor(n_estimators=4, oob_score=True, n_jobs=1, random_state=1)
    model.fit(X_train, y_train)
    # print("Log Loss:", score(model, X_test, y_test))

    # Predict
    print("\nPredict and create Benchmarks...")
    y_pred = model.predict(X_test)
    y_rand = np.random.rand(len(y_pred)) * 2 - 1
    y_zero = np.zeros(len(y_pred))

    # Metric
    print("\nCalculate Metric")
    print(f"Log Loss: pred {my_log_loss(y_test, y_pred)}")
    print(f"        : rand {my_log_loss(y_test, y_rand)}")
    print(f"        : zero {my_log_loss(y_test, y_zero)}")

    print("Done.")
