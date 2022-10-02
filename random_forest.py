from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer
import pandas as pd
import numpy as np
from pathlib import Path

from utils.metric import my_log_loss

DATA_DIR = Path.cwd().joinpath("data")

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
