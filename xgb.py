from sklearn.model_selection import train_test_split
import xgboost as xgb
import numpy as np
from pathlib import Path
import pandas as pd
import copy

from utils.metric import my_log_loss

DATA_DIR = Path.cwd().joinpath("data")
MODEL_DIR = Path.cwd().joinpath("models")


def xgb_my_log_loss(y_test_model: np.ndarray, y_pred_model: xgb.DMatrix):
    y_pred_model = y_pred_model.get_label()
    return "LogLoss", my_log_loss(y_test_model, y_pred_model)


if __name__ == "__main__":
    FEATURES = ["ball_pos_x", "ball_pos_y", "ball_pos_z", "ball_vel_x", "ball_vel_y", "ball_vel_z"]
    TARGET = "team_scoring_within_10sec"
    subsets = [f"train_{i}" for i in range(10)]
    subsets = [f"train_{i}" for i in range(5)]

    X_test, y_test = np.zeros((0, len(FEATURES))), np.zeros(0)
    prev_subset = subsets[0]
    models = {}
    for i, subset in enumerate(subsets):
        # Load Data and split
        print(f"\nRead File {subset}")
        df = pd.read_pickle(DATA_DIR.joinpath("processed", f"{subset}_final.pkl"))
        X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
            df[FEATURES].values, df[TARGET].values, test_size=0.1, random_state=42
        )
        X_test = np.vstack((X_test, X_test_i))
        y_test = np.hstack((y_test, y_test_i))

        xg_train_i = xgb.DMatrix(X_train_i, label=y_train_i)
        xg_test_i = xgb.DMatrix(X_test_i, label=y_test_i)

        params = {"disable_default_eval_metric": 1}
        if i == 0:
            model = xgb.train(params, xg_train_i, 30, custom_metric=xgb_my_log_loss)
        else:
            model = xgb.train(
                params,
                xg_train_i,
                30,
                custom_metric=xgb_my_log_loss,
                xgb_model=MODEL_DIR.joinpath(f"{prev_subset}.model"),
            )

        model.save_model(MODEL_DIR.joinpath(f"{subset}.model"))
        models[subset] = copy.deepcopy(model)
        prev_subset = subset

# Evaluate Models with Test Set
xg_test = xgb.DMatrix(X_test, label=y_test)

for name, model in models.items():
    y_pred = model.predict(xg_test)
    print(f"{name}: {my_log_loss(xg_test.get_label(), y_pred)}")

print("Done.")
