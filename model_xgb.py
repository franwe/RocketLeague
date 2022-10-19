from tabnanny import verbose
from sklearn.model_selection import train_test_split
import xgboost as xgb
import numpy as np
from pathlib import Path
import pandas as pd
import copy
import wandb

DATA_DIR = Path.cwd().joinpath("data")
MODEL_DIR = Path.cwd().joinpath("models")


def get_accuracy(labels, predicted, correct_pred, total_pred):
    for label, prediction in zip(labels, predicted):
        if label == prediction:
            correct_pred[label.item()] += 1
        total_pred[label.item()] += 1
    accuracy = {f"accuracy_{k}": correct_pred[k] / n for k, n in total_pred.items()}
    return accuracy, correct_pred, total_pred


config = dict(
    use_label_encoder=False,
    eval_metric="mlogloss",
    dataset="RocketLeague",
    architecture="XGBClassifier",
    training_batches=5,
)

if __name__ == "__main__":
    FEATURES = ["ball_pos_x", "ball_pos_y", "ball_pos_z", "ball_vel_x", "ball_vel_y", "ball_vel_z"]
    TARGET = "team_scoring_within_10sec"
    subsets = [f"train_{i}" for i in range(10)]
    subsets = [f"train_{i}" for i in range(2)]

    wandb.init(project="xgb-test", config=config, mode="online")
    # access all HPs through wandb.config, so logging matches execution!
    config = wandb.config

    X_test, y_test = np.zeros((0, len(FEATURES))), np.zeros(0)
    models = {}
    example_ct = 0
    for i, subset in enumerate(subsets):
        # Load Data and split
        print(f"\nRead File {subset}")
        df = pd.read_pickle(DATA_DIR.joinpath("processed", f"{subset}_train.pkl"))
        X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
            df[FEATURES].values, df[TARGET].values, test_size=0.1, random_state=42
        )
        X_test = np.vstack((X_test, X_test_i))
        y_test = np.hstack((y_test, y_test_i))

        xg_train_i = xgb.DMatrix(X_train_i, label=y_train_i)
        xg_test_i = xgb.DMatrix(X_test_i, label=y_test_i)

        params = {"disable_default_eval_metric": 1}
        print("\t...train...")
        if i == 0:
            model = xgb.XGBClassifier(
                use_label_encoder=config.use_label_encoder,
                eval_metric=config.eval_metric,
            )
            model.fit(X_train_i, y_train_i, verbose=True)
        else:
            model = model.fit(X_train_i, y_train_i, xgb_model=model.get_booster())
        example_ct += len(X_train_i)

        y_pred_i = model.predict(X_train_i)
        labels_list = [0, 1, 2]
        total_pred = {i: 1 for i in labels_list}
        correct_pred = {i: 0 for i in labels_list}
        accuracy, correct_pred, total_pred = get_accuracy(y_train_i, y_pred_i, correct_pred, total_pred)
        wandb.log(accuracy, step=example_ct)
        print(f"\taccuracy: {accuracy}")

        model.save_model(MODEL_DIR.joinpath(f"xgb_{subset}.model"))
        models[subset] = copy.deepcopy(model)


# Evaluate Models with Test Set
df = pd.read_pickle(DATA_DIR.joinpath("processed", f"all_test.pkl"))
X_test = df[FEATURES].values
y_test = df[TARGET].values
# xg_test = xgb.DMatrix(X_test, label=y_test)

print("Test accuracy developments:")
for name, model in models.items():
    y_pred = model.predict(X_test)
    labels_list = [0, 1, 2]
    total_pred = {i: 1 for i in labels_list}
    correct_pred = {i: 0 for i in labels_list}
    accuracy, correct_pred, total_pred = get_accuracy(y_test, y_pred, correct_pred, total_pred)
    print(f"\t{name} accuracy: {accuracy}")

print("Done.")
wandb.finish()
