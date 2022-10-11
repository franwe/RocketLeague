import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

DATA_DIR = Path.cwd().joinpath("data")


def reduce_mem_usage(df, verbose=True):
    start_mem = df.memory_usage().sum() / 1024**2
    numerics = ["int8", "int16", "int32", "int64", "float16", "float32", "float64"]

    for col in df.columns:
        if col == "team_scoring_next":
            continue
        col_type = df[col].dtypes
        limit = abs(df[col]).max()

        for tp in numerics:
            cond1 = str(col_type)[0] == tp[0]
            if tp[0] == "i":
                cond2 = limit <= np.iinfo(tp).max
            else:
                cond2 = limit <= np.finfo(tp).max

            if cond1 and cond2:
                df[col] = df[col].astype(tp)
                break

    end_mem = df.memory_usage().sum() / 1024**2

    reduction = (start_mem - end_mem) * 100 / start_mem
    if verbose:
        print(f"[INFO] Mem. usage decreased to {end_mem:.2f}" f" MB {reduction:.2f}% reduction.")
    return df


def load_df(filename):
    dtypes_df = pd.read_csv(DATA_DIR.joinpath("raw", "train_dtypes.csv"))
    dtypes = {k: v for (k, v) in zip(dtypes_df.column, dtypes_df.dtype)}
    df = pd.read_csv(DATA_DIR.joinpath("raw", filename), dtype=dtypes)
    df = reduce_mem_usage(df)
    return df


BALL_ZERO_MASK = (
    lambda df: (df["ball_pos_x"] == 0)
    & (df["ball_pos_y"] == 0)
    & (df["ball_vel_x"] == 0)
    & (df["ball_vel_y"] == 0)
    & (df["ball_vel_z"] == 0)
)


def preprocess_target(df):
    both_score = sum(df["team_A_scoring_within_10sec"] * df["team_B_scoring_within_10sec"])
    if both_score > 0:
        raise ValueError("Both Teams will score in next 10sec! Need to redesign target.")
    df["team_scoring_within_10sec"] = 0
    df.loc[df["team_A_scoring_within_10sec"] == 1, "team_scoring_within_10sec"] = 1
    df.loc[df["team_B_scoring_within_10sec"] == 1, "team_scoring_within_10sec"] = 2
    return df


def wrap_up_df(df, features, target):
    return df[features + [target]]


def pickle_final_df(df, filename):
    df.to_pickle(DATA_DIR.joinpath("processed", filename))


def csv_final_df(df, filename, index=False):
    df.to_csv(DATA_DIR.joinpath("processed", filename), index=index)


def split_train_test(df):
    train_i, test_i = train_test_split(df, test_size=0.1)
    train_i.reset_index(drop=True, inplace=True)
    test_i.reset_index(drop=True, inplace=True)
    return train_i, test_i


def get_overall_min_max(df, df_range):
    df_range["min"] = np.minimum(df_range["min"], df.min())
    df_range["max"] = np.maximum(df_range["max"], df.max())
    return df_range


if __name__ == "__main__":
    random_state = 1
    np.random.seed(random_state)
    FEATURES = ["ball_pos_x", "ball_pos_y", "ball_pos_z", "ball_vel_x", "ball_vel_y", "ball_vel_z"]
    TARGET = "team_scoring_within_10sec"
    subsets = [f"train_{i}" for i in range(10)]

    all_test = pd.DataFrame()
    df_range = pd.DataFrame({"min": 10.0**5, "max": -(10.0**5)}, index=FEATURES + [TARGET])

    for subset in subsets:
        df = load_df(filename=f"{subset}.csv")

        # Exploration
        shortest_game_id = df["game_num"].value_counts().index[-1]
        short_game = df[df["game_num"] == shortest_game_id]

        # Preprocessing
        df = preprocess_target(df)

        # Wrapup and Train-Test-Split
        df = wrap_up_df(df, FEATURES, TARGET)
        df_range = get_overall_min_max(df, df_range)
        train_i, test_i = split_train_test(df)
        all_test = pd.concat([all_test, test_i], ignore_index=True)

        print(f"Save file {subset}")
        pickle_final_df(train_i, filename=f"{subset}_train.pkl")
        csv_final_df(train_i, filename=f"{subset}_train.csv")

    pickle_final_df(all_test, filename=f"all_test.pkl")
    csv_final_df(all_test, filename=f"all_test.csv")
    csv_final_df(df_range, "all_ranges.csv", index=True)

    print("Done")
