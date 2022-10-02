import pandas as pd
import numpy as np
from pathlib import Path

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
    dtypes_df = pd.read_csv(DATA_DIR.joinpath("train_dtypes.csv"))
    dtypes = {k: v for (k, v) in zip(dtypes_df.column, dtypes_df.dtype)}
    df = pd.read_csv(DATA_DIR.joinpath(filename), dtype=dtypes)
    df = reduce_mem_usage(df)
    return df


BALL_ZERO_MASK = (
    lambda df: (df["ball_pos_x"] == 0)
    & (df["ball_pos_y"] == 0)
    & (df["ball_vel_x"] == 0)
    & (df["ball_vel_y"] == 0)
    & (df["ball_vel_z"] == 0)
)

if __name__ == "__main__":
    train0_df = load_df(filename="train_0.csv")
    print(train0_df.columns)

    shortest_game_id = train0_df["game_num"].value_counts().index[-1]
    short_game = train0_df[train0_df["game_num"] == shortest_game_id]

    print("Done")
