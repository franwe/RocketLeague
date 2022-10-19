import pandas as pd
import numpy as np
from pathlib import Path

from utils.data import pickle_final_df, csv_final_df

DATA_DIR = Path.cwd().joinpath("data")
DATA_IN_DIR = DATA_DIR.joinpath("processed")
DATA_OUT_DIR = DATA_DIR.joinpath("processed", "downsampled")

subsets = [f"train_{i}" for i in range(10)]

if __name__ == "__main__":
    group_sample_size = 5000
    random_state = 1
    DATA_DIR = Path.cwd().joinpath("data")
    DATA_IN_DIR = DATA_DIR.joinpath("processed")
    DATA_OUT_DIR = DATA_DIR.joinpath("processed", f"downsampled_{group_sample_size}")

    np.random.seed(random_state)
    for subset in subsets:
        # only downsample train sets
        filename = f"{subset}_train.pkl"
        df = pd.read_pickle(DATA_IN_DIR.joinpath(filename))
        new_df = (
            df.groupby("team_scoring_within_10sec")
            .apply(lambda x: x.sample(n=group_sample_size))
            .reset_index(drop=True)
        )
        new_df = new_df.sample(frac=1).reset_index(drop=True)

        pickle_final_df(new_df, filename=f"{subset}_train.pkl", dir=DATA_OUT_DIR)
        csv_final_df(new_df, filename=f"{subset}_train.csv", dir=DATA_OUT_DIR)
