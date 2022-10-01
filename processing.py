import pandas as pd
from pathlib import Path

DATA_DIR = Path.cwd().joinpath("data")

dtypes_df = pd.read_csv(DATA_DIR.joinpath("train_dtypes.csv"))
dtypes = {k: v for (k, v) in zip(dtypes_df.column, dtypes_df.dtype)}
train0_df = pd.read_csv(DATA_DIR.joinpath("train_0.csv"), dtype=dtypes)

shortest_game_id = train0_df["game_num"].value_counts().index[-1]
short_game = train0_df[train0_df["game_num"] == shortest_game_id]

print("Done")