from io import StringIO
from typing import List
import pandas as pd
from abc import ABC, abstractmethod
import torch


class BatchReader:
    def __init__(self, files, N, skip_header=True, sep=","):
        self.files = files
        self.N = N
        self.header = ""
        self.skip_header = skip_header
        self.sep = sep

    def read(self, formatting=None):
        if not formatting:
            formatting = self.format_df
        lines = ""
        i = 0
        for file in self.files:
            with open(file, "r") as infile:
                current_file_header = ""
                for line in infile:
                    if not current_file_header and self.skip_header:
                        current_file_header = line.strip()
                        if self.header and current_file_header != self.header:
                            raise ValueError(
                                f"Headers of files don't match!"
                                f" - header expected: {self.header}"
                                f" - header seen    : {current_file_header}"
                                f" - file: {file} "
                            )
                        self.header = current_file_header
                        continue
                    lines += line
                    i += 1
                    if i >= self.N:
                        yield formatting(lines)
                        lines = ""
                        i = 0
                if not line.endswith("\n"):
                    lines += "\n"
        if i > 0:
            yield formatting(lines)

    def get_header(self):
        with open(self.files[0]) as f:
            first_line = f.readline()
        self.header = first_line.strip()
        return first_line

    def format_df(self, batch):
        s = StringIO(batch)
        df = pd.read_csv(s, sep=self.sep)
        df.columns = [c for c in self.header.split(self.sep)]
        return df

    def format_none(self, batch):
        return batch


class DataPreparer(ABC):
    def __init__(self, features, target, range_filename):
        self.features = features
        self.target = target
        self.range_filename = range_filename
        self.ranges = self._load_ranges()

    @abstractmethod
    def prepare_data(df: pd.DataFrame):
        raise NotImplementedError

    @abstractmethod
    def normalize_data(self, values: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def _load_ranges(self):
        ranges = pd.read_csv(self.range_filename, index_col=0)
        return ranges


class DataPreparerTorch(DataPreparer):
    def __init__(self, features, target, range_filename, device):
        DataPreparer.__init__(self, features, target, range_filename)
        self.device = device

    def prepare_data(self, df):
        values = self.normalize_data(df[self.features]).values
        labels = df[self.target].values
        values, labels = torch.from_numpy(values), torch.from_numpy(labels.reshape(len(labels)))
        values, labels = values.float(), labels.long()
        values, labels = values.to(self.device), labels.to(self.device)
        return values, labels

    def normalize_data(self, values: pd.DataFrame) -> pd.DataFrame:
        normalized_values = pd.DataFrame()
        for col in values.columns:
            min = self.ranges.loc[col, "min"]
            max = self.ranges.loc[col, "max"]
            normalized_values[col] = (values[col] - min) / (max - min)
        return normalized_values


if __name__ == "__main__":

    files = [
        "/Users/franwe/repos/RocketLeague/data/tmp_files/0.csv",
        "/Users/franwe/repos/RocketLeague/data/tmp_files/1.csv",
        "/Users/franwe/repos/RocketLeague/data/tmp_files/2.csv",
    ]

    br = BatchReader(files, 7)
    batches = br.read()

    for batch in batches:
        print(batch)

    range_filename = "/Users/franwe/repos/RocketLeague/data/tmp_files/ranges.csv"
    dpt = DataPreparerTorch(features=["a"], target="", range_filename=range_filename, device=None)
    last_batch_normalized = dpt.normalize_data(batch)
    print(last_batch_normalized)
