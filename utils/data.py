from io import StringIO
import pandas as pd


class BatchReader:
    def __init__(self, files, N, skip_header=True, sep=","):
        self.files = files
        self.N = N
        self.header = ""
        self.skip_header = skip_header
        self.sep = sep

    def read(self):
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
                        yield lines
                        lines = ""
                        i = 0
                if not line.endswith("\n"):
                    lines += "\n"
        if i > 0:
            yield lines

    def get_header(self):
        with open(self.files[0]) as f:
            first_line = f.readline()
        self.header = first_line.strip()
        return first_line

    @staticmethod
    def make_df(batch):
        s = StringIO(batch)
        df = pd.read_csv(s, sep=br.sep)
        df.columns = [c for c in br.header.split(br.sep)]
        return df


if __name__ == "__main__":

    files = [
        "/Users/franwe/repos/RocketLeague/data/tmp_files/0.csv",
        "/Users/franwe/repos/RocketLeague/data/tmp_files/1.csv",
        "/Users/franwe/repos/RocketLeague/data/tmp_files/2.csv",
    ]

    br = BatchReader(files, 4)
    batches = br.read()

    for batch in batches:
        df = br.make_df(batch)
        print(df)
