def batch_files(files, N):
    lines = []
    for file in files:
        with open(file, "r") as infile:
            for line in infile:
                lines.append(line)
                if len(lines) >= N:
                    yield lines
                    lines = []
    if len(lines) > 0:
        yield lines


if __name__ == "__main__":
    files = [
        "/Users/franwe/repos/RocketLeague/data/tmp_files/0.csv",
        "/Users/franwe/repos/RocketLeague/data/tmp_files/1.csv",
        "/Users/franwe/repos/RocketLeague/data/tmp_files/2.csv",
    ]

    batches = batch_files(files, 4)
    for batch in batches:
        print(batch)
