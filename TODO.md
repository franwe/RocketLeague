# Priorities 

1. fix NNet to not create nan
2. Stable train/test set
3. processed data as csv

_______________________________________________________________________________

# List

## Data Processing
- check if sign of velocity makes sense, or maybe has some other coordinate
  system
- when normalizing, need first __overall__ min/max - not per file
- create fixed train and test set
- save processed data as csv
    - `BatchReader` should use from `data/processed`

## Features
- players on field
- players of team on field
- ball / players position + velocity (1, 2, 3)
- distances 
    - ball / players from goal
    - players from ball
    - players from players

## pytorch NNet

- NNet for sparse values (only few 1 / -1, many zeros)
- gets nan at some point
- make sure to use our loss
- test maybe without `test_loader`?
