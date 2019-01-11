from glob import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from datetime import timezone
import os.path
import gc
import sys


# one argument: the data folder
data_folder = sys.argv[1]

# split full
dfs = []
for f in sorted(glob(data_folder + "/train_test/split_full/train/log_*")):
    df = pd.read_pickle(f)
    df = df[['session_code', 'session_position', 'track_code', 'skip']].copy()
    dfs.append(df)
    gc.collect()
dff = pd.concat(dfs, axis=0)
dff.to_pickle(data_folder + "/train_test/split_full/train/collaborative_data.pkl.gz")


# split 0
dfs = []
for f in sorted(glob(data_folder + "/train_test/split_0/train/log_*")):
    df = pd.read_pickle(f)
    df = df[['session_code', 'session_position', 'track_code', 'skip']].copy()
    dfs.append(df)
    gc.collect()
dff = pd.concat(dfs, axis=0)
dff.to_pickle(data_folder + "/train_test/split_0/train/collaborative_data.pkl.gz")
