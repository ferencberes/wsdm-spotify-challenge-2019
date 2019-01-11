from glob import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from datetime import timezone
import os.path
import sys


# one argument: the data folder
data_folder = sys.argv[1]


for f in sorted(glob(data_folder + "/train_test/split_full/train/log_*")):
    data = pd.read_pickle(f)

    unique_sessions = data.session_code.unique()

    np.random.shuffle(unique_sessions)

    train_data = data[data.session_code.isin(unique_sessions[:-len(unique_sessions) // 5])]
    test_data = data[data.session_code.isin(unique_sessions[-len(unique_sessions) // 5:])]

    train_data.to_pickle(f.replace(data_folder + "/train_test/split_full/train/", data_folder + "/train_test/split_0/train/"))

    test_data.to_pickle(f.replace(data_folder + "/train_test/split_full/train/", data_folder + "/train_test/split_0/train_ext/"))

    test_prehistory = test_data[test_data['session_position'] <= (test_data['session_length'] // 2)]
    test_input = test_data[test_data['session_position'] > (test_data['session_length'] // 2)]

    test_prehistory.to_pickle(f.replace(data_folder + "/train_test/split_full/train/log_", data_folder + "/train_test/split_0/test/log_prehistory_"))

    test_input[['session_position', 'session_length', 'session_code', 'track_code']].to_pickle(f.replace(data_folder + "/train_test/split_full/train/log_", data_folder + "/train_test/split_0/test/log_input_"))

for f in sorted(glob(data_folder + "/train_test/split_0/train_ext/log_*")):
    data = pd.read_pickle(f)
    data['skip'] = data['skip'] >= 2
    data = data[data['session_position'] > (data['session_length'] / 2)]
    data[['session_position', 'session_length', 'session_code', 'skip']].to_pickle(f.replace(data_folder + "/train_test/split_0/train_ext/", data_folder + "/train_test/split_0/ground_truth/"))
