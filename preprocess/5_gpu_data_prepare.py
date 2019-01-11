import pandas as pd
import gc
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
from glob import glob
from datetime import timezone
import os.path
import sys


# one argument: the split folder
split_folder = sys.argv[1]


inp = pd.concat([
    pd.read_pickle(i)
    for i in sorted(glob(split_folder + "/test/log_input_*"))
], axis=0)


prehist = pd.concat([
    pd.read_pickle(i)
    for i in sorted(glob(split_folder + "/test/log_prehistory_*"))
], axis=0)

inp_session_codes = inp.session_code.values
prehist_session_codes = prehist.session_code.values


prehist_split_indices = np.where(np.ediff1d(prehist_session_codes))[0] + 1
inp_split_indices = np.where(np.ediff1d(inp_session_codes))[0] + 1

prehist_signed_track_values = prehist['track_code'].values * ((prehist['skip'].values >= 2) * 2 - 1)
inp_track_values = inp['track_code'].values

session_lengths = inp['session_length'].values[np.concatenate([[0], inp_split_indices])]
zero_lengths = 20 - session_lengths


zero_pads = np.split(np.zeros(zero_lengths.sum()), np.cumsum(zero_lengths))[:-1]
prehist_signed_track_values_split = np.split(prehist_signed_track_values, prehist_split_indices)
inp_track_values_split = np.split(inp_track_values, inp_split_indices)


correct_order = [i for t in zip(prehist_signed_track_values_split, inp_track_values_split, zero_pads) for i in t]
padded_input = np.concatenate(correct_order)

np.save(split_folder + "/test/combined_fixed_length_orig", padded_input.astype(np.int32))
