import pandas as pd
import gc
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import sklearn.metrics
from glob import glob
import sys


# one argument: the split folder
split_folder = sys.argv[1]


for split in range(5):
    chunks = np.array_split(sorted(glob(split_folder + "/train/log_*")), 5)
    test_files = chunks[split]
    train_files = np.concatenate([chunks[i] for i in np.arange(len(chunks))[np.arange(len(chunks)) != split]])

    inp = pd.concat([
        pd.read_pickle(i)
        for i in train_files
    ], axis=0)

    inp_session_codes = inp.session_code.values
    inp_split_indices = np.where(np.ediff1d(inp_session_codes))[0] + 1
    inp_track_values = inp['track_code'].values
    inp_skip_values = (inp['skip'].values >= 2) * 2 - 1
    session_lengths = inp['session_length'].values[np.concatenate([[0], inp_split_indices])]
    del inp
    zero_lengths = 20 - session_lengths

    zero_pads = np.split(np.zeros(zero_lengths.sum()), np.cumsum(zero_lengths))[:-1]
    inp_track_values_split_signed = np.split(inp_skip_values * inp_track_values, inp_split_indices)

    correct_order = [i for t in zip(inp_track_values_split_signed, zero_pads) for i in t]
    combined_fixed_length = np.concatenate(correct_order)

    np.save(split_folder + "/train/combined_fixed_length_split_" + str(split) + "_train", combined_fixed_length)
