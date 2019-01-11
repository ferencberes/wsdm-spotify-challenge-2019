import pandas as pd
import numpy as np
import sys

split_folder = sys.argv[1]

data = pd.read_pickle(split_folder + '/train/collaborative_data.pkl.gz')

sess_val_counts = data.session_code.value_counts()
session_wlen20 = sess_val_counts[sess_val_counts == 20].index
data_20 = data[data.session_code.isin(session_wlen20)]

session_codes_np = data_20.session_code.values
skip_values_np = data_20.skip.values

skip_values_mat = skip_values_np.reshape(-1, 20)

# binary patterns
skip_2_values_mat = skip_values_mat >= 2
ind = np.lexsort(np.transpose(np.fliplr(skip_2_values_mat[:, :10])))
ordered_skip_2_values_mat = skip_2_values_mat[ind]
group_inds = np.where(np.any(np.diff(ordered_skip_2_values_mat[:, :10], axis=0), axis=1))[0] + 1
groups = np.split(ordered_skip_2_values_mat, group_inds)
group_2_means = np.stack([i.mean(axis=0) for i in groups], axis=0)
group_2_nums = np.array([i.shape[0] for i in groups])
np.save(split_folder + '/models/skip_means_binary', group_2_means)
np.save(split_folder + '/models/skip_nums_binary', group_2_nums)


# ternary patterns
skip_3_values_mat = np.minimum(skip_values_mat, 2)
skip_3_values_mat[:, 10:] = skip_3_values_mat[:, 10:] >= 2

ind = np.lexsort(np.transpose(np.fliplr(skip_3_values_mat[:, :10])))
ordered_skip_3_values_mat = skip_3_values_mat[ind]
group_inds = np.where(np.any(np.diff(ordered_skip_3_values_mat[:, :10], axis=0), axis=1))[0] + 1
groups = np.split(ordered_skip_3_values_mat, group_inds)
group_3_means = np.stack([i.mean(axis=0) for i in groups], axis=0)
group_3_nums = np.array([i.shape[0] for i in groups])

np.save(split_folder + '/models/skip_means_ternary', group_3_means)
np.save(split_folder + '/models/skip_nums_ternary', group_3_nums)
