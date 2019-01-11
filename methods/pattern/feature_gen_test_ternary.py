from glob import glob
import pandas as pd
import numpy as np
import sys

split_folder = sys.argv[1]

inp_files = sorted(glob(split_folder + "/test/log_input_*"))
pre_files = sorted(glob(split_folder + "/test/log_prehistory_*"))

skip_patterns = np.load(split_folder + '/models/skip_means_ternary.npy')
skip_nums = np.load(split_folder + '/models/skip_nums_ternary.npy')

for(inp_full_name, pre_full_name) in zip(inp_files, pre_files):
    print(inp_full_name)

    inp_full = pd.read_pickle(inp_full_name)
    pre_full = pd.read_pickle(pre_full_name)

    pre_skip_vals = np.minimum(pre_full.skip.values, 2)
    pre_session_codes = pre_full.session_code.values
    session_group_indices = np.where(np.ediff1d(pre_session_codes))[0] + 1
    skip_groups = np.split(pre_skip_vals, session_group_indices)
    session_groups = np.split(pre_session_codes, session_group_indices)

    inp_session_codes = inp_full.session_code.values
    inp_session_group_indices = np.concatenate([[0], np.where(np.ediff1d(inp_session_codes))[0] + 1])
    inp_session_lengths = np.concatenate([np.ediff1d(inp_session_group_indices), [len(inp_session_codes) - inp_session_group_indices[-1]]])

    powersequences = [[] for i in range(11)]
    for i in range(0, 11):
        powersequences[i] = 3**np.flip(np.arange(i))

    pred_pattern_needed = []
    for i in skip_groups:
        pred_pattern_needed.append(np.dot(powersequences[len(i)], i))

    preds_20 = skip_patterns[:, 10:][pred_pattern_needed]
    nums_20 = skip_nums[pred_pattern_needed]
    preds = np.concatenate([i[:l] for (i, l) in zip(preds_20, inp_session_lengths)])
    preds_confidence = np.concatenate([np.repeat(i, l) for (i, l) in zip(nums_20, inp_session_lengths)])
    inp_full['pred_ternary_proba'] = preds
    inp_full['pred_ternary_confidence'] = preds_confidence

    fname = split_folder + '/features/test/ternary_pattern_' + inp_full_name[len(split_folder + "/test/log_input_"):]
    inp_full[['pred_ternary_proba', 'pred_ternary_confidence']].to_pickle(fname)
