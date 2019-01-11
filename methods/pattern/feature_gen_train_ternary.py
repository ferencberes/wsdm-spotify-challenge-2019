from glob import glob
import pandas as pd
import numpy as np
import sys

split_folder = sys.argv[1]

inp_files = sorted(glob(split_folder + "/train/log_*"))

skip_patterns = np.load(split_folder + '/models/skip_means_ternary.npy')
skip_nums = np.load(split_folder + '/models/skip_nums_ternary.npy')

for inp_full_name in inp_files:
    print(inp_full_name)

    inp_full = pd.read_pickle(inp_full_name)

    pre_skip_vals = np.minimum(inp_full.skip.values, 2)
    pre_session_codes = inp_full.session_code.values
    session_group_indices = np.where(np.ediff1d(pre_session_codes))[0] + 1
    skip_groups = np.split(pre_skip_vals, session_group_indices)
    session_groups = np.split(pre_session_codes, session_group_indices)

    powersequences = [[] for i in range(11)]
    for i in range(0, 11):
        powersequences[i] = 3**np.flip(np.arange(i))

    pred_pattern_needed = []
    ground_truths = []
    for i in skip_groups:
        l = i.shape[0] // 2
        pred_pattern_needed.append(np.dot(powersequences[l], i[:l]))
        ground_truths.append((i[l:] >= 2).astype(int))

    preds_20 = skip_patterns[:, 10:][pred_pattern_needed]
    pred_nums = skip_nums[pred_pattern_needed]
    preds = np.concatenate([(i[:len(gt)] * n - gt) / (n - 1) for (i, gt, n) in zip(preds_20, ground_truths, pred_nums)])
    preds_confidence = np.repeat(pred_nums, [len(l) - len(l) // 2 for l in session_groups])
    fname = split_folder + '/features/train/ternary_pattern_' + inp_full_name[len(split_folder + "/test/log_"):]
    pd.DataFrame({'pred_ternary_proba': preds, 'pred_ternary_confidence': preds_confidence}).to_pickle(fname)
