# coding: utf-8

import sys, os
import pprint as pp
import pandas as pd
import numpy as np

import turicreate as tc
import turicreate.aggregate as agg

sys.path.insert(0, '../utils/')
from sframe_utils import *
from wsdm_utils import *

# Parameters
if len(sys.argv) == 5:
    date_min = sys.argv[1]
    date_max = sys.argv[2]
    max_log_index = int(sys.argv[3])
    stats_experiment_id = sys.argv[4]
else:
    raise RuntimeError("feature_engineering.py <date_min>  <date_max> <max_log_index> <stats_experiment_id>")

experiment_dir = "/mnt/idms/fberes/data/wsdmcup19/deploy/split_0/"
data_dir = "/mnt/idms/projects/recsys2018/WSDM/data"
compressed_track_file_path = "/mnt/idms/fberes/data/wsdmcup19/deploy/all_tracks.pickle.gz"
MAX_THREADS = 20
os.environ["OMP_NUM_THREADS"] = str(MAX_THREADS)

folder = "%s/train/%s_%s_with_%s/" % (experiment_dir, date_min, date_max, stats_experiment_id)
if not os.path.exists(folder):
    os.makedirs(folder)
print("Files will be saved in this folder:", folder)

print("# 1. Load track files")

tr_info_df = pd.read_pickle(compressed_track_file_path)
tr_info = tc.SFrame(tr_info_df)

track_cols = tr_info.column_names()
track_feats = track_cols.copy()
track_feats.remove("track_code")

track_feats_variance = np.array([tr_info[col].var() for col in track_feats])

print("# 2. Load session files")

sessions_directory = "%s/train_test/split_0/train/" % data_dir
# TODO: Domokos preprocess needed!
features_dir = "%s/train_test/split_0/features/train" % data_dir
ss = SessionFileSelector(sessions_directory)
print(ss.log_summary.head())

session_filter = {
     "date_min": date_min,
     "date_max": date_max,
     "log_types": None if max_log_index == 9 else list(range(max_log_index+1)),
     "features_dir":features_dir
}

session_data = ss.load_files(**session_filter)

# # Select random sessions
uniq_codes = np.unique(np.array(session_data["session_code"]))
print("Number of unique session:", len(uniq_codes))
selected_sessions = list(np.random.choice(uniq_codes, int(len(uniq_codes)*0.6), replace=True))
print(len(set(selected_sessions)) / len(uniq_codes))

print(len(session_data))
session_data = session_data.filter_by(selected_sessions, "session_code")
print(len(session_data))

print("# 3. Prepare data for classification")

print("## i.) Merge with track information")

session_data = session_data.join(tr_info, how="left", on="track_code")

print("#### free memory")
del tr_info

print("## ii.) Track info based aggregations")

track_stats_dir = "%s/train/%s/" % (experiment_dir, stats_experiment_id)

for part in ["first","second","both"]:
    track_infos = tc.load_sframe("%s/%s_track_infos" % (track_stats_dir, part))
    print(part, track_infos.shape)
    session_data = batch_join(session_data, track_infos, ["track_code"])
    print("#### free memory")
    del track_infos

print("## iii.) Repeat count of tracks")

track_repeat = session_data.groupby(["session_code","track_code"], operations={"repeat_cnt":agg.COUNT()})

track_repeat = track_repeat[track_repeat["repeat_cnt"] > 1]
keys = list(zip(track_repeat["session_code"],track_repeat["track_code"]))
track_repeat_dict = dict(zip(keys, track_repeat["repeat_cnt"]))
del track_repeat

session_data["repeat_cnt"] = session_data.apply(lambda x: track_repeat_dict.get((x["session_code"],x["track_code"]), 1))
del track_repeat_dict

print("## iv.) Mahalanobis distance")

def get_dists(sf, cols, variance_dict):
    for i, col in enumerate(cols):
        s_err = np.square(sf[col]-sf["%s_MEAN" % col]) / variance_dict[col]
        if i == 0:
            s_err_sum = s_err
        else:
            s_err_sum += s_err
        print(col, (i+1) / len(cols))
    return np.sqrt(s_err_sum / len(cols))

track_means = session_data.groupby("session_code", operations={"%s_MEAN" % col : agg.MEAN(col) for col in track_feats})
session_data = batch_join(session_data, track_means, ["session_code"])
del track_means

var_dict = dict(zip(track_feats, track_feats_variance))
session_data["dist_from_sess_mean"] = get_dists(session_data, track_feats, var_dict)
session_data = session_data.remove_columns(["%s_MEAN" % col for col in track_feats])

print("## v.) aggregations for total session")

agg_cols_total = session_data.column_names().copy()
session_cols = ['session_position', 'session_length', 'context_switch',
       'no_pause_before_play', 'short_pause_before_play',
       'long_pause_before_play', 'hist_user_behavior_n_seekfwd',
       'hist_user_behavior_n_seekback', 'hist_user_behavior_is_shuffle',
       'hour_of_day', 'date', 'premium', 'session_code', 'track_code', 'skip',
       'hist_user_behavior_reason_end_code',
       'hist_user_behavior_reason_start_code', 'context_type_code']
for col in session_cols:
    agg_cols_total.remove(col)

def session_aggr(sf, cols, key="session_code"):
    mean_operations = {("%s_mean" % col): agg.MEAN(col) for col in cols}
    std_operations = {("%s_std" % col): agg.STD(col) for col in cols}
    min_operations = {("%s_min" % col): agg.MIN(col) for col in cols}
    max_operations = {("%s_max" % col): agg.MAX(col) for col in cols}
    all_operations = {}
    all_operations.update(mean_operations)
    all_operations.update(std_operations)
    all_operations.update(min_operations)
    all_operations.update(max_operations)
    return sf.groupby(key_column_names=[key], operations=all_operations)

session_stats = session_aggr(session_data, agg_cols_total)

print("## vi.) Session heterosity")

ops = {
    "num_uniq_tracks": agg.COUNT_DISTINCT("track_code"),
    "session_length": agg.MIN("session_length")
}

uniq_info = session_data.groupby("session_code", operations=ops)
uniq_info["track_heterogenity"] = uniq_info["num_uniq_tracks"] / uniq_info["session_length"]
uniq_info["track_repetition"] = uniq_info["session_length"] - uniq_info["num_uniq_tracks"] 
uniq_info = uniq_info.remove_column("session_length")
session_stats = session_stats.join(uniq_info, on="session_code")
del uniq_info
session_stats.save("%s/sess_stats" % folder, format='binary')

print("## vii.) Separate first and second half of the playlist")

session_data["position_over_length"] = session_data["session_position"] / session_data["session_length"]

session_data = drop_columns(session_data, ["track_code"])
print(session_data.shape)

eval_enabled_cols = session_data.column_names().copy()
excluded = ['context_switch', 'no_pause_before_play', 'short_pause_before_play',
       'long_pause_before_play', 'hist_user_behavior_n_seekfwd',
       'hist_user_behavior_n_seekback', 'hist_user_behavior_is_shuffle',
       'hour_of_day', 'date', 'premium', 'hist_user_behavior_reason_end_code',
       'hist_user_behavior_reason_start_code', 'context_type_code']
for col in excluded:
    eval_enabled_cols.remove(col)

first_part_data, second_part_data = separate_session(session_data, eval_enabled_cols)

print("## viii.) Label refactor")

print("#### free memory")
del session_data

print("### a.) first part corrections")

first_part_data = process_skip_information(first_part_data, exec_onehot=True)
cols_to_drop = [
    "pred_binary_proba",
    "pred_binary_confidence",
    "pred_ternary_proba",
    "pred_ternary_confidence",
]
first_part_data = drop_columns(first_part_data, cols_to_drop)

print("### b.) second part corrections")

second_part_data = process_skip_information(second_part_data, exec_onehot=False)
second_part_data = second_part_data.remove_column("skip")

print("## ix.) Onehot Encoding")

first_part_data = get_dummies(first_part_data, ["context_type_code","hist_user_behavior_reason_start_code", "hist_user_behavior_reason_end_code"])

print("## x.) Calculate stats from first part")

agg_cols_first = list(set(first_part_data.column_names())-set(agg_cols_total))
agg_cols_first.remove("session_code")
agg_cols_first.remove("session_position")
agg_cols_first.remove("session_length")
agg_cols_first.remove("position_over_length")

first_part_stats = session_aggr(first_part_data, agg_cols_first)

print("## 4.) Export first part")

first_part_data = batch_join(first_part_data, session_stats, keys="session_code")
first_part_data = first_part_data.sort(["session_code","session_position"])
print(first_part_data.shape)
first_part_data.save("%s/first_part_sess" % folder, format='binary')

print("#### free memory")
del first_part_data

print("# 5.) Export second part")

second_part_data = batch_join(second_part_data, session_stats, keys="session_code")
del session_stats

second_part_data = batch_join(second_part_data, first_part_stats, keys="session_code")
del first_part_stats

second_part_data = second_part_data.sort(["session_code","session_position"])
print(second_part_data.shape)
second_part_data.save("%s/second_part_sess" % folder, format='binary')

print("###DONE###")