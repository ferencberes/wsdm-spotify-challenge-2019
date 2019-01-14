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
if len(sys.argv) == 8:
    data_dir = sys.argv[1]
    experiment_dir_root = sys.argv[2]
    MAX_THREADS = int(sys.argv[3])
    date_min = sys.argv[4]
    date_max = sys.argv[5]
    max_log_index = int(sys.argv[6])
    model_experiment_id = sys.argv[7]
else:
    raise RuntimeError("predict_with_model.py <data_dir> <experiment_dir> <max_threads> <date_min>  <date_max> <max_log_index> <model_experiment_id>")

experiment_id = "%s_%s" % (date_min, date_max)
experiment_dir = "%s/split_0/" % experiment_dir_root
compressed_track_file_path = "%s/all_tracks.pickle.gz" % data_dir
stats_experiment_id = model_experiment_id.split("_with_")[1]
ground_truth_exist = False
#ground_truth_exist = True
os.environ["OMP_NUM_THREADS"] = str(MAX_THREADS)

print("# 1. Load track files")

tr_info_df = pd.read_pickle(compressed_track_file_path)
tr_info = tc.SFrame(tr_info_df)
track_cols = tr_info.column_names()
track_feats = track_cols.copy()
track_feats.remove("track_code")
track_feats_variance = np.array([tr_info[col].var() for col in track_feats])

print("# 2. Load session files")

if ground_truth_exist:
    sessions_directory = "%s/train_test/split_0/test/" % data_dir
    features_dir = "%s/train_test/split_0/features/test/" % data_dir
    ss = SessionFileSelector(sessions_directory, mode="test")
else:
    sessions_directory = "%s/train_test/split_full/test"  % data_dir
    features_dir = "%s/train_test/split_full/features/test/" % data_dir
    ss = SessionFileSelector(sessions_directory, mode="submission")

print(ss.log_summary.head())

session_filter = {
     "date_min": date_min,
     "date_max": date_max,
     "log_types": None if max_log_index == 9 else list(range(max_log_index+1)),
     "features_dir":features_dir
}

session_data = ss.load_files(**session_filter)
print(len(session_data))

print("# 3. Load model")

model_path = "%s/models/%s/" % (experiment_dir, model_experiment_id)
model = tc.load_model(model_path)
print(model_path)

model_features = model.features
print(model_features)

print("## Drop irrelevant columns from track info")

print(tr_info.shape)
track_cols = []
for tr_col in tr_info.column_names():
    for m_col in model_features:
        if tr_col in m_col:
            track_cols.append(tr_col)
            break
track_cols += ["track_code"]
print(track_cols)

print("## Selected categorical columns")

selected_cat_cols = [col for col in model_features if "ONEHOT" in col ]
selected_cat_cols = [col.replace("_min","") for col in selected_cat_cols]
selected_cat_cols = [col.replace("_max","") for col in selected_cat_cols]
selected_cat_cols = [col.replace("_mean","") for col in selected_cat_cols]
selected_cat_cols = [col.replace("_std","") for col in selected_cat_cols]
selected_cat_cols = list(set(selected_cat_cols))
print(len(selected_cat_cols))
print(selected_cat_cols)

print("# 4. Prepare data for classification")

print("## i.) Merge with track information")

session_data = session_data.join(tr_info, how="left", on="track_code")
print("#### free memory")
del tr_info

print("## ii.) Track info based aggregations")

track_stats_dir = "%s/train/%s/" % (experiment_dir, stats_experiment_id)
track_infos = tc.load_sframe("%s/%s_track_infos" % (track_stats_dir, "first"))
track_inf_cols = {
    "first":set(["track_code"]),
    "second":set(["track_code"]),
    "both":set(["track_code"]),
}
for m_col in model_features:
    if "_track_" in m_col:
        splitted = m_col.split("_")
        if splitted[-1] in ["mean", "std", "min", "max"]:
            m_col_origi = "_".join(splitted[:-1])
        else:
            m_col_origi = m_col
        part = m_col_origi.split("_")[-1]
        track_inf_cols[part].add(m_col_origi)

for part in ["first","second","both"]:
    track_infos = tc.load_sframe("%s/%s_track_infos" % (track_stats_dir, part))
    print(part, track_infos.shape)
    session_data = batch_join(session_data, track_infos[list(track_inf_cols[part])], ["track_code"])
    print("#### free memory")
    del track_infos

print("## iii.) Repeat count of tracks")

calc_repeat_cnt = False
for m_col in model_features:
    if "repeat_cnt" in m_col:
        calc_repeat_cnt = True
        break
print("calculate repeat_cnt:", calc_repeat_cnt)

if calc_repeat_cnt:
    track_repeat = session_data.groupby(["session_code","track_code"], operations={"repeat_cnt":agg.COUNT()})
    track_repeat = track_repeat[track_repeat["repeat_cnt"] > 1]
    keys = list(zip(track_repeat["session_code"],track_repeat["track_code"]))
    track_repeat_dict = dict(zip(keys, track_repeat["repeat_cnt"]))
    del track_repeat
    session_data["repeat_cnt"] = session_data.apply(lambda x: track_repeat_dict.get((x["session_code"],x["track_code"]), 1))
    del track_repeat_dict
    print("'repeat_cnt' generated!")
else:
    print("'repeat_cnt' skipped!")

print("## iv.) Mahalanobis distance")

calc_distances = False
for m_col in model_features:
    if "dist_from_sess_mean" in m_col:
        calc_distances = True
        break
print("calculate dist_from_sess_mean:", calc_distances)

def get_dists(sf, cols, variance_dict):
    for i, col in enumerate(cols):
        s_err = np.square(sf[col]-sf["%s_MEAN" % col]) / variance_dict[col]
        if i == 0:
            s_err_sum = s_err
        else:
            s_err_sum += s_err
        print(col, (i+1) / len(cols))
    return np.sqrt(s_err_sum / len(cols))

if calc_distances:
    track_means = session_data.groupby("session_code", operations={"%s_MEAN" % col : agg.MEAN(col) for col in track_feats})
    session_data = batch_join(session_data, track_means, ["session_code"])
    del track_means
    var_dict = dict(zip(track_feats, track_feats_variance))
    session_data["dist_from_sess_mean"] = get_dists(session_data, track_feats, var_dict)
    session_data = session_data.remove_columns(["%s_MEAN" % col for col in track_feats])
    print("'dist_from_sess_mean' generated!")
else:
    print("'dist_from_sess_mean' skipped!")

track_cols_to_remove = list(set(track_feats)-set(track_cols))
session_data = session_data.remove_columns(track_cols_to_remove)
print("track cols to remove:", track_cols_to_remove)

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
    if col in agg_cols_total:
        agg_cols_total.remove(col)

cols_for_total_aggr = []
for m_col in model_features:
    for col in agg_cols_total:
        if col in m_col:
            cols_for_total_aggr.append(m_col)
            break

print("cols for total aggregations:", cols_for_total_aggr)

agg_total_operations = {}
for col in cols_for_total_aggr:
    parts = col.split("_")
    feat = "_".join(parts[:-1])
    if parts[-1] == "mean" and feat != "dist_from_sess":
        agg_total_operations[col] = agg.MEAN(feat)
    elif parts[-1] == "std":
        agg_total_operations[col] = agg.STD(feat)
    elif parts[-1] == "min":
        agg_total_operations[col] = agg.MIN(feat)
    elif parts[-1] == "max":
        agg_total_operations[col] = agg.MAX(feat)
    else:
        print(col)
        continue

session_stats = session_data.groupby(key_column_names=["session_code"], operations=agg_total_operations)

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
    if col in eval_enabled_cols:
        eval_enabled_cols.remove(col)
first_part_data, second_part_data = separate_session(session_data, eval_enabled_cols)

cols_to_drop = [
    "pred_binary_proba",
    "pred_binary_confidence",
    "pred_ternary_proba",
    "pred_ternary_confidence",
]
first_part_data = drop_columns(first_part_data, cols_to_drop)
print("#### free memory")
del session_data

print("## viii.) Label refactor")

print("### a.) first part corrections")

first_part_data = process_skip_information(first_part_data, exec_onehot=True)

print("### b.) second part corrections")
second_part_data = second_part_data.remove_column("skip")

print("## ix.) Onehot Encoding")

categorical_cols = [
    "context_type_code",
    "hist_user_behavior_reason_start_code",
    "hist_user_behavior_reason_end_code"
]

first_part_data = get_dummies(first_part_data, categorical_cols, selected_columns=selected_cat_cols)

print("## x.) Calculate stats from first part")

agg_cols_first = list(set(first_part_data.column_names())-set(agg_cols_total))
agg_cols_first.remove("session_code")
agg_cols_first.remove("session_position")
agg_cols_first.remove("session_length")
agg_cols_first.remove("position_over_length")

cols_for_first_aggr = []
for m_col in model_features:
    for col in agg_cols_first:
        if col in m_col:
            cols_for_first_aggr.append(m_col)
            break

print("cols for first aggregations:", cols_for_first_aggr)

agg_first_operations = {}
for col in cols_for_first_aggr:
    parts = col.split("_")
    feat = "_".join(parts[:-1])
    if parts[-1] == "mean" and feat != "dist_from_sess":
        agg_first_operations[col] = agg.MEAN(feat)
    elif parts[-1] == "std":
        agg_first_operations[col] = agg.STD(feat)
    elif parts[-1] == "min":
        agg_first_operations[col] = agg.MIN(feat)
    elif parts[-1] == "max":
        agg_first_operations[col] = agg.MAX(feat)
    else:
        print(col)
        continue
first_part_stats = first_part_data.groupby(key_column_names=["session_code"], operations=agg_first_operations)

print("# 5.) Export second part")

second_part_data = batch_join(second_part_data, session_stats, keys="session_code")
del session_stats
second_part_data = batch_join(second_part_data, first_part_stats, keys="session_code")
del first_part_stats
second_part_data = second_part_data.sort(["session_code","session_position"])
print(second_part_data.shape)

print("# 6.) Predict")

model_features_tmp = model_features.copy()
if not "session_code" in model_features_tmp:
    model_features_tmp.append("session_code")
if not "session_position" in model_features_tmp:
    model_features_tmp.append("session_position")
print(len(model_features_tmp))
second_part_data = second_part_data[model_features_tmp]

print("### Load ground truth labels")

if ground_truth_exist:
    output_folder = "%s/test/%s/" % (experiment_dir, model_experiment_id)
else:
    output_folder = "%s/split_full/test/%s/" % (experiment_dir, model_experiment_id)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

if ground_truth_exist:
    pred_session_filter = session_filter.copy()
    pred_session_filter["features_dir"] = None
    ll = SessionFileSelector("%s/train_test/split_0/ground_truth/" % data_dir, mode="label")
    ground_truth_labels = ll.load_files(**pred_session_filter)[["session_code","session_position","skip"]]
    ground_truth_labels = ground_truth_labels.rename({"skip":"target"})
    print(ground_truth_labels.head(5))
    second_part_data = second_part_data.join(ground_truth_labels, on=["session_code", "session_position"])
    print(value_counts(second_part_data, "target"))

print("Saving second part of the sessions STARTED...")

second_part_data.save("%s/predict_data_%s" % (output_folder, experiment_id), format='binary')
print("Saving second part of the sessions DONE")

session_codes = np.array(second_part_data["session_code"])
second_part_data = second_part_data.remove_column("session_code")
missing_info = num_missing(second_part_data)
print(missing_info)

print("### Make predictions")

if ground_truth_exist:
    pred = sframe_model_predict(model, second_part_data, session_codes, "target")
    map_vector = meanap_mt(pred["skip"], pred["prediction"], pred["session_code"], threads=MAX_THREADS)
    print("mAP:", np.mean(map_vector))
else:
    pred = sframe_model_predict(model, second_part_data, session_codes, label_col=None)
print(pred.head())
pred.export_csv("%s/%s.csv" % (output_folder, experiment_id))

print("###DONE###")