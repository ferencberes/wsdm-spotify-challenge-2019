# coding: utf-8

import sys, os
import turicreate as tc
import turicreate.aggregate as agg

import pandas as pd
import numpy as np
import pprint as pp

sys.path.insert(0, '../utils/')
from wsdm_utils import *

# Parameters
if len(sys.argv) == 5:
    experiment_dir_root = sys.argv[1]
    MAX_THREADS = int(sys.argv[2])
    experiment_id = sys.argv[3]
    num_rnd_sessions = sys.argv[4]
else:
    raise RuntimeError("train_gbt_model.py <experiment_dir> <max_threads> <experiment_id> <num_rnd_sessions>")

experiment_dir = "%s/split_0/" % experiment_dir_root
model_params = {"max_iterations":80, "max_depth":5, "min_child_weight":100, "validation_set":'auto'}
max_num_training_records = 10000000
k = 100
os.environ["OMP_NUM_THREADS"] = str(MAX_THREADS)

# # 1. Load prepared data
# 
#    * with all features
#    * session and track information is joined

input_path = "%s/train/%s/second_part_sess" % (experiment_dir, experiment_id)
print(input_path)
prepared_data = tc.load_sframe(input_path)
print(prepared_data.shape)

print(prepared_data.column_names())

# # 2. Extract important features
# 
#    * train for 1M records: 4min 54s
#    * most of the time is sframe to pandas dataframe conversion

mt = ModelTrainerSFrame(prepared_data)

result_dict = mt.train(model_param_dict=model_params, k=num_rnd_sessions, test_ratio=0.3)
tr_pred = sframe_model_predict(result_dict["model"], result_dict["train_features"], result_dict["train_session_codes"], "target")
tr_map_vector = meanap_mt(tr_pred["skip"], tr_pred["prediction"], tr_pred["session_code"], threads=MAX_THREADS)
te_pred = sframe_model_predict(result_dict["model"], result_dict["test_features"], result_dict["test_session_codes"], "target")
te_map_vector = meanap_mt(te_pred["skip"], te_pred["prediction"], te_pred["session_code"], threads=MAX_THREADS)

print(np.mean(tr_map_vector), np.mean(te_map_vector))

importance = sframe_important_feats(result_dict["model"])[:k]
print(importance.print_rows(k))

# # 3. Train model only for important columns
# 
#    * train: 5M records (50 selected features) 4min 7s
#    * train: 15M records (50 selected features) 18min 12s eval: 2min 34s

selected_columns = list(importance["name"])
if not "session_position" in selected_columns:
    selected_columns.append("session_position")

if len(mt.data) > max_num_training_records:
    tr_ratio = max_num_training_records / len(mt.data)
    te_ratio = 1.0 - tr_ratio
else:
    te_ratio = 0.4
print(te_ratio)

final_result_dict = mt.train(model_param_dict=model_params, columns=selected_columns, test_ratio=te_ratio)

ftr_pred = sframe_model_predict(final_result_dict["model"], final_result_dict["train_features"], final_result_dict["train_session_codes"], "target")
ftr_map_vector = meanap_mt(ftr_pred["skip"], ftr_pred["prediction"], ftr_pred["session_code"], threads=MAX_THREADS)
fte_pred = sframe_model_predict(final_result_dict["model"], final_result_dict["test_features"], final_result_dict["test_session_codes"], "target")
fte_map_vector = meanap_mt(fte_pred["skip"], fte_pred["prediction"], fte_pred["session_code"], threads=MAX_THREADS)

print(np.mean(ftr_map_vector), np.mean(fte_map_vector))

# # 4. Save GBT model

model_folder = "%s/models" % experiment_dir
if not os.path.exists(model_folder):
    os.makedirs(model_folder)
model_path = "%s/%s" % (model_folder, experiment_id)
final_result_dict["model"].save(model_path)

print("###DONE###")