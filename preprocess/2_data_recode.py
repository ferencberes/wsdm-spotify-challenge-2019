from glob import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from datetime import timezone
import os.path
import concurrent.futures
import sys

# one argument: the data folder
data_folder = sys.argv[1]


def recode(column, min_val=0, prev_code_map=None):
    if prev_code_map is None:
        uniques = column.unique()
        codes = range(min_val, len(uniques) + min_val)
        code_map = dict(zip(uniques, codes))
    else:
        uniques = list(set(column.unique()) - set(prev_code_map.keys()))
        min_val = max(list(prev_code_map.values()) + [min_val]) + 1
        codes = range(min_val, len(uniques) + min_val)
        code_map = prev_code_map
        code_map.update(dict(zip(uniques, codes)))

    return (column.map(code_map), code_map)


def reverse_code(column, code_map):
    inv_map = {v: k for k, v in code_map.items()}
    return column.map(inv_map)


# creating codes


session_ids = pd.read_csv(data_folder + "/unique_session_ids.csv", header=None, names=['session_id'])
track_ids = pd.read_csv(data_folder + "/unique_track_ids.csv", header=None, names=['track_id'])
session_ids['session_code'] = range(len(session_ids))
track_ids['track_code'] = range(len(track_ids))
session_ids.to_pickle(data_folder + '/session_codes.pkl.gz')
track_ids.to_pickle(data_folder + '/track_codes.pkl.gz')
files = glob(data_folder + "/training_set/*.csv")
behavior_map = {}
context_map = {}
file0 = pd.read_csv(files[0])
file0['hist_user_behavior_reason_end_code'], behavior_map = recode(file0['hist_user_behavior_reason_end'], prev_code_map=behavior_map)
file0['hist_user_behavior_reason_start_code'], behavior_map = recode(file0['hist_user_behavior_reason_start'], prev_code_map=behavior_map)
del file0['hist_user_behavior_reason_end']
del file0['hist_user_behavior_reason_start']

file0['context_type_code'], context_map = recode(file0['context_type'], prev_code_map=context_map)
del file0['context_type']
pd.DataFrame.from_records(list(behavior_map.items()), columns=['hist_user_behavior_reason', 'behavior_code']).to_pickle(data_folder + '/behavior_codes.pkl.gz')
pd.DataFrame.from_records(list(context_map.items()), columns=['context_type', 'context_type_code']).to_pickle(data_folder + '/context_type_codes.pkl.gz')

# actual coding

session_ids = pd.read_pickle(data_folder + '/session_codes.pkl.gz')
track_ids = pd.read_pickle(data_folder + '/track_codes.pkl.gz')
behavior_ids = pd.read_pickle(data_folder + '/behavior_codes.pkl.gz')
context_ids = pd.read_pickle(data_folder + '/context_type_codes.pkl.gz')
session_code_map = dict(session_ids.values)
track_code_map = dict(track_ids.values)
behavior_map = dict(behavior_ids.values)
context_map = dict(context_ids.values)
files = glob(data_folder + "/training_set/*.csv")

# initial attempt, too slow, run only on the first x files, but needed to gather all behavior codes
print("\r", 0, len(files), end=" ")
for i, f in list(enumerate(files))[:90]:
    fname = f.replace(data_folder + '/training_set/', data_folder + '/train_test/split_full/train/')[:-4] + '.pkl.gz'
    if not os.path.isfile(fname):
        file0 = pd.read_csv(f)
        file0['session_code'] = file0['session_id'].map(session_code_map)
        del file0['session_id']
        file0['track_code'] = file0['track_id_clean'].map(track_code_map)
        del file0['track_id_clean']

        file0['skip'] = file0['skip_1'].astype(np.int8) + file0['skip_2'].astype(np.int8) + file0['skip_3'].astype(np.int8)
        del file0['skip_1']
        del file0['skip_2']
        del file0['skip_3']
        del file0['not_skipped']

        file0['hist_user_behavior_reason_end_code'], behavior_map = recode(file0['hist_user_behavior_reason_end'], prev_code_map=behavior_map)
        file0['hist_user_behavior_reason_start_code'], behavior_map = recode(file0['hist_user_behavior_reason_start'], prev_code_map=behavior_map)
        del file0['hist_user_behavior_reason_end']
        del file0['hist_user_behavior_reason_start']

        file0['context_type_code'] = file0['context_type'].map(context_map)
        del file0['context_type']

        file0['session_position'] = file0['session_position'].astype(np.uint8)
        file0['session_length'] = file0['session_length'].astype(np.uint8)
        file0['context_switch'] = file0['context_switch'].astype(np.uint8)
        file0['no_pause_before_play'] = file0['no_pause_before_play'].astype(np.uint8)
        file0['short_pause_before_play'] = file0['short_pause_before_play'].astype(np.uint8)
        file0['long_pause_before_play'] = file0['long_pause_before_play'].astype(np.uint8)
        file0['hist_user_behavior_n_seekfwd'] = file0['hist_user_behavior_n_seekfwd'].astype(np.uint8)
        file0['hist_user_behavior_n_seekback'] = file0['hist_user_behavior_n_seekback'].astype(np.uint8)
        file0['hour_of_day'] = file0['hour_of_day'].astype(np.uint8)
        file0['session_code'] = file0['session_code'].astype(np.uint32)
        file0['track_code'] = file0['track_code'].astype(np.uint32)
        file0['hist_user_behavior_reason_end_code'] = file0['hist_user_behavior_reason_end_code'].astype(np.uint8)
        file0['hist_user_behavior_reason_start_code'] = file0['hist_user_behavior_reason_start_code'].astype(np.uint8)
        file0['context_type_code'] = file0['context_type_code'].astype(np.uint8)
        file0['date'] = pd.Series([dt.datetime.strptime(date, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp() for date in file0['date'].values]).astype(np.uint32)
        file0.to_pickle(fname)

        pd.DataFrame.from_records(list(behavior_map.items()), columns=['hist_user_behavior_reason', 'behavior_code']).to_pickle(data_folder + '/behavior_codes.pkl.gz')
    print("\r", i + 1, len(files), end=" ")

print("")


# actual working method (in reasonable time), with parallelization
def code(f):
    fname = f.replace(data_folder + '/training_set/', data_folder + '/train_test/split_full/train/')[:-4] + '.pkl.gz'
    if not os.path.isfile(fname):
        file0 = pd.read_csv(f)
        file0['session_code'] = file0['session_id'].map(session_code_map)
        del file0['session_id']
        file0['track_code'] = file0['track_id_clean'].map(track_code_map)
        del file0['track_id_clean']

        file0['skip'] = file0['skip_1'].astype(np.int8) + file0['skip_2'].astype(np.int8) + file0['skip_3'].astype(np.int8)
        del file0['skip_1']
        del file0['skip_2']
        del file0['skip_3']
        del file0['not_skipped']

        file0['hist_user_behavior_reason_end_code'] = file0['hist_user_behavior_reason_end'].map(behavior_map)
        file0['hist_user_behavior_reason_start_code'] = file0['hist_user_behavior_reason_start'].map(behavior_map)
        del file0['hist_user_behavior_reason_end']
        del file0['hist_user_behavior_reason_start']

        file0['context_type_code'] = file0['context_type'].map(context_map)
        del file0['context_type']

        file0['session_position'] = file0['session_position'].astype(np.uint8)
        file0['session_length'] = file0['session_length'].astype(np.uint8)
        file0['context_switch'] = file0['context_switch'].astype(np.uint8)
        file0['no_pause_before_play'] = file0['no_pause_before_play'].astype(np.uint8)
        file0['short_pause_before_play'] = file0['short_pause_before_play'].astype(np.uint8)
        file0['long_pause_before_play'] = file0['long_pause_before_play'].astype(np.uint8)
        file0['hist_user_behavior_n_seekfwd'] = file0['hist_user_behavior_n_seekfwd'].astype(np.uint8)
        file0['hist_user_behavior_n_seekback'] = file0['hist_user_behavior_n_seekback'].astype(np.uint8)
        file0['hour_of_day'] = file0['hour_of_day'].astype(np.uint8)
        file0['session_code'] = file0['session_code'].astype(np.uint32)
        file0['track_code'] = file0['track_code'].astype(np.uint32)
        file0['hist_user_behavior_reason_end_code'] = file0['hist_user_behavior_reason_end_code'].astype(np.uint8)
        file0['hist_user_behavior_reason_start_code'] = file0['hist_user_behavior_reason_start_code'].astype(np.uint8)
        file0['context_type_code'] = file0['context_type_code'].astype(np.uint8)
        file0['date'] = pd.Series([dt.datetime.strptime(date, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp() for date in file0['date'].values]).astype(np.uint32)
        file0.to_pickle(fname)


executor = concurrent.futures.ProcessPoolExecutor(5)
a = list(executor.map(code, files))
executor.shutdown()


# same for test files
def code_test_1(f):
    fname = f.replace(data_folder + '/test_set/', data_folder + '/train_test/split_full/test/')[:-4] + '.pkl.gz'
    if not os.path.isfile(fname):
        file0 = pd.read_csv(f)
        file0['session_code'] = file0['session_id'].map(session_code_map)
        del file0['session_id']
        file0['track_code'] = file0['track_id_clean'].map(track_code_map)
        del file0['track_id_clean']

        file0['skip'] = file0['skip_1'].astype(np.int8) + file0['skip_2'].astype(np.int8) + file0['skip_3'].astype(np.int8)
        del file0['skip_1']
        del file0['skip_2']
        del file0['skip_3']
        del file0['not_skipped']

        file0['hist_user_behavior_reason_end_code'] = file0['hist_user_behavior_reason_end'].map(behavior_map)
        file0['hist_user_behavior_reason_start_code'] = file0['hist_user_behavior_reason_start'].map(behavior_map)
        del file0['hist_user_behavior_reason_end']
        del file0['hist_user_behavior_reason_start']

        file0['context_type_code'] = file0['context_type'].map(context_map)
        del file0['context_type']

        file0['session_position'] = file0['session_position'].astype(np.uint8)
        file0['session_length'] = file0['session_length'].astype(np.uint8)
        file0['context_switch'] = file0['context_switch'].astype(np.uint8)
        file0['no_pause_before_play'] = file0['no_pause_before_play'].astype(np.uint8)
        file0['short_pause_before_play'] = file0['short_pause_before_play'].astype(np.uint8)
        file0['long_pause_before_play'] = file0['long_pause_before_play'].astype(np.uint8)
        file0['hist_user_behavior_n_seekfwd'] = file0['hist_user_behavior_n_seekfwd'].astype(np.uint8)
        file0['hist_user_behavior_n_seekback'] = file0['hist_user_behavior_n_seekback'].astype(np.uint8)
        file0['hour_of_day'] = file0['hour_of_day'].astype(np.uint8)
        file0['session_code'] = file0['session_code'].astype(np.uint32)
        file0['track_code'] = file0['track_code'].astype(np.uint32)
        file0['hist_user_behavior_reason_end_code'] = file0['hist_user_behavior_reason_end_code'].astype(np.uint8)
        file0['hist_user_behavior_reason_start_code'] = file0['hist_user_behavior_reason_start_code'].astype(np.uint8)
        file0['context_type_code'] = file0['context_type_code'].astype(np.uint8)
        file0['date'] = pd.Series([dt.datetime.strptime(date, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp() for date in file0['date'].values]).astype(np.uint32)
        file0.to_pickle(fname)


files = glob(data_folder + "/test_set/log_prehistory_*.csv")

executor = concurrent.futures.ProcessPoolExecutor(5)
a = list(executor.map(code_test_1, files))
executor.shutdown()

# and for test input files
files = glob(data_folder + "/test_set/log_input_*.csv")


def code_test_2(f):
    fname = f.replace(data_folder + '/test_set/', data_folder + '/train_test/split_full/test/')[:-4] + '.pkl.gz'
    if not os.path.isfile(fname):
        file0 = pd.read_csv(f)
        file0['session_code'] = file0['session_id'].map(session_code_map)
        del file0['session_id']
        file0['track_code'] = file0['track_id_clean'].map(track_code_map)
        del file0['track_id_clean']

        file0['session_position'] = file0['session_position'].astype(np.uint8)
        file0['session_length'] = file0['session_length'].astype(np.uint8)
        file0['session_code'] = file0['session_code'].astype(np.uint32)
        file0['track_code'] = file0['track_code'].astype(np.uint32)
        file0.to_pickle(fname)


executor = concurrent.futures.ProcessPoolExecutor(5)
a = list(executor.map(code_test_2, files))
executor.shutdown()
