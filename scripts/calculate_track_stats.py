# coding: utf-8
import sys, os
sys.path.insert(0, '../utils/')
from sframe_utils import *
from wsdm_utils import *

# Parameters
if len(sys.argv) == 5:
    date_min = sys.argv[1]
    date_max = sys.argv[2]
    max_log_index = int(sys.argv[3])
    which_part = sys.argv[4]
    if not which_part in ["first","second","both"]:
        raise RuntimeError("Choose 'which_part' from values: 'first'/'second'/'both'")
else:
    raise RuntimeError("calculate_track_stats.py <date_min>  <date_max> <max_log_index> <which_part:'first'/'second'/'both'>")
experiment_id = "track_stats_%s_%s" % (date_min, date_max)
log_types = None if max_log_index == 9 else list(range(max_log_index+1))
experiment_dir = "/mnt/idms/fberes/data/wsdmcup19/deploy/split_0/"
data_dir = "/mnt/idms/projects/recsys2018/WSDM/data"
MAX_THREADS = 20
os.environ["OMP_NUM_THREADS"] = str(MAX_THREADS)

print("# 1. Load session files")

cols = ["session_code","session_position","session_length","track_code","skip"]

ss_tr = SessionFileSelector("%s/train_test/split_full/train/" % data_dir, mode="train")
print(ss_tr.log_summary.head())
session_data_tr = ss_tr.load_files(date_min, date_max, log_types)
print(len(session_data_tr))
session_data_tr = session_data_tr[cols]

if which_part != "second":
    ss_te = SessionFileSelector("%s/train_test/split_full/test/" % data_dir, mode="submission")
    print(ss_te.log_summary.head())
    session_data_te = ss_te.load_files(date_min, date_max, log_types)
    print(len(session_data_te))
    session_data_te = session_data_te[["session_code","session_position","session_length","track_code","skip"]]
    session_data = concatenate(session_data_tr, session_data_te)
    print("#### free memory")
    del session_data_te
    del session_data_tr
else:
    session_data = session_data_tr

print("Total number of records:", len(session_data))

print("# 2. Prepare data for aggregations")

print("## i.) Separate first and second half of the playlist")

if which_part == "both":
    data_for_stats = session_data
else:
    first_part_data, second_part_data = separate_session(session_data, cols)
    if which_part == "first":
        del session_data
        del second_part_data
        data_for_stats = first_part_data
    else:
        del session_data
        del first_part_data
        data_for_stats = second_part_data

print("## ii.) Label refactor")

data_for_stats = data_for_stats[data_for_stats["skip"] != None]
data_for_stats = process_skip_information(data_for_stats)

print("## iii.) Track info based aggregations")

track_infos = aggregate_track_info(data_for_stats, ["track_code"], "track")

def rename_track_info_cols(sf):
    """Rename coumns due to later joins during data preparation"""
    cols = sf.column_names()
    for c in ["track_code"]:
        if c in cols:
            cols.remove(c)
    new_cols = ["%s_%s" % (c, which_part) for c in cols]
    return sf.rename(dict(zip(cols,new_cols)))

track_infos = rename_track_info_cols(track_infos)

output_folder = "%s/train/%s" % (experiment_dir, experiment_id)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
track_infos.save("%s/%s_track_infos" % (output_folder, which_part), format='binary')

print("###DONE###")