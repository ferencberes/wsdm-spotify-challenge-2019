# coding: utf-8
import os, sys

sys.path.insert(0, '../utils/')
from sframe_utils import *

import pandas as pd
import turicreate as tc
import turicreate.aggregate as agg

# # Parameters
raw_track_data_folder = "/mnt/idms/fberes/data/wsdmcup19/raw_track_features/track_features"
# TODO: Domokos preprocess needed!
track_code_file = "/mnt/idms/projects/recsys2018/WSDM/data/track_codes.pkl.gz"
output_file = "/mnt/idms/fberes/data/wsdmcup19/deploy/all_tracks.pickle.gz"
MAX_THREADS = 20
os.environ["OMP_NUM_THREADS"] = str(MAX_THREADS)

print("# # Load data")
tr_info = tc.SFrame.read_csv("%s/tf_000000000000.csv" % raw_track_data_folder)
tr_info = tr_info.append(tc.SFrame.read_csv("%s/tf_000000000001.csv" % raw_track_data_folder))
print(len(tr_info))
print(tr_info.head(2))

print("# # Transformations")

print("# ## Onehot encoding")
tr_info = get_dummies(tr_info, ["mode"])

print("# ## Track code join")
track_decoder = pd.read_pickle(track_code_file)
print(track_decoder.head())
track_id_dict = dict(zip(track_decoder["track_id"],track_decoder["track_code"]))
tr_info["track_code"] = tr_info["track_id"].apply(lambda x: track_id_dict[x])
tr_info = tr_info.remove_column("track_id")

print("# # Export transformed data")
print(tr_info.head(3))
print(len(tr_info))

print("# ## Pandas pickle")
tr_df = tr_info.to_dataframe()
print(len(tr_df))
tr_df.to_pickle(output_file)
print("Done")