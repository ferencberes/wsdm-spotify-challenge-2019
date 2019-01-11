import os, pickle, time
import pprint as pp
import pandas as pd
import numpy as np
import turicreate as tc

from sframe_utils import drop_columns, concatenate, separate_session

### Files ###

def join_features(log_path, bin_path, ter_path, autoenc_path):
    log_info = tc.SFrame(pd.read_pickle(log_path))
    bin_info = tc.SFrame(pd.read_pickle(bin_path))
    ter_info = tc.SFrame(pd.read_pickle(ter_path))
    aut_info = tc.SFrame(pd.read_pickle(autoenc_path))
    #print("files loaded")
    cols = log_info.column_names()
    fp, sp = separate_session(log_info, cols)
    #print("separation done")
    for c in bin_info.column_names():
        sp[c] = bin_info[c]
        #print(c)
    for c in ter_info.column_names():
        sp[c] = ter_info[c]
        #print(c)
    for c in aut_info.column_names():
        sp[c] = aut_info[c]
        #print(c)
    #print("columns were copied")
    return concatenate(fp, sp)

class SessionFileSelector():
    def __init__(self, folder, mode="train"):
        self.folder = folder
        self.mode = mode
        files_in_dir = [f for f in os.listdir(folder) if "log_" in f]
        splitted_files = [f.split("_") for f in files_in_dir]
        if self.mode == "train":
            part_info = ['input' for item in splitted_files]
            log_info = [item[1] for item in splitted_files]
            date_info = [item[2] for item in splitted_files]
        elif self.mode == "test":
            part_info = [item[1] for item in splitted_files]
            log_info = [item[2] for item in splitted_files]
            date_info = [item[3] for item in splitted_files]
        elif self.mode == "label":
            part_info = ['label' for item in splitted_files]
            log_info = [item[1] for item in splitted_files]
            date_info = [item[2] for item in splitted_files]
        elif self.mode == "submission":
            part_info = [item[1] for item in splitted_files]
            log_info = [0 for item in splitted_files]
            date_info = [item[2] for item in splitted_files]
        else:
            raise RuntimeError("Invalid mode! Choose from 'train' or 'test'")          
        self.log_summary = pd.DataFrame({
            "date":date_info,
            "type":log_info,
            "part":part_info,
            "file":files_in_dir
        })
        self.log_summary["date"] = pd.to_datetime(self.log_summary["date"])
        self.log_summary["type"] = self.log_summary["type"].astype("int64")
        self.log_summary = self.log_summary.sort_values(["date","type","part"]).reset_index(drop=True)
        self.print_info(self.log_summary)
        
    def print_info(self, df):
        print("number of files: %i" % len(df))
        print("file parts:\n%s" % df["part"].value_counts())
        print("date min: %s" % df["date"].min(), "date max: %s" % df["date"].max())
        print("type min: %s" % df["type"].min(), "type max: %s" % df["type"].max())
        
    def load_files(self, date_min="2018-07-15", date_max="2018-09-18", log_types=None, features_dir=None):
        dt_min = pd.to_datetime(date_min)
        dt_max = pd.to_datetime(date_max)
        selected_df = self.log_summary[(self.log_summary["date"] >= dt_min) & (self.log_summary["date"] <= dt_max)]
        if log_types != None:
            selected_df = selected_df[selected_df["type"].isin(log_types)]
        print("INFO:")
        self.print_info(selected_df)
        idx = 0
        print("Loading files STARTED..")
        for _, row in selected_df.iterrows():
            f, log_type, date, = row["file"], str(row["type"]), str(row["date"]).split(" ")[0]
            if features_dir == None or "prehistory" in f:
                data_part = tc.SFrame(pd.read_pickle("%s/%s" % (self.folder,f)))
            else:
                if self.mode == "train":
                    binary_path = "%s/binary_pattern__%s_%s_000000000000.pkl.gz" % (features_dir, log_type, date.replace("-",""))
                    ternary_path = "%s/ternary_pattern__%s_%s_000000000000.pkl.gz" % (features_dir, log_type, date.replace("-",""))
                    autoenc_path = "%s/autoencoder_1__%s_%s_000000000000.pkl.gz" % (features_dir, log_type, date.replace("-",""))
                elif self.mode == "test":
                    binary_path = "%s/binary_pattern_%s_%s_000000000000.pkl.gz" % (features_dir, log_type, date.replace("-",""))
                    ternary_path = "%s/ternary_pattern_%s_%s_000000000000.pkl.gz" % (features_dir, log_type, date.replace("-",""))
                    autoenc_path = "%s/autoencoder_1_%s_%s_000000000000.pkl.gz" % (features_dir, log_type, date.replace("-",""))
                elif self.mode == "submission":
                    binary_path = "%s/binary_pattern_%s_000000000000.pkl.gz" % (features_dir, date.replace("-",""))
                    ternary_path = "%s/ternary_pattern_%s_000000000000.pkl.gz" % (features_dir, date.replace("-",""))
                    autoenc_path = "%s/autoencoder_1_%s_000000000000.pkl.gz" % (features_dir, date.replace("-",""))
                else:
                    raise RuntimeError("Invalid more with additional features!!!")
                data_part = join_features("%s/%s" % (self.folder,f), binary_path, ternary_path, autoenc_path)
            if idx == 0:
                loaded_data = data_part
            else:
                loaded_data = concatenate(loaded_data,data_part)
            idx += 1
            print(idx, len(loaded_data))
        print("Loading files FINISHED")
        return loaded_data

### Training ###

def select_rnd_sessions(data, id_col, k, seed=None):
    uniq_ids = np.unique(np.array(data[id_col]))
    print("Number of unique sessions: %i" % len(uniq_ids))
    np.random.seed(seed=seed)
    selected_sessions = list(np.random.choice(uniq_ids, k, replace=False))
    print("Number of selected sessions: %i" % len(selected_sessions))
    selected_sf = data.filter_by(selected_sessions, id_col)
    print("Number of selected records (session tracks): %i" % len(selected_sf))
    return selected_sf

def train_test_split(selected_sf, id_col, test_ratio):
    codes = np.array(selected_sf[id_col])
    selected_sf.remove_column(id_col, inplace=True)
    if test_ratio == 1.0:
        return None, None, selected_sf, codes
    else:
        idx = int(len(selected_sf)*(1.0-test_ratio))
        while codes[idx-1]==codes[idx]:
            # do not split sessions in half
            idx+=1
        tr_codes = codes[:idx]
        te_codes = codes[idx:]
        tr_sf = selected_sf[:idx]
        te_sf = selected_sf[idx:]
        print(tr_sf.shape, te_sf.shape)
        return tr_sf, tr_codes, te_sf, te_codes
    
class ModelTrainerSFrame():
    def __init__(self, data, label_col="target", id_col="session_code", seed=None):
        """pandas.DataFrame and turicreate.SFrame can also be provided as 'data'"""
        self.label_col = label_col
        self.id_col = id_col
        self.data = data
        self.seed = seed
    
    def train(self, model_param_dict={"max_iterations":2, "max_depth":3, "validation_set":None}, columns=None, k=None, test_ratio=0.4):
        if columns == None:
            columns = self.data.column_names()
        else:
            if not self.id_col in columns:
                columns.append(self.id_col)
            if not self.label_col in columns:
                columns.append(self.label_col)
        if k == None:
            selected_data = self.data[columns]
        else:
            selected_data = select_rnd_sessions(self.data[columns], self.id_col, k, seed=self.seed)
        tr_sf, tr_codes, te_sf, te_codes = train_test_split(selected_data, self.id_col, test_ratio)
        print("Train-test split DONE")
        model_param_dict.update({"dataset":tr_sf, "target":self.label_col})
        model = tc.boosted_trees_classifier.create(**model_param_dict)
        return {
            "model": model,
            "train_features": tr_sf,
            "train_session_codes": tr_codes,
            "test_features": te_sf,
            "test_session_codes": te_codes
        }

def sframe_important_feats(model):
    imp_sf = model.get_feature_importance()
    return imp_sf[imp_sf["count"]>0]
    
def sframe_model_predict(model, features, codes, label_col):
    pred = model.classify(features)
    pred["session_code"] = codes
    pred["session_position"] = features["session_position"]
    col_order = ["session_code","session_position","prediction","probability"]
    if label_col != None:
        pred["skip"] = features[label_col]
        col_order.append("skip")
    return pred.rename({"class":"prediction"})[col_order]
    
### Evaluation ###

def meanap_np(skips, preds, groups):
    group_indices = np.where(np.ediff1d(groups))[0]+1
    is_correct_pred = np.split((preds==skips).astype(np.uint8), group_indices)
    results = [np.mean((np.cumsum(a)*a)/(np.arange(len(a))+1)) for a in is_correct_pred]
    return results

def meanap_np_tup(tup):
    return meanap_np(*tup)

def meanap_mt(skips, preds, groups, threads):
    datalen = len(skips)
    proposed_splits = [i*(datalen//threads) for i in range(1,threads)]
    for i in range(len(proposed_splits)):
        while groups[proposed_splits[i]-1]==groups[proposed_splits[i]]:
            proposed_splits[i]+=1

    inputs = list(zip(
        np.split(skips, proposed_splits),
        np.split(preds, proposed_splits),
        np.split(groups, proposed_splits)
    ))
    import concurrent.futures
    executor = concurrent.futures.ProcessPoolExecutor(threads)
    res = np.concatenate(list(executor.map(meanap_np_tup, inputs)))
    executor.shutdown()
    return res