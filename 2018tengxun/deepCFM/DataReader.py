"""
A data parser for Porto Seguro's Safe Driver Prediction competition's dataset.
URL: https://www.kaggle.com/c/porto-seguro-safe-driver-prediction
"""
import pandas as pd
import numpy as np
import config

class FeatureDictionary(object):
    def __init__(self, trainfile=None, testfile=None,
                 dfTrain=None, dfTest=None, numeric_cols=[], ignore_cols=[],multi_value_cols=[]):
        assert not ((trainfile is None) and (dfTrain is None)), "trainfile or dfTrain at least one is set"
        assert not ((trainfile is not None) and (dfTrain is not None)), "only one can be set"
        assert not ((testfile is None) and (dfTest is None)), "testfile or dfTest at least one is set"
        assert not ((testfile is not None) and (dfTest is not None)), "only one can be set"
        self.trainfile = trainfile
        self.testfile = testfile
        self.dfTrain = dfTrain
        self.dfTest = dfTest
        self.numeric_cols = numeric_cols
        self.ignore_cols = ignore_cols
        self.multi_value_cols = multi_value_cols
        self.gen_feat_dict()

    def gen_feat_dict(self):
        if self.dfTrain is None:
            dfTrain = pd.read_csv(self.trainfile)
        else:
            dfTrain = self.dfTrain
        if self.dfTest is None:
            dfTest = pd.read_csv(self.testfile)
        else:
            dfTest = self.dfTest
        df = pd.concat([dfTrain, dfTest])
        self.feat_dict = {}
        tc = 0
        for col in df.columns:
            if col in self.ignore_cols or col in self.multi_value_cols:
                continue
            if col in self.numeric_cols:
                # map to a single index
                self.feat_dict[col] = tc
                tc += 1
            else:
                if col in ['os','ct','marriageStatus']:
                    df[col] = df[col].map(lambda x : ' '.join(sorted(x.split())))
                us = df[col].unique()
                self.feat_dict[col] = dict(zip(us, range(tc, len(us)+tc)))
                tc += len(us)
        self.feat_dim = tc

        self.multi_feat_dict = {}
        for i,col in enumerate(self.multi_value_cols):
            for feat_flied in df[col].tolist():
                list_flied = str(feat_flied).split()
                list_flied= ['%s_'%i+str(v) for v in list_flied]
                for v in list_flied:
                    if v not in self.multi_feat_dict.keys():
                        self.multi_feat_dict[v] = 1
                    else:
                        self.multi_feat_dict[v] += 1
        # print(len(self.multi_feat_dict))
        self.multi_feat_dict = [v for v in self.multi_feat_dict.keys() if self.multi_feat_dict[v]>100]
        # self.multi_feat_dict = [v for v in self.multi_feat_dict.keys()]

        print(len(self.multi_feat_dict))

        self.multi_feat_dict.append('UNK')
        self.multi_feat_dict.insert(0,'0')
        self.vocab_size = len(self.multi_feat_dict)
        self.multi_feat_dict = {v:idx for idx,v in enumerate(self.multi_feat_dict)}


class DataParser(object):
    def __init__(self, feat_dict):
        self.feat_dict = feat_dict

    def parse(self, infile=None, df=None, has_label=False):
        assert not ((infile is None) and (df is None)), "infile or df at least one is set"
        assert not ((infile is not None) and (df is not None)), "only one can be set"
        if infile is None:
            dfi = df.copy()
        else:
            dfi = pd.read_csv(infile)
        if has_label:
            # dfi.loc[dfi["label"]==-1,'label'] = 0
            y = dfi["label"].values.tolist()
            dfi.drop(["uid", "label"], axis=1, inplace=True)
        else:
            ids = dfi[["aid","uid"]]
            dfi.drop(["uid"], axis=1, inplace=True)

        dfi = dfi.drop(self.feat_dict.multi_value_cols, axis=1)
        # dfi for feature index
        # dfv for feature value which can be either binary (1/0) or float (e.g., 10.24)
        dfv = dfi.copy()
        for col in dfi.columns:
            if col in self.feat_dict.ignore_cols:
                dfi.drop(col, axis=1, inplace=True)
                dfv.drop(col, axis=1, inplace=True)
                continue
            if col in self.feat_dict.numeric_cols:
                dfi[col] = self.feat_dict.feat_dict[col]
            else:
                if col in ['os','ct','marriageStatus']:
                    dfi[col] = dfi[col].map(lambda x : ' '.join(sorted(x.split())))
                dfi[col] = dfi[col].map(self.feat_dict.feat_dict[col])
                dfv[col] = 1.

        # list of list of feature indices of each sample in the dataset
        # print(dfi.info())
        dfi.fillna(0,inplace=True)
        Xi = dfi.values.tolist()
        # list of list of feature values of each sample in the dataset
        # print(dfv.info())
        dfv.fillna(0,inplace=True)
        Xv = dfv.values.tolist()

        multi_value_martix = []
        for i in range(df.shape[0]):
            sample_v2idx = []
            for j,col in enumerate(self.feat_dict.multi_value_cols):
                v2idx = self.map_v2idx(col,str(df.loc[i,col]),j)
                sample_v2idx.append(v2idx)
            multi_value_martix.append(sample_v2idx)
        multi_value_martix = np.array(multi_value_martix)

        if has_label:
            return Xi, Xv, multi_value_martix, y
        else:
            return Xi, Xv, multi_value_martix, ids

    def map_v2idx(self,col,feat_col,j):
        seq_length = config.MAXLEN
        v2idx = np.zeros(seq_length,dtype=np.int32)
        feat_col = sorted(feat_col.split()[:seq_length])
        feat_col= ['%s_'%j +str(v) for v in feat_col]
        for i,v in enumerate(feat_col):
            try:
                v2idx[i] = self.feat_dict.multi_feat_dict[v]
            except KeyError:
                v2idx[i] = self.feat_dict.multi_feat_dict['UNK']
        return v2idx

if __name__ == '__main__':
    dfTrain = pd.read_csv(config.TRAIN_FILE)
    print(dfTrain.columns.tolist())
    dfTest = pd.read_csv(config.TEST_FILE)
    dfTrain['house'].fillna(0,inplace=True)
    dfTest['house'].fillna(0,inplace=True)
    dfTrain.fillna('',inplace=True)
    dfTest.fillna('',inplace=True)

    fd = FeatureDictionary(dfTrain=dfTrain, dfTest=dfTest,
                           numeric_cols=config.NUMERIC_COLS,
                           ignore_cols=config.IGNORE_COLS,
                           multi_value_cols=config.MULTI_VALUE_COLS)

    data_parser = DataParser(feat_dict=fd)
    Xi_train, Xv_train, mvm_train ,y_train = data_parser.parse(df=dfTrain, has_label=True)

    print(mvm_train[:5])
    print(mvm_train.shape)
    print(mvm_train[:,3,:][:5])


    dfm_params = {}
    dfm_params["feature_size"] = fd.feat_dim
    dfm_params["field_size"] = len(Xi_train[0])
    dfm_params["vocab_size"] = fd.vocab_size
    dfm_params["num_multiVal_feat"] = len(fd.multi_value_cols)
    dfm_params["sequence_length"] = 5
    print(dfm_params)


