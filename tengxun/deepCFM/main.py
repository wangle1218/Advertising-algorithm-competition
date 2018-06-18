
import os
import sys
import gc
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.preprocessing import LabelEncoder

import config
from metrics import gini_norm
from DataReader import FeatureDictionary, DataParser
# sys.path.append("..")
from DeepCFM import DeepCFM

gini_scorer = make_scorer(gini_norm, greater_is_better=True, needs_proba=True)


def _load_data():

    dfTrain = pd.read_csv(config.TRAIN_FILE)
    dfTest = pd.read_csv(config.TEST_FILE)
    # dfTrain['label'].replace(-1,0,inplace=True)
    # dfTest['label'] = -1

    # dfTrain = pd.concat([dfTrain,dfTest])
    dfTrain['house'].fillna(0,inplace=True)
    dfTrain['LBS'].fillna(-1,inplace=True)
    dfTrain['LBS'] = dfTrain['LBS'].map(lambda x : int(x))
    dfTrain.fillna('',inplace=True)

    dfTest['house'].fillna(0,inplace=True)
    dfTest['LBS'].fillna(-1,inplace=True)
    dfTest['LBS'] = dfTest['LBS'].map(lambda x : int(x))
    dfTest.fillna('',inplace=True)

    return dfTrain ,dfTest

def remove_lowcase(se):
    count = dict(se.value_counts())
    se = se.map(lambda x : -1 if count[x]<10 else x)
    return se

def static_feat(df,df_val, feature):
    df = df.groupby(feature)['label'].agg(['sum','count']).reset_index()

    new_feat_name = feature + '_stas'
    df.loc[:,new_feat_name] = 100 * (df['sum'] + 0.001) / (df['count'] + 0.001)
    df.loc[:,new_feat_name] = df.loc[:,new_feat_name].astype(np.float16)
    df_stas = df[[feature,new_feat_name]]
    df_val = pd.merge(df_val, df_stas, how='left', on=feature)

    return df_val

def gen_combian_feat(df):
    """
    ['aid','campaignId']
    ['age','gender','education','consumptionAbility','LBS']
    """
    lbl = LabelEncoder()
    ad_feat_list = ['advertiserId','campaignId','adCategoryId']
    user_feat_list = ['LBS','age','consumptionAbility','education','gender']
    df['index2'] = list(range(df.shape[0]))
    for afeat in ad_feat_list:
        for ufeat in user_feat_list:
            concat_feat = afeat + '_' + ufeat
            df[concat_feat] = df['aid'].astype('str') + df[afeat].astype('str') + '_' +df[ufeat].astype('str')
            df[concat_feat] = lbl.fit_transform(df[concat_feat])
            df.loc[:,concat_feat] = df.loc[:,concat_feat].astype(np.int32)
            df[concat_feat] = remove_lowcase(df[concat_feat])

            data = df[['index2',concat_feat,'label']].copy()

            training = data[data.label!=-1]  
            training = training.reset_index(drop=True)
            predict = data[data.label==-1]
            del data
            gc.collect()

            df_stas_feat = None
            kf = KFold(n_splits = 3,random_state=2018,shuffle=True)
            for train_index, val_index in kf.split(training):
                X_train = training.loc[train_index,:]
                X_val = training.loc[val_index,:]

                X_val = static_feat(X_train,X_val, concat_feat)
                df_stas_feat = pd.concat([df_stas_feat,X_val],axis=0)

            X_pred = static_feat(training, predict, concat_feat)
            df_stas_feat = pd.concat([df_stas_feat,X_pred],axis=0)

            del df_stas_feat['label']
            del df_stas_feat[concat_feat]
            del df[concat_feat]
            del training
            del predict
            gc.collect()

            df = pd.merge(df, df_stas_feat,how='left',on='index2')
            print(afeat, ufeat,'done!')

    del df['index2']
    gc.collect()
    training = df[df.label!=-1]  
    training = training.reset_index(drop=True)
    predict = df[df.label==-1]
    predict = predict.reset_index(drop=True)
    del predict['label']

    return training, predict

def _run_base_model_dfm(dfTrain, dfTest, folds, dfm_params):
    fd = FeatureDictionary(dfTrain=dfTrain, dfTest=dfTest,
                           numeric_cols=config.NUMERIC_COLS,
                           ignore_cols=config.IGNORE_COLS,
                           multi_value_cols=config.MULTI_VALUE_COLS)
    data_parser = DataParser(feat_dict=fd)
    Xi_train, Xv_train,  Xmv_train, y_train = data_parser.parse(df=dfTrain, has_label=True)
    Xi_test, Xv_test, Xmv_test, ids_test = data_parser.parse(df=dfTest)

    # pickle.dump((Xi_train, Xv_train,  Xmv_train, y_train), open('./data/train_set1.pkl','wb'))
    # pickle.dump((Xi_test, Xv_test, Xmv_test, ids_test), open('./data/test_set1.pkl','wb'))
    # Xi_train, Xv_train,  Xmv_train, y_train = pickle.load(open('./data/train_set1.pkl','rb'))
    # y_train = np.array(y_train)
    # y_train = np.where(y_train<0,0,y_train)
    # y_train = list(y_train)
    # Xi_test, Xv_test, Xmv_test, ids_test = pickle.load(open('./data/test_set1.pkl','rb'))

    dfm_params["feature_size"] = fd.feat_dim
    dfm_params["field_size"] = len(Xi_train[0])
    dfm_params["vocab_size"] = fd.vocab_size
    dfm_params["num_multiVal_feat"] = len(fd.multi_value_cols)
    dfm_params["sequence_length"] = config.MAXLEN
    print(dfm_params)
    del fd
    del data_parser
    gc.collect()

    y_test_meta = np.zeros((dfTest.shape[0], 1), dtype=float)
    _get = lambda x, l: [x[i] for i in l]

    for i, (train_idx, valid_idx) in enumerate(folds):
        Xi_train_, Xv_train_, Xmv_train_, y_train_ = \
        _get(Xi_train, train_idx), _get(Xv_train, train_idx), _get(Xmv_train, train_idx), _get(y_train, train_idx)
        Xi_valid_, Xv_valid_, Xmv_valid_, y_valid_ = \
        _get(Xi_train, valid_idx), _get(Xv_train, valid_idx),_get(Xmv_train, valid_idx), _get(y_train, valid_idx)

        dfm = DeepCFM(**dfm_params)
        dfm.fit(Xi_train_, Xv_train_,Xmv_train_, y_train_, Xi_valid_, Xv_valid_,Xmv_valid_, y_valid_)

        y_test_meta[:,0] += dfm.predict(Xi_test, Xv_test, Xmv_test)

        break

    # y_test_meta /= float(len(folds))

    # save result
    _make_submission(ids_test, y_test_meta, "submission1.csv")

    # _plot_fig(gini_results_epoch_train, gini_results_epoch_valid, clf_str)

    return y_test_meta


def _make_submission(ids, y_pred, filename="submission.csv"):
    ids['score'] = y_pred.flatten()
    print(ids.describe())
    ids.to_csv(os.path.join(config.SUB_DIR, filename), index=False, header=True ,float_format="%.6f")


def _plot_fig(train_results, valid_results, model_name):
    colors = ["red", "blue", "green"]
    xs = np.arange(1, train_results.shape[1]+1)
    plt.figure()
    legends = []
    for i in range(train_results.shape[0]):
        plt.plot(xs, train_results[i], color=colors[i], linestyle="solid", marker="o")
        plt.plot(xs, valid_results[i], color=colors[i], linestyle="dashed", marker="o")
        legends.append("train-%d"%(i+1))
        legends.append("valid-%d"%(i+1))
    plt.xlabel("Epoch")
    plt.ylabel("Normalized Gini")
    plt.title("%s"%model_name)
    plt.legend(legends)
    plt.savefig("./fig/%s.png"%model_name)
    plt.close()


if __name__ == '__main__':
    # load data
    dfTrain,dfTest = _load_data()

    # dfTrain, dfTest = gen_combian_feat(dfTrain)

    # folds
    folds = list(StratifiedKFold(n_splits=config.NUM_SPLITS, shuffle=True,
                                 random_state=config.RANDOM_SEED).split(dfTrain.values, dfTrain['label'].values))


    # ------------------ DeepCFM Model ------------------
    # params
    dfm_params = {
        "use_fm": True,
        "use_deep": True,
        "embedding_size": 8,
        "cnn_embedding_size":40,
        "fc_sizes" : [128, 128],
        "filter_sizes" : [[1,2,3,4,5],[3,4,5]], 
        "num_filters" : 50,
        "dropout_fm": [1.0, 1.0],
        "deep_layers": [128, 128],
        "dropout_deep": [0.5, 0.5, 0.5],
        "deep_layers_activation": tf.nn.relu,
        "epoch": 2,
        "batch_size": 2048,
        "learning_rate": 0.0005,
        "optimizer_type": "adam",
        "batch_norm": 1,
        "batch_norm_decay": 0.995,
        "l2_reg": 0.0001,
        "verbose": True,
        # "eval_metric": gini_norm,
        "random_seed": config.RANDOM_SEED
    }

    y_test_dfm = _run_base_model_dfm(dfTrain, dfTest, folds, dfm_params)




