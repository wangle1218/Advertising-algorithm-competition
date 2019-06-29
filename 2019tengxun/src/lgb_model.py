# -*- coding:utf-8 -*-

import pandas as pd
import lightgbm as lgb
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.model_selection import train_test_split
from scipy import sparse
import pickle
import os,sys
import gc
import math
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def cal_𝐒𝐌𝐀𝐏𝐄(label,pred):
    pred = np.where(pred<0,0,pred)
    pred = np.expm1(pred)
    label = np.expm1(label)

    total_smape = 0.
    for i in range(len(label)):
        if label[i]+pred[i] == 0:
            smape = 0.
        else:
            smape = np.abs(label[i]-pred[i]) / ((label[i]+pred[i])/2)
        total_smape += smape

    smape = total_smape / len(label)
    smape = (1 - smape /2)
    return smape

def smape(pred,label):
    pred = np.where(pred<0,0,pred)
    pred = np.expm1(pred)
    pred = np.where(pred<1,1,pred)
    label = np.expm1(label.get_label())

    total_smape = np.abs(label-pred) / ((label+pred)/2)
    return 'smape', np.mean(total_smape), False

def lgb_model(train,label,test):
    feature_names = train.columns.tolist()
    train = train.values
    test = test.values
    label = np.log1p(label)
    kf = KFold(n_splits=10,random_state=24,shuffle=True)
    best_score = []
    pred_list = []
    for i,(train_index, val_index) in enumerate(kf.split(train)):
        X_train = train[train_index]
        y_train = label[train_index]
        X_val = train[val_index]
        y_val = label[val_index]

        lgb_train = lgb.Dataset(train,label=label)
        lgb_eval = lgb.Dataset(X_val,label=y_val, reference=lgb_train)
        seed = 2014 + i*12

        params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': {'mae'},
            'max_depth':-1,
            'num_leaves':31,
            'min_data_in_leaf':50,
            'learning_rate': 0.025,
            'lambda_l1':6.5,
            'lambda_l2':1.8,
            # 'max_bin': 75000,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'verbosity': 0,
            'seed':seed
        }

        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=10000,)
                        # valid_sets=(lgb_train,lgb_eval),
                        # # feval=smape,
                        # early_stopping_rounds=100)
        y_val_pred = gbm.predict(X_val, num_iteration=gbm.best_iteration)
        score = cal_𝐒𝐌𝐀𝐏𝐄(y_val, y_val_pred)
        print("*"*20,'\n',score,best_score,'\n','*'*20)
        imp = pd.DataFrame({'column': feature_names,
                            'importance': gbm.feature_importance(),
                        }).sort_values(by='importance')
        imp.to_csv('../tmp_data/imp_%d.csv' % i, index=False)
        # if score < 0.48:
        #     continue
        best_score.append(score)
        y_pred = gbm.predict(test, num_iteration=gbm.best_iteration)
        y_pred = np.where(y_pred<0,0,y_pred)
        pred_list.append(np.expm1(y_pred))
        # pred_list.append(y_pred)

    print(best_score, '\n',np.mean(best_score))

    return np.mean(np.array(pred_list),axis=0)


if __name__ == '__main__':
    test_id,label,train_num = pickle.load(open('../tmp_data/label.pkl','rb'))
    train = pickle.load(open('../tmp_data/lgb_train.pkl','rb'))
    test = pickle.load(open('../tmp_data/lgb_test.pkl','rb'))

    if sys.argv[1] == 'partdata':
        del train['广告id']
        del test['广告id']
    
    print(train.shape)

    pred = lgb_model(train,label,test)
    pred = np.where(pred<0,0,pred)

    test_id['预估日曝光'] = pred
    # test_id.loc[(test_id['预估日曝光']<1.3)&(test_id['预估日曝光']>1),'预估日曝光'] = 0.95
    # test_id['预估日曝光'] = test_id['预估日曝光'] + test_id['出价']**0.5 / 1000
    sub = test_id.sort_values(by=['广告id','出价'])
    test_id = test_id.sort_values(by=['广告id','预估日曝光'])

    sub['预估日曝光'] = test_id['预估日曝光'].values 
    sub['预估日曝光'] =sub['预估日曝光'].map(lambda x: np.around(x,4))
    sub.sort_values(by=['样本id'],inplace=True)
    sub[['样本id','预估日曝光']].to_csv('../tmp_data/submission-%s.csv' % sys.argv[1],index=False,header=None)
    print(sub.describe())

