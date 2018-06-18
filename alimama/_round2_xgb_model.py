# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import pickle
import os
import gc
from _round2_lgb_model import concat_data
import warnings
warnings.filterwarnings("ignore")


def ceate_feature_map(features):
    with open('./tmp/xgb.fmap', 'w') as f:
        for i,feat in enumerate(features):
            f.write('{0}\t{1}\tq\n'.format(i, feat))

def xgb_predict(training,label,predict):
    drop_list = ['instance_id','context_hour','index','context_day','city_delivery_max',\
                'city_positive_max','item_price_level_std','user_gender_id','user_id',\
                'item_price_level_diff_std','item_price_level_diff_sum','item_price_level_diff_mean',\
                'item_price_level_diff_var']
    col = [c for c in training.columns if c not in drop_list]
    training = training[col]
    test = predict[col]
    dtest = xgb.DMatrix(test)

    X_train, X_val, y_train, y_val = train_test_split(training, label, test_size=0.1, random_state=42)

    dtrain = xgb.DMatrix(X_train,y_train)
    dval = xgb.DMatrix(X_val,y_val)
    watchlist = [(dtrain,'train'),(dval,'val')]

    params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'gamma': 0.6,
        'max_depth': 8,
        'lambda': 1,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'min_child_weight': 3,
        'silent': 1,
        'eta': 0.05,
        'seed': 25,
        'eval_metric':'logloss',
        'nthread': -1,
        }

    num_rounds = 300
    plst = params.items()
    clf = xgb.train(plst, dtrain, num_rounds, evals=watchlist, early_stopping_rounds=20)

    ceate_feature_map(X_train.columns.tolist())
    importance = clf.get_fscore(fmap='./tmp/xgb.fmap')
    importance = sorted(importance.items(), key=lambda d : d[1],reverse = True)
    feat_imp = pd.DataFrame(importance, columns=['feature', 'fscore'])
    feat_imp['fscore_rate'] = feat_imp['fscore'] / feat_imp['fscore'].sum()
    feat_imp.to_csv("tmp/xgb_importance.txt", sep=' ',index=False,header=True)

    pred = clf.predict(dtest)
    predict['predicted_score'] = pred
    predict = predict[['instance_id', 'predicted_score']]

    sub = pd.read_csv("data/round2_ijcai_18_test_b_20180510.txt", sep = "\s+")
    sub = sub[['instance_id','item_id']]
    sub = pd.merge(sub,predict, on = 'instance_id', how = 'left')
    sub = sub.fillna(0)

    print(sub.shape)
    sub[['instance_id', 'predicted_score']].to_csv('sub/xgb_b1.txt',sep=" ",index=False, float_format='%.8f')
    print(sub['predicted_score'].describe())



if __name__ == '__main__':
    training,label,predict = concat_data()
    xgb_predict(training,label,predict)




