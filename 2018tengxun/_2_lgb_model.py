# -*- coding:utf-8 -*-

import pandas as pd
import lightgbm as lgb
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.model_selection import train_test_split
from scipy import sparse
import pickle
import os
import gc
import math
import numpy as np
import _1_gen_features
import warnings
warnings.filterwarnings("ignore")

def get_xtype_col(train_df,xtype):
    dtype_df = train_df.dtypes.reset_index()
    dtype_df.columns = ['col','type']
    return dtype_df[dtype_df.type==xtype].col.values.tolist()

def get_userfeat_data(path):
    df_user = pd.read_csv(path,header=0)
    return df_user

def concat_data(df_user,df_ad,data,gp):
    data = pd.merge(data,df_ad,on='aid',how='left')
    # flist = ['appIdAction', 'appIdInstall','LBS','age','consumptionAbility','education','gender','os','ct','marriageStatus',\
    #         'house','carrier']
    flist = ['appIdAction', 'appIdInstall']
    data = pd.merge(data,df_user.drop(flist,axis=1),on='uid',how='left')

    data = _1_gen_features.gen_combian_feat(data,gp)

    data = _1_gen_features.get_user_cv_feat(data)
    # print(data.columns.tolist())

    data = _1_gen_features.gen_userfeat(data,gp)
    # data=pd.merge(data,df_user_stas,on='uid',how='left')
    # print('merge susseed df_user_stas')
    # del df_user_stas
    # gc.collect()

    data = data.fillna(-1.)
    # df = pd.read_csv('./sub/train_feat_imp_%d.txt' % gp, sep=' ',header=None)
    # drop_feat = df.loc[df[1]<5,0].tolist()
    # drop_feat = [col for col in drop_feat if col in data.columns]
    # data.drop(drop_feat,axis=1,inplace=True)
    # gc.collect()
    # print('drop succeed')

    # int_col = get_xtype_col(data,'int')
    # for col in int_col:
    #     data[col] = data[col].astype(np.float16)

    # df_user_w2v = _1_gen_features.gen_w2v_feat(df_user)
    # data=pd.merge(data,df_user_w2v,on='uid',how='left')
    # print('merge susseed df_user_w2v')
    # del df_user_w2v
    # gc.collect()

    del data['uid']
    print(data.columns.tolist())
    training = data[data.label!=-1]  
    training = training.reset_index(drop=True)
    label = training['label'].values
    predict = data[data.label==-1]
    del training['label']
    del predict['label']
    del data 
    gc.collect()

    non_cv_col = training.columns.tolist()
    non_cv_col.remove('index')
    train_cv = training['index']
    predict_cv = predict['index']
    training=training[non_cv_col]
    predict=predict[non_cv_col]

    training = training.astype(np.float16)
    predict = predict.astype(np.float16)

    cv = CountVectorizer(ngram_range=(1, 1), max_df=0.8, min_df=5)
    train_cv = cv.fit_transform(train_cv)
    predict_cv = cv.transform(predict_cv)
    training = sparse.hstack((training, train_cv))
    predict = sparse.hstack((predict, predict_cv))

    del train_cv
    del predict_cv
    gc.collect()

    training = training.tocsr()
    predict = predict.tocsr()

    print("training, predict shape:",training.shape, predict.shape)

    return training,label,predict

def gen_feat(data):
    flist = ['appIdAction', 'appIdInstall']
    data.drop(flist,axis=1,inplace=True)
    data = _1_gen_features.get_user_cv_feat(data)
    data = _1_gen_features.gen_userfeat(data,1)
    data = data.fillna(-1.)

    del data['uid']
    print(data.columns.tolist())
    training = data[data.label!=-1]  
    training = training.reset_index(drop=True)
    label = training['label'].values
    predict = data[data.label==-1]
    del training['label']
    del predict['label']
    del data 
    gc.collect()

    non_cv_col = training.columns.tolist()
    non_cv_col.remove('index')
    train_cv = training['index']
    predict_cv = predict['index']
    training=training[non_cv_col]
    predict=predict[non_cv_col]

    training = training.astype(np.float16)
    predict = predict.astype(np.float16)

    cv = CountVectorizer(ngram_range=(1, 1), max_df=0.8, min_df=5)
    train_cv = cv.fit_transform(train_cv)
    predict_cv = cv.transform(predict_cv)
    training = sparse.hstack((training, train_cv))
    predict = sparse.hstack((predict, predict_cv))

    del train_cv
    del predict_cv
    gc.collect()

    training = training.tocsr()
    predict = predict.tocsr()

    print("training, predict shape:",training.shape, predict.shape)

    return training,label,predict

def lgb_predict(training,label,predict):
    # feature_list = training.columns.tolist()
    # training = training.values
    # predict = predict.values
    print(".....")

    clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=41, reg_alpha=0.0, reg_lambda=1.8,
        max_depth=-1, n_estimators=2000, objective='binary',
        subsample=0.6, colsample_bytree=0.7, subsample_freq=1,
        learning_rate=0.08, min_child_samples=50, random_state=42, n_jobs=-1
    )
    # X_train, X_val, y_train, y_val = train_test_split(training, label, test_size=0.1, random_state=42)
    shuffle_indices = np.random.permutation(np.arange(len(label)))
    training = training[shuffle_indices]
    label = label[shuffle_indices]
    train_num = int(0.95*len(label))
    X_train, X_val = training[:train_num],training[train_num:]
    y_train, y_val = label[:train_num],label[train_num:]
    print("spilt done")
    del training
    del label
    gc.collect()

    clf.fit(X_train, y_train, eval_set=[(X_train, y_train),(X_val, y_val)],\
            eval_metric='auc',early_stopping_rounds=50)

    # feat_imp = pd.Series(clf.feature_importance(), feature_list).sort_values(ascending=False)
    # feat_imp.to_csv("sub/feat_imp_%d.txt" % gp, sep=' ',index=True,header=False)

    pred = clf.predict_proba(predict)[:,1]
    best_auc = clf.best_score_['valid_1']['auc']

    return pred,best_auc


def lgbcv_predict(training,label,predict,gp):

    clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=51, reg_alpha=0.0, reg_lambda=5.5,
        max_depth=-1, n_estimators=3000, objective='binary',
        subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
        learning_rate=0.035, min_child_samples=50, random_state=142, n_jobs=-1
    )

    kf = KFold(n_splits = 5,random_state=2017,shuffle=True)
    best_auc = []
    pred_list = []
    for train_index, val_index in kf.split(training):
        # X_train = training.loc[train_index,:]
        X_train = training[train_index]
        y_train = label[train_index]
        # X_val = training.loc[val_index,:]
        X_val = training[val_index]
        y_val = label[val_index]

        clf.fit(X_train, y_train, eval_set=[(X_train, y_train),(X_val, y_val)],\
                eval_metric='auc',early_stopping_rounds=50)
        
        # predictors = [i for i in X_train.columns]
        # feat_imp = pd.Series(clf.feature_importance(), predictors).sort_values(ascending=False)
        # feat_imp.to_csv("sub/feat_imp_%d.txt" % gp, sep=' ',index=True,header=False)

        predi = clf.predict_proba(predict)[:,1]
        pred_list.append(predi)
        best_auc.append(clf.best_score_['valid_1']['auc'])
        gc.collect()

    pred = np.mean(np.array(pred_list),axis=0)
    best_auc = np.mean(best_auc)

    return pred,best_auc

def main():
    train = pd.read_csv('./data/final_competition_data/train.csv')
    test = pd.read_csv('./data/final_competition_data/test1.csv')
    df_ad = pd.read_csv('./data/final_competition_data/adFeature.csv')

    sub = None
    auc_list = []
    aid_group = pickle.load(open('./data/aid_group.pkl','rb'))
    for gp in aid_group.keys():
        if gp >= 1:
            aid_list = aid_group[gp]

            userFeat_path = './data/final_competition_data/userFeature_%d.txt' % gp
            df_user = get_userfeat_data(userFeat_path)

            # aid_list = [692,1119]
            train_gp = train[train['aid'].isin(aid_list)]
            test_gp = test[test['aid'].isin(aid_list)]
            df_ad_gp = df_ad[df_ad['aid'].isin(aid_list)]

            train_gp.loc[train_gp['label']==-1,'label']=0
            train_gp = train_gp.sample(frac=0.65,random_state=1)
            test_gp['label']=-1
            data = pd.concat([train_gp,test_gp],axis=0)

            training,label,predict = concat_data(df_user,df_ad_gp,data,gp)

            pred,auc = lgbcv_predict(training,label,predict,gp)
            del training
            del predict
            del label
            gc.collect()

            test_gp['score'] = pred
            # test_gp.to_csv('./sub/submission%d.csv' % gp,header=True,index=False)
            sub = pd.concat([sub,test_gp],axis=0)
            auc_list.append(auc)

    sub = pd.merge(test,sub,how='left',on=['aid','uid'])
    del sub['label']
    sub.fillna(0,inplace=True)
    sub.to_csv('./sub/submission.csv',header=True,index=False, float_format='%.6f')

    print("Best auc of 9 aid groups: ",auc_list)
    print("Mean auc of validation: ",np.mean(auc_list))

def main2():
    sub = None
    auc_list = []

    for i in range(3,0,-1):
        train = pd.read_csv('./data/combine/merge_train%s.txt'%i)
        train.sample(frac=1).reset_index(drop=True)

        test = pd.read_csv('./data/combine/merge_test%s.txt'%i)
        sub_i = test[['aid','uid']].copy()
        train = pd.concat([train,test],axis=0)

        train,label,test = gen_feat(train)

        pred,auc = lgb_predict(train,label,test)

        sub_i['score'] = pred
        sub_i.to_csv('./sub/lgb/submission%d.csv' % i,header=True,index=False,float_format='%.6f')
        sub = pd.concat([sub,sub_i],axis=0)
        auc_list.append(auc)
        print(sub_i.describe())

    sub.fillna(0,inplace=True)
    sub.to_csv('./sub/submission.csv',header=True,index=False, float_format='%.6f')

    print("Best auc of 12 aid groups: ",auc_list)
    print("Mean auc of validation: ",np.mean(auc_list))


if __name__ == '__main__':
    main2()





