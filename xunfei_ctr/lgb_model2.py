
import datetime
from sklearn.feature_selection import chi2, SelectPercentile
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from data_helper import remove_lowcase, load_and_clearn_data
from scipy import sparse
import lightgbm as lgb
import warnings
import time
import pandas as pd
import numpy as np
import os
import re

def make_features():

    data,trn_num,te_num = load_and_clearn_data()
    data['advert_industry_inner_1'] = data['advert_industry_inner'].apply(lambda x: x.split('_')[0])
    ad_cate_feature = ['adid', 'advert_id', 'orderid', 'advert_industry_inner_1',
                        'advert_industry_inner','campaign_id',
                        'creative_id', 'creative_type', 'creative_tp_dnf','creative_has_deeplink',
                        'creative_is_jump', 'creative_is_download']
    media_cate_feature = ['app_cate_id', 'f_channel', 'app_id', 'inner_slot_id']
    content_cate_feature = ['city', 'carrier', 'province', 'nnt', 'devtype', 'osv', 'os', 'make', 'model']
    origin_cate_list = ad_cate_feature + media_cate_feature + content_cate_feature
    # 编码，加速
    for i in origin_cate_list:
        data[i] = data[i].map(
            dict(zip(data[i].unique(), range(0, data[i].nunique()))))
    count_feature_list = []

    def feature_count(data, features=[], is_feature=True):
        if len(set(features)) != len(features):
            print('equal feature !!!!')
            return data
        new_feature = 'count'
        nunique = []
        for i in features:
            nunique.append(data[i].nunique())
            new_feature += '_' + i.replace('add_', '')
        if len(features) > 1 and len(data[features].drop_duplicates()) <= np.max(nunique):
            print(new_feature, 'is unvalid cross feature:')
            return data
        temp = data.groupby(features).size().reset_index().rename(columns={0: new_feature})
        data = data.merge(temp, 'left', on=features)
        if is_feature:
            count_feature_list.append(new_feature)
        if 'day_' in new_feature:
            print('fix:', new_feature)
            data.loc[data.day == 3, new_feature] = data[data.day == 3][new_feature] * 4
        return data


    for i in origin_cate_list:
        n = data[i].nunique()
        if n > 5:
            data = feature_count(data, [i])
            data = feature_count(data, ['day', 'hour', i])

    ratio_feature_list = []
    for i in ['adid']:
        for j in content_cate_feature:
            data = feature_count(data, [i, j])
            if data[i].nunique() > 5 and data[j].nunique() > 5:
                data['ratio_'+j+'_of_'+i] = data['count_'+i+'_'+j] / data['count_'+i]
                data['ratio_'+i+'_of_'+j] = data['count_'+i+'_'+j] / data['count_'+j]
                ratio_feature_list.append('ratio_'+j+'_of_'+i)
                ratio_feature_list.append('ratio_'+i+'_of_'+j)

    for i in media_cate_feature:
        for j in content_cate_feature+ad_cate_feature:
            new_feature = 'inf_'+i+'_'+j
            data = feature_count(data, [i, j])
            if data[i].nunique() > 5 and data[j].nunique() > 5:
                data['ratio_'+j+'_of_'+i] = data['count_'+i+'_'+j] / data['count_'+i]
                data['ratio_'+i+'_of_'+j] = data['count_'+i+'_'+j] / data['count_'+j]
                ratio_feature_list.append('ratio_'+j+'_of_'+i)
                ratio_feature_list.append('ratio_'+i+'_of_'+j)

    cate_feature = origin_cate_list
    num_feature = ['creative_width', 'creative_height', 'hour'] + count_feature_list + ratio_feature_list
    feature = cate_feature + num_feature
    print(len(feature), feature)
    # 低频过滤
    for feature in cate_feature:
        if 'count_' + feature in data.keys():
            print(feature)
            data.loc[data['count_' + feature] < 2, feature] = -1
            data[feature] = data[feature] + 1

    predict = data[(data.label == -1) & (data.data_type == 2)]
    predict_result = predict[['instance_id']]
    predict_result['predicted_score'] = 0
    predict_x = predict.drop('label', axis=1)
    train_x = data[data.label != -1].reset_index(drop=True)
    train_y = train_x.pop('label').values
    base_train_csr = sparse.csr_matrix((len(train_x), 0))
    base_predict_csr = sparse.csr_matrix((len(predict_x), 0))

    ohc = OneHotEncoder()
    for feature in cate_feature:
        ohc.fit(data[feature].values.reshape(-1, 1))
        base_train_csr = sparse.hstack((base_train_csr, 
                                ohc.transform(train_x[feature].values.reshape(-1, 1))),
                                'csr','bool')
        base_predict_csr = sparse.hstack((base_predict_csr,
                                ohc.transform(predict[feature].values.reshape(-1, 1))),
                                'csr','bool')
    print('one-hot prepared !')

    cv = CountVectorizer(min_df=10)
    for feature in ['user_tags']:
        data[feature] = data[feature].astype(str)
        cv.fit(data[feature])
        base_train_csr = sparse.hstack((base_train_csr, 
                                        cv.transform(train_x[feature].astype(str))),
                                        'csr', 'bool')
        base_predict_csr = sparse.hstack((base_predict_csr,
                                        cv.transform(predict_x[feature].astype(str))),
                                        'csr','bool')
    print('cv prepared !')

    train_csr = sparse.hstack(
        (sparse.csr_matrix(train_x[num_feature]), base_train_csr),'csr').astype('float32')
    predict_csr = sparse.hstack(
        (sparse.csr_matrix(predict_x[num_feature]), base_predict_csr),'csr').astype('float32')

    feature_select = SelectPercentile(chi2, percentile=95)
    feature_select.fit(train_csr, train_y)
    train_csr = feature_select.transform(train_csr)
    predict_csr = feature_select.transform(predict_csr)
    print('feature select')
    print(train_csr.shape)

    return train_csr,predict_csr,train_y,predict_result

def lgbcv_predict(training,label,predict):

    clf = lgb.LGBMClassifier(
                boosting_type='gbdt', num_leaves=61, reg_alpha=3, reg_lambda=1,
                max_depth=-1, n_estimators=5000, objective='binary',
                subsample=0.8, colsample_bytree=0.8, subsample_freq=1,
                learning_rate=0.035, random_state=2018, n_jobs=-1)

    kf = KFold(n_splits=5,random_state=2018,shuffle=True)
    best_logloss = []
    pred_list = []
    for i,(train_index, val_index) in enumerate(kf.split(training)):
        t0 = time.time()
        X_train = training[train_index]
        y_train = label[train_index]
        X_val = training[val_index]
        y_val = label[val_index]

        clf.fit(X_train, y_train, eval_set=[(X_train, y_train),(X_val, y_val)],\
                eval_metric='logloss',early_stopping_rounds=50)

        predi = clf.predict_proba(test,num_iteration=clf.best_iteration_)[:,1]
        pred_list.append(predi)
        print(np.mean(predi), len(predi))
        best_logloss.append(clf.best_score_['valid_1']['binary_logloss'])
        gc.collect()
        t = time.time() - t0 // 60
        print(i,t)

    pred = np.mean(np.array(pred_list),axis=0)
    print(best_logloss,'\n',np.mean(best_logloss),np.std(best_logloss))

    return pred


if __name__ == '__main__':
    data_path = '../data/predict_result.pkl'
    if os.path.exists(data_path):
        label,predict_result = pickle.load(open(data_path,'rb'))
        train = sparse.load_npz('../data/lgb2_train_csr.npz').tocsr()
        test = sparse.load_npz('../data/lgb2_predict_csr.npz').tocsr()
    else:
        train,test,label,predict_result = make_features()

        sparse.save_npz('../data/lgb2_train_csr.npz', train_csr)
        sparse.save_npz('../data/lgb2_predict_csr.npz', predict_csr)
        pickle.dump((label,predict_result),open(data_path,'wb'))

    print(train.shape)
    pred = lgbcv_predict(train,label,test)
    predict_result['predicted_score'] = pred

    predict_result[['instance_id', 'predicted_score']].to_csv('../output/lgb2_res.csv',sep=",",
        index=False, float_format='%.8f')

    print(predict_result['predicted_score'].describe())
    print(predict_result['predicted_score'].mean())









