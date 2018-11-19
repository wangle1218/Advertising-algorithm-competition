
import pandas as pd 
import numpy as np
import lightgbm as lgb
from scipy import sparse
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import chi2, SelectPercentile
from data_helper import remove_lowcase, load_and_clearn_data
import time
import datetime
import pickle
import gc
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

def make_features():
    data,trn_num,te_num = load_and_clearn_data()
    data['inner_slot_id_1'] = data['inner_slot_id'].map(lambda x : x.split('_')[0])
    data['advert_industry_inner_1'] = data['advert_industry_inner'].map(lambda x : x.split('_')[0])
    data['advert_industry_inner_2'] = data['advert_industry_inner'].map(lambda x : x.split('_')[1])
    data['ad_industry'] = data['advert_id'].astype('str') + '_' + data['advert_industry_inner_2'].astype('str')

    data['app_cate_id'] = data['app_cate_id'].astype(int)
    data['app_id'] = data['app_id'].astype(int)
    data['user'] = data['user_tags'].map(dict(zip(data['user_tags'].unique(),
                                            range(0, data['user_tags'].nunique()))))
    data['ad_area'] = data['creative_width'] * data['creative_height']
    data['ad_wh_ratio'] = data['creative_width'] / data['creative_height']

    id_col =['city','province','make','model','adid','advert_id','orderid',
            'advert_industry_inner','campaign_id','creative_id','creative_tp_dnf',
            'app_cate_id','f_channel','app_id','user','inner_slot_id',
            'inner_slot_id_1','advert_industry_inner_1','osv','advert_industry_inner_2',
            'ad_industry']

    lbl = LabelEncoder()
    for col in id_col:
        data[col] = lbl.fit_transform(data[col])
        data[col] = remove_lowcase(data[col]) 

    # 组合特征
    data['os_osv'] = data['os'].astype(str) + '_' + data['osv'].astype(str)
    data['os_osv'] = lbl.fit_transform(data['os_osv'])
    data['os_osv'] = remove_lowcase(data['os_osv'])

    data['adid_city'] = data['advert_id'].astype(str) + ' ' + data['city'].astype(str)
    data['adid_city'] = lbl.fit_transform(data['adid_city'])
    data['adid_city'] = remove_lowcase(data['adid_city'])

    data['adid_province'] = data['advert_id'].astype(str) + ' ' + data['province'].astype(str)
    data['adid_province'] = lbl.fit_transform(data['adid_province'])
    data['adid_province'] = remove_lowcase(data['adid_province'])

    # 广告主的每个广告投放比例
    advert_ad_cnt = data.groupby(['advert_id','adid'])['click'].agg({'advert_ad_cnt':'count'}).reset_index()
    advert_cnt = data.groupby('advert_id')['click'].agg({'advert_cnt':'count'}).reset_index()
    advert_ad_cnt = pd.merge(advert_ad_cnt,advert_cnt,how='left',on='advert_id')
    advert_ad_cnt['advert_ad_ratio'] = advert_ad_cnt['advert_ad_cnt'] / advert_ad_cnt['advert_cnt']
    del advert_ad_cnt['advert_ad_cnt']
    data = pd.merge(data, advert_ad_cnt, how='left', on=['advert_id','adid'])

    # add ctr feature
    data['period'] = data['day']
    data['period'][data['period'] < 27] = data['period'][data['period'] < 27] + 31
    for feat_1 in ['advert_id', 'advert_industry_inner', 'campaign_id', 'creative_height',
                   'creative_tp_dnf', 'creative_width', 'province', 'f_channel']:
        res = pd.DataFrame()
        temp = data[[feat_1, 'period', 'click']]
        for period in range(27, 35):
            if period == 27:
                count = temp.groupby([feat_1]).apply(
                    lambda x: x['click'][(x['period'] <= period).values].count()).reset_index(name=feat_1 + '_all')
                count1 = temp.groupby([feat_1]).apply(
                    lambda x: x['click'][(x['period'] <= period).values].sum()).reset_index(name=feat_1 + '_1')
            else:
                count = temp.groupby([feat_1]).apply(
                    lambda x: x['click'][(x['period'] < period).values].count()).reset_index(name=feat_1 + '_all')
                count1 = temp.groupby([feat_1]).apply(
                    lambda x: x['click'][(x['period'] < period).values].sum()).reset_index(name=feat_1 + '_1')
            count[feat_1 + '_1'] = count1[feat_1 + '_1']
            count.fillna(value=0, inplace=True)
            count[feat_1 + '_rate'] = round(count[feat_1 + '_1'] / count[feat_1 + '_all'], 5)
            count['period'] = period
            count.drop([feat_1 + '_all', feat_1 + '_1'], axis=1, inplace=True)
            count.fillna(value=0, inplace=True)
            res = res.append(count, ignore_index=True)
        print(feat_1, ' over')
        data = pd.merge(data, res, how='left', on=[feat_1, 'period'])

    del data['day']
    del data['period']

    # count ,cvr
    cdata = data.loc[data['click'].notnull(),:].copy()
    aid_count = cdata.groupby('advert_id')['make'].agg({'aid_count':'count'}).reset_index()
    make_count = cdata.groupby('make')['make'].agg({'make_count':'count'}).reset_index()
    advert_count = cdata.groupby('advert_industry_inner_1')['make'].agg({'advert_count':'count'}).reset_index()
    city_count = cdata.groupby('city')['make'].agg({'city_count':'count'}).reset_index()
    province_count = cdata.groupby('province')['make'].agg({'province_count':'count'}).reset_index()

    adid_city = cdata.groupby('adid_city')['click'].agg({'adid_city_count':'count'}).reset_index()
    adid_province = cdata.groupby('adid_province')['click'].agg({'adid_province_count':'count'}).reset_index()

    data = pd.merge(data, aid_count, how='left', on='advert_id')
    data = pd.merge(data, make_count, how='left', on='make')
    data = pd.merge(data, advert_count, how='left', on='advert_industry_inner_1')
    data = pd.merge(data, city_count, how='left', on='city')
    data = pd.merge(data, province_count, how='left', on='province')

    data = pd.merge(data, adid_city, how='left', on='adid_city')
    data = pd.merge(data, adid_province, how='left', on='adid_province')

    data.fillna(999,inplace=True)

    for col in data.columns:
        if data[col].dtype == 'float64':
            data[col] = data[col].astype(np.float32)
        elif data[col].dtype == 'int64':
            data[col] = data[col].astype(np.int32)

    full_user_tags = data['user_tags'].copy()
    train, test = data[:trn_num], data[-te_num:]
    print(train.info())

    test_id = test['instance_id'].values
    label = train['click'].values
    del data
    gc.collect()

    cv = CountVectorizer(min_df=5)
    cv.fit(full_user_tags)
    train_a = cv.transform(train['user_tags'])
    test_a = cv.transform(test['user_tags'])

    drop_col = ['instance_id','user_tags','click']
    train.drop(drop_col,axis=1,inplace=True)
    test.drop(drop_col,axis=1,inplace=True)

    train = sparse.hstack((train, train_a), 'csr')
    test = sparse.hstack((test, test_a), 'csr')

    try:
        feature_select = SelectPercentile(chi2, percentile=95)
        feature_select.fit(train, label)
        train = feature_select.transform(train)
        test = feature_select.transform(test)
        print("chi2 select finish")
    except:
        pass

    return train,test,label,test_id

def lgbcv_predict(training,label,predict):

    clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=48, reg_alpha=0.1, reg_lambda=3,
        max_depth=-1, n_estimators=3000, objective='binary',
        subsample=0.7, colsample_bytree=0.75, subsample_freq=1,
        min_child_weight=5,
        learning_rate=0.035, min_child_samples=15, random_state=2018, n_jobs=-1)

    kf = KFold(n_splits=10,random_state=2018,shuffle=True)
    best_logloss = []
    pred_list = []
    for i,(train_index, val_index) in enumerate(kf.split(training)):
        print(i)
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

    pred = np.mean(np.array(pred_list),axis=0)
    print(best_logloss,'\n',np.mean(best_logloss),np.std(best_logloss))

    return pred


if __name__ == '__main__':
    data_path = '../data/lgb1_label_testid.pkl'
    if os.path.exists(data_path):
        label,test_id = pickle.load(open(data_path,'rb'))
        train = sparse.load_npz('../data/lgb1_train_csr.npz').tocsr()
        test = sparse.load_npz('../data/lgb1_predict_csr.npz').tocsr()
    else:
        train,test,label,test_id = make_features()

        sparse.save_npz('../data/lgb1_train_csr.npz', train)
        sparse.save_npz('../data/lgb1_predict_csr.npz', test)
        pickle.dump((label,test_id),open(data_path,'wb'))

    print(train.shape)
    pred = lgbcv_predict(train,label,test)

    sub = pd.DataFrame()
    sub['instance_id'] = test_id
    sub['predicted_score'] = pred
    sub = sub.fillna(0)

    print(sub.shape)
    sub[['instance_id', 'predicted_score']].to_csv('../output/lgb1_res.csv',sep=",",
        index=False, float_format='%.8f')

    print(sub['predicted_score'].describe())
    print(sub['predicted_score'].mean())







