# -*- coding:utf-8 -*-


import pandas as pd 
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import StratifiedKFold,KFold
from scipy import sparse
import time,datetime
import pickle
import os,sys
import gc
from tqdm import tqdm
import math
import warnings
warnings.filterwarnings("ignore")

def apply_test_time(utc_time):
    if utc_time > 1556034659.0:
        return utc_time
    else:
        return 1556034659.0

def load_data(dtyp):
    test = pd.read_csv('../tmp_data/testB_data.csv')
    test_columns = test.columns.tolist()
    test['创建/修改时间'] = test['创建时间'].map(lambda x : apply_test_time(x))
    test['label'] = 0
    test_id = test[['样本id','广告id','出价']].copy()
    del test['样本id']

    test_columns.remove('样本id')
    feat_list = test_columns + ['创建/修改时间','label']
    train = pd.read_csv('../tmp_data/train_data.csv')
    train = train.drop_duplicates()
    train = train.sample(frac=1)
    train = train.loc[train['创建时间']!=0,:]
    if dtyp == 'partdata':
        train = train.loc[~((train['创建时间']<1546357859.0)&(train['label']==0)),:]
    # train = train.loc[train['label']>0,:]

    label = train['label'].values 
    print(train['label'].describe())
    train_num = len(label)

    train = train[feat_list]
    print(test.columns)
    print(train.columns)

    train['flag'] = 1
    test['flag'] = 2
    data = pd.concat([train, test],axis=0,ignore_index=True)
    print(data.info())

    return data,test_id,label,train_num


ad_logs = {20190410: pickle.load(open('../tmp_data/ad_log_410.pkl','rb')),
          20190411: pickle.load(open('../tmp_data/ad_log_411.pkl','rb')),
          20190412: pickle.load(open('../tmp_data/ad_log_412.pkl','rb')),
          20190413: pickle.load(open('../tmp_data/ad_log_413.pkl','rb')),
          20190414: pickle.load(open('../tmp_data/ad_log_414.pkl','rb')),
          20190415: pickle.load(open('../tmp_data/ad_log_415.pkl','rb')),
          20190416: pickle.load(open('../tmp_data/ad_log_416.pkl','rb')),
          20190417: pickle.load(open('../tmp_data/ad_log_417.pkl','rb')),
          20190418: pickle.load(open('../tmp_data/ad_log_418.pkl','rb')),
          20190419: pickle.load(open('../tmp_data/ad_log_419.pkl','rb')),
          20190420: pickle.load(open('../tmp_data/ad_log_420.pkl','rb')),
          20190421: pickle.load(open('../tmp_data/ad_log_421.pkl','rb')),
          20190422: pickle.load(open('../tmp_data/ad_log_422.pkl','rb')),
          20190423: pickle.load(open('../tmp_data/ad_log_423.pkl','rb')),
          20190424: pickle.load(open('../tmp_data/ad_log_testB.pkl','rb'))}


def pre_day_expo(utc_time,adid,days):
    times = time.localtime(int(utc_time - (days -1) * 86400))

    if times.tm_mday < 10:
        day = '0' + str(times.tm_mday)
    else:
        day = str(times.tm_mday)
    if times.tm_mon < 10:
        month = '0' + str(times.tm_mon)
    else:
        month = str(times.tm_mon)
    dates = int(str(times.tm_year) + month + day)

    if dates < 20190410 or dates > 20190424:
        return np.nan

    if adid in ad_logs[dates].keys():
        if 'expo_times' in ad_logs[dates][adid].keys():
            label = ad_logs[dates][adid]['expo_times']
        else:
            label = 0
    else:
        label = np.nan
    return label

def hist_mean_expo(utc_time,adid):
    times = time.localtime(int(utc_time))
    if times.tm_mday < 10:
        day = '0' + str(times.tm_mday)
    else:
        day = str(times.tm_mday)
    if times.tm_mon < 10:
        month = '0' + str(times.tm_mon)
    else:
        month = str(times.tm_mon)
    dates = int(str(times.tm_year) + month + day)

    mean_expo = 0
    k = 0
    for d in ad_logs.keys():
        if d<dates and adid in ad_logs[d].keys():
            if 'expo_times' in ad_logs[d][adid].keys():
                k += 1
                mean_expo += ad_logs[d][adid]['expo_times']
            else:
                mean_expo += 0

    if k == 0:
        mean_expo = 0
    else:
        mean_expo = mean_expo/k

    return mean_expo


def statis_feat(train, test, col):
    temp = train.groupby(col,as_index=False)['label'].agg({col+'_mean':'mean'})
    test = pd.merge(test, temp, on=col, how='left')
    # test[col+'_mean'].fillna(test[col+'_mean'].mean(), inplace=True)
    return test

def remove_lowcase(se):
    count = dict(se.value_counts())
    se = se.map(lambda x : -1 if count[x]<10 else x)
    return se


def cal_range_num(utc_time,adid,p=False):
    if p:
        times = time.localtime(int(utc_time+86400))
    else:
        times = time.localtime(int(utc_time))

    if times.tm_mday < 10:
        day = '0' + str(times.tm_mday)
    else:
        day = str(times.tm_mday)
    if times.tm_mon < 10:
        month = '0' + str(times.tm_mon)
    else:
        month = str(times.tm_mon)
    dates = int(str(times.tm_year) + month + day)
    if dates > 20190424:
        dates = 20190424
    try:
        return ad_logs[dates][adid]['cover_range']
    except:
        return 0

def cal_hist_win_ration(utc_time,adid,types='mean'):
    times = time.localtime(int(utc_time))
    if times.tm_mday < 10:
        day = '0' + str(times.tm_mday)
    else:
        day = str(times.tm_mday)
    if times.tm_mon < 10:
        month = '0' + str(times.tm_mon)
    else:
        month = str(times.tm_mon)
    dates = int(str(times.tm_year) + month + day)

    win_ration_list = []
    for d in ad_logs.keys():
        k = 0
        expo = 0
        if dates > d and adid in ad_logs[d].keys():
            k += 1
            if 'expo_times' in ad_logs[d][adid].keys():
                expo += ad_logs[d][adid]['expo_times']

            cover = ad_logs[d][adid]['cover_range']
            win_ration_list.append(expo/cover)

    if len(win_ration_list)>0:
        if types =='mean':
            return np.mean(win_ration_list)
        elif types =='sum':
            return np.sum(win_ration_list)
        elif types =='std':
            return np.std(win_ration_list)
        elif types =='max':
            return np.max(win_ration_list)
        else:
            return np.min(win_ration_list)
    else:
        return np.nan


def gen_features(data,train_num):
    # data['pre_day1_expo'] = data.apply(lambda x : pre_day_expo(x['创建/修改时间'], x['广告id'],1), axis=1)
    data['pre_day2_expo'] = data.apply(lambda x : pre_day_expo(x['创建/修改时间'], x['广告id'],2), axis=1)
    data['pre_day3_expo'] = data.apply(lambda x : pre_day_expo(x['创建/修改时间'], x['广告id'],3), axis=1)
    data['pre_day4_expo'] = data.apply(lambda x : pre_day_expo(x['创建/修改时间'], x['广告id'],4), axis=1)
    data['pre_day5_expo'] = data.apply(lambda x : pre_day_expo(x['创建/修改时间'], x['广告id'],5), axis=1)
    data['pre_day6_expo'] = data.apply(lambda x : pre_day_expo(x['创建/修改时间'], x['广告id'],6), axis=1)
    data['pre_day7_expo'] = data.apply(lambda x : pre_day_expo(x['创建/修改时间'], x['广告id'],7), axis=1)
    data['pre_day8_expo'] = data.apply(lambda x : pre_day_expo(x['创建/修改时间'], x['广告id'],8), axis=1)

    col_list = ['pre_day%d_expo'%i for i in range(2,9)]
    data['pre_expo_mean'] = data[col_list].mean(axis=1)
    data['pre_expo_sum'] = data[col_list].sum(axis=1)
    data['pre_expo_std'] = data[col_list].std(axis=1)
    data['pre_expo_count'] = data[col_list].count(axis=1)
    data['pre_expo_min'] = data[col_list].min(axis=1)
    data['pre_expo_max'] = data[col_list].max(axis=1)
    # data['pre_expo_mode'] = data[col_list].mode(axis=1)

    data_hist_diff = data[col_list].copy()
    data_hist_diff.fillna(0,inplace=True)
    data_hist_diff = data_hist_diff.diff(axis=1)
    del data_hist_diff['pre_day2_expo']
    data_hist_diff.columns = [col+'_diff' for col in data_hist_diff.columns]
    data_hist_diff['diff_mean'] = data_hist_diff.mean(axis=1)
    data_hist_diff['diff_sum'] = data_hist_diff.sum(axis=1)
    data_hist_diff['diff_std'] = data_hist_diff.std(axis=1)
    data = pd.concat([data,data_hist_diff],axis=1)

    data['hist_mean_expo'] = data.apply(lambda x : hist_mean_expo(x['创建/修改时间'], x['广告id']), axis=1)
    data['iter_day'] = data.apply(lambda x : (x['创建/修改时间']-x['创建时间'])//(86400*7), axis=1)
    data['iter_day2'] = data.apply(lambda x : (x['创建/修改时间']-x['创建时间'])//86400, axis=1)
    print("时序特征完成")

    # data['modifly_times'] = 
    # data.loc[data['出价']>2000,'出价'] = data.loc[data['出价']>2000,'出价'].values ** 0.5 + 2000

    # 预测当天的覆盖范围
    data['广告id'] = data['广告id'].astype(int)
    data['predict_day_range_num'] = data.apply(lambda x : cal_range_num(x['创建/修改时间'], x['广告id'],p=True), axis=1)
    data['today_range_num'] = data.apply(lambda x : cal_range_num(x['创建/修改时间'], x['广告id']), axis=1)
    data['diff_range_num'] = data['predict_day_range_num'] - data['today_range_num']
    data['hist_mean_win_ration'] = data.apply(lambda x : cal_hist_win_ration(x['创建/修改时间'], x['广告id'],types='mean'), axis=1)
    data['hist_mean_win_ration'].fillna(data['hist_mean_win_ration'].mean(),inplace=True)
    data['cal_predict_day_expo'] =  data['predict_day_range_num'].values * data['hist_mean_win_ration'].values
    data['cal_today_expo'] = data['today_range_num'].values * data['hist_mean_win_ration'].values
    data['today_diff_expo'] = data['cal_today_expo'].values - data['pre_day2_expo'].values
    data['cal_predict_day_expo2'] = data['today_diff_expo'].values + data['cal_predict_day_expo'].values

    data['hist_sum_win_ration'] = data.apply(lambda x : cal_hist_win_ration(x['创建/修改时间'], x['广告id'],types='sum'), axis=1)
    data['hist_std_win_ration'] = data.apply(lambda x : cal_hist_win_ration(x['创建/修改时间'], x['广告id'],types='std'), axis=1)
    data['hist_max_win_ration'] = data.apply(lambda x : cal_hist_win_ration(x['创建/修改时间'], x['广告id'],types='max'), axis=1)
    data['hist_min_win_ration'] = data.apply(lambda x : cal_hist_win_ration(x['创建/修改时间'], x['广告id'],types='min'), axis=1)

    # rank, count 特征
    adid_data = data[['广告账户id','广告id','创建时间']].sort_values(by=['广告账户id','创建时间']).drop_duplicates(subset=['广告账户id','广告id'],keep='first').reset_index(drop=True)
    adid_count = adid_data.groupby(by=['广告账户id'])['广告id'].size()
    data['ad_acount_have_ad_num'] = data['广告账户id'].map(lambda x : adid_count[x])

    adrank = adid_data.groupby(by=['广告账户id'])['创建时间'].rank()
    adid_data['ad_rank'] = adrank.values

    data = pd.merge(data, adid_data[['广告账户id','广告id','ad_rank']], on=['广告账户id','广告id'],how='left')
    data['ad_rank'].fillna(1, inplace=True)

    #---
    adid_data = data[['广告账户id','商品id','创建时间']].sort_values(by=['广告账户id','创建时间']).drop_duplicates(subset=['广告账户id','商品id'],keep='first').reset_index(drop=True)
    adid_count = adid_data.groupby(by=['广告账户id'])['商品id'].size()
    data['ad_acount_have_commodity_num'] = data['广告账户id'].map(lambda x : adid_count[x])

    adrank = adid_data.groupby(by=['广告账户id'])['创建时间'].rank()
    adid_data['commodity_id_rank'] = adrank.values

    data = pd.merge(data, adid_data[['广告账户id','商品id','commodity_id_rank']], on=['广告账户id','商品id'],how='left')
    data['commodity_id_rank'].fillna(1, inplace=True)

    #---
    # adid_data = data[['广告账户id','商品类型','创建时间']].sort_values(by=['广告账户id','创建时间']).drop_duplicates(subset=['广告账户id','商品类型'],keep='first').reset_index(drop=True)
    # adid_count = adid_data.groupby(by=['广告账户id'])['商品类型'].size()
    # data['ad_acount_have_adtype_num'] = data['广告账户id'].map(lambda x : adid_count[x])

    # adrank = adid_data.groupby(by=['广告账户id'])['创建时间'].rank()
    # adid_data['commodity_type_rank'] = adrank.values

    # data = pd.merge(data, adid_data[['广告账户id','商品类型','commodity_type_rank']], on=['广告账户id','商品类型'],how='left')
    # data['commodity_type_rank'].fillna(1, inplace=True)
    print("排序特征完成")

    data['广告行业id'] = data['广告行业id'].astype(str)
    data['广告行业id'] = data['广告行业id'].map(lambda x : x.split(',')[0])
    data['广告行业id'] = data['广告行业id'].astype(int)


    # 统计特征
    items = ['广告账户id','商品id', '商品类型','广告行业id','素材尺寸','目标转化类型','计费类型']

    commbian_feat = []
    lbl = LabelEncoder()
    for i in range(len(items)-1):
        for j in range(i+1,len(items)):
            new_col = items[i]+'-'+items[j]
            commbian_feat.append(new_col)
            data[new_col] = data[items[i]].astype('str') + '_' +data[items[j]].astype('str')
            data[new_col] = lbl.fit_transform(data[new_col])
            data[new_col] = remove_lowcase(data[new_col])

    data['index'] = list(range(data.shape[0]))
    for col in commbian_feat + items[:5]:
        data[col] = data[col].astype(int)
        df_cv = data[['index',col,'label','flag']].copy()

        train = df_cv.loc[df_cv['flag'] == 1,:].reset_index(drop=True)
        test = df_cv.loc[df_cv['flag'] != 1,:]

        temp = train.groupby(col,as_index=False)['label'].agg({col+'_mean':'mean'})
        test = pd.merge(test, temp, on=col, how='left')
        # test[col+'_mean'].fillna(test[col+'_mean'].mean(), inplace=True)

        df_stas_feat = None
        kf = KFold(n_splits=5,random_state=2018,shuffle=True)
        for train_index, val_index in kf.split(train):
            X_train = train.loc[train_index,:]
            X_val = train.loc[val_index,:]

            X_val = statis_feat(X_train,X_val, col)
            df_stas_feat = pd.concat([df_stas_feat,X_val],axis=0)

        df_stas_feat = pd.concat([df_stas_feat,test],axis=0)
        df_stas_feat.drop([col,'label','flag'], axis=1, inplace=True)
        data = pd.merge(data, df_stas_feat,how='left',on='index')

    items_mean = [it+'_mean' for it in items[:5]]
    data['items_mean_mean'] = data[items_mean].mean(axis=1)
    data['items_mean_sum'] = data[items_mean].sum(axis=1)
    data['items_mean_std'] = data[items_mean].std(axis=1)

    items_mean = [it+'_mean' for it in commbian_feat]
    data['commbian_feat_mean_mean'] = data[items_mean].mean(axis=1)
    data['commbian_feat_mean_sum'] = data[items_mean].sum(axis=1)
    data['commbian_feat_mean_std'] = data[items_mean].std(axis=1)

    # data['items_mean_count'] = data[items_mean].count(axis=1)
    print("交叉统计，组合特征完成")

    for col in items + commbian_feat:
        data[col] = data[col].astype('category')

    del data['index']
    del data['创建/修改时间']
    del data['计费类型']
    # data.to_csv('../tmp_data/training_data.csv',index=False)

    train = data.loc[data['flag'] == 1,:].reset_index(drop=True)
    test = data.loc[data['flag'] != 1,:]
    del train['label']
    del test['label']
    del train['flag']
    del test['flag']
    del train['创建时间']
    del test['创建时间']
    del data
    gc.collect()

    return train, test

if __name__ == '__main__':
    if sys.argv[1] == 'partdata':
        data,test_id,label,train_num = load_data('partdata')
    else:
        data,test_id,label,train_num = load_data('fulldata')
    train,test = gen_features(data,train_num)

    pickle.dump(train,open('../tmp_data/lgb_train.pkl','wb'))
    pickle.dump(test,open('../tmp_data/lgb_test.pkl','wb'))
    pickle.dump((test_id,label,train_num),open('../tmp_data/label.pkl','wb'))









