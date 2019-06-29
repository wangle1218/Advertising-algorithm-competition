# -*- coding:utf-8 -*-

import pandas as pd 
import numpy as np
import pickle
import time,datetime
import math

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
          20190424: pickle.load(open('../tmp_data/ad_log_testB.pkl','rb'))}


def apply_test_time(utc_time):
    if utc_time > 1556034659.0:
        return utc_time
    else:
        return 1556034659.0

def softmax(n):
    range_n = [i**1.1 for i in range(1,n+1)]
    return np.exp(range_n) / np.sum(np.exp(range_n))

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
            k += 1
            if 'expo_times' in ad_logs[d][adid].keys():
                mean_expo += ad_logs[d][adid]['expo_times']
            else:
                mean_expo += 0

    if k == 0:
        mean_expo = 0
    else:
        mean_expo = mean_expo/k

    return mean_expo

def apply_hist_expo(adid,date):
    if adid in ad_logs[date].keys():
        try:
            return ad_logs[date][adid]['expo_times']
        except:
            return 0
    else:
        return np.nan
    
def apply_hist_range(adid,date):
    if adid in ad_logs[date].keys():
        return ad_logs[date][adid]['cover_range']
    else:
        return np.nan

def get_hist_expo():
    test = pd.read_csv('../tmp_data/testB_data.csv')
    test_columns = test.columns.tolist()
    test['创建/修改时间'] = test['创建时间'].map(lambda x : apply_test_time(x))
    test_id = test[['样本id','广告id','出价']].copy()

    test['expo410'] = test.apply(lambda x: apply_hist_expo(x['广告id'], 20190410), axis=1)
    test['range410'] = test.apply(lambda x: apply_hist_range(x['广告id'], 20190410), axis=1)

    test['expo411'] = test.apply(lambda x: apply_hist_expo(x['广告id'], 20190411), axis=1)
    test['range411'] = test.apply(lambda x: apply_hist_range(x['广告id'], 20190411), axis=1)

    test['expo412'] = test.apply(lambda x: apply_hist_expo(x['广告id'], 20190412), axis=1)
    test['range412'] = test.apply(lambda x: apply_hist_range(x['广告id'], 20190412), axis=1)

    test['expo413'] = test.apply(lambda x: apply_hist_expo(x['广告id'], 20190413), axis=1)
    test['range413'] = test.apply(lambda x: apply_hist_range(x['广告id'], 20190413), axis=1)

    test['expo414'] = test.apply(lambda x: apply_hist_expo(x['广告id'], 20190414), axis=1)
    test['range414'] = test.apply(lambda x: apply_hist_range(x['广告id'], 20190414), axis=1)

    test['expo415'] = test.apply(lambda x: apply_hist_expo(x['广告id'], 20190415), axis=1)
    test['range415'] = test.apply(lambda x: apply_hist_range(x['广告id'], 20190415), axis=1)

    test['expo416'] = test.apply(lambda x: apply_hist_expo(x['广告id'], 20190416), axis=1)
    test['range416'] = test.apply(lambda x: apply_hist_range(x['广告id'], 20190416), axis=1)

    test['expo417'] = test.apply(lambda x: apply_hist_expo(x['广告id'], 20190417), axis=1)
    test['range417'] = test.apply(lambda x: apply_hist_range(x['广告id'], 20190417), axis=1)

    test['expo418'] = test.apply(lambda x: apply_hist_expo(x['广告id'], 20190418), axis=1)
    test['range418'] = test.apply(lambda x: apply_hist_range(x['广告id'], 20190418), axis=1)

    test['expo419'] = test.apply(lambda x: apply_hist_expo(x['广告id'], 20190419), axis=1)
    test['range419'] = test.apply(lambda x: apply_hist_range(x['广告id'], 20190419), axis=1)

    test['expo420'] = test.apply(lambda x: apply_hist_expo(x['广告id'], 20190420), axis=1)
    test['range420'] = test.apply(lambda x: apply_hist_range(x['广告id'], 20190420), axis=1)

    test['expo421'] = test.apply(lambda x: apply_hist_expo(x['广告id'], 20190421), axis=1)
    test['range421'] = test.apply(lambda x: apply_hist_range(x['广告id'], 20190421), axis=1)

    test['expo422'] = test.apply(lambda x: apply_hist_expo(x['广告id'], 20190422), axis=1)
    test['range422'] = test.apply(lambda x: apply_hist_range(x['广告id'], 20190422), axis=1)

    test['range424'] = test.apply(lambda x: apply_hist_range(x['广告id'], 20190424), axis=1)


    hist_win_ration = ['win_ration_%d' % i for i in range(410,423)]
    for d in range(410,423):
        test['win_ration_%d' % d] = test['expo%d'%d] / test['range%d'%d]

    test['hist_mean_win_ration'] = test[hist_win_ration].mean(1)
    # test['hist_mean_win_ration'].fillna(0.027335, inplace=True)
    test['hist_mean_expo'] = test['hist_mean_win_ration'] * test['range424']
    print(test['hist_mean_expo'].describe())
    null_idx = test.loc[test['hist_mean_expo'].isnull(),:].index

    return test, test_id, null_idx


def get_ceil(x):
    if (x > 4) and (np.ceil(x) - x <= 0.2):
        return np.ceil(x)
    else:
        return x


def get_merge_result(test,test_id,null_idx,sub_path):
    dfmodel1 = pd.read_csv(sub_path,header=None)
    model_pred1 = dfmodel1[1].values

    print('模型预测新广告曝光\n',dfmodel1.loc[null_idx,1].describe())

    test.loc[test['hist_mean_expo'].isnull(),'hist_mean_expo'] = dfmodel1.loc[null_idx,1].values 
    pred = test['hist_mean_expo'].values

    test_id['预估日曝光'] = pred
    sub = test_id.sort_values(by=['广告id','出价'])
    test_id = test_id.sort_values(by=['广告id','预估日曝光'])

    sub['预估日曝光'] = test_id['预估日曝光'].values
    sub.sort_values(by=['样本id'],inplace=True)

    rule_pred = sub['预估日曝光'].values

    pred = np.where(rule_pred<model_pred1,rule_pred,model_pred1)
    sub['预估日曝光'] = pred

    print(sub_path.split('-')[-1])
    if sub_path.split('-')[-1] == 'partdata.csv':
        threshold = 2.7
    else:
        threshold = 2.7

    sub.loc[(sub['预估日曝光']<threshold)&(sub['预估日曝光']>1.0),'预估日曝光'] = 0.95
    sub['预估日曝光'] = sub['预估日曝光'].map(lambda x : get_ceil(x))
    sub['预估日曝光'] = sub['预估日曝光'] + sub['出价']**0.5 / 800
    sub['预估日曝光'] = sub['预估日曝光'].map(lambda x : np.around(x,4))

    sub2 = sub.sort_values(by=['广告id','出价'])
    sub = sub.sort_values(by=['广告id','预估日曝光'])
    sub2['预估日曝光'] = sub['预估日曝光'].values
    sub2.sort_values(by=['样本id'],inplace=True)

    print('提交结果\n',sub['预估日曝光'].describe())
    print('提交结果小于1\n',sub.loc[sub['预估日曝光']<=1,'预估日曝光'].describe(),'\n----------------')

    return sub2[['样本id','预估日曝光']]


test, test_id, null_idx = get_hist_expo()
sub_df1 = get_merge_result(test.copy(),test_id.copy(),null_idx,'../tmp_data/submission-partdata.csv')
sub_df2 = get_merge_result(test.copy(),test_id.copy(),null_idx,'../tmp_data/submission-fulldata.csv')

pred = sub_df1['预估日曝光'].values *0.4 + sub_df2['预估日曝光'].values *0.6
sub_df1['预估日曝光'] = pred
sub_df1['预估日曝光'] =sub_df1['预估日曝光'].map(lambda x: np.around(x,4))
sub_df1.to_csv('../submission.csv',index=False,header=None)

print('提交结果\n',sub_df1['预估日曝光'].describe())
print('提交结果小于1\n',sub_df1.loc[sub_df1['预估日曝光']<=1,'预估日曝光'].describe())



