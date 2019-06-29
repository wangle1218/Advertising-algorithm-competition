# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import time,datetime
import pickle

input_data_dir = '../data/total_data'


def ivt2date(x):
    times = time.localtime(int(x))
    year = str(times.tm_year)
    month = str(times.tm_mon)
    day = str(times.tm_mday)
    if times.tm_mday < 10:
        day = '0'+str(times.tm_mday)
    if times.tm_mon < 10:
        month = '0'+str(times.tm_mon)
    return int(year+month+day)


"""
ad_log: adid # dict
            :expo_times #当日曝光次数 int
            :cover_range #当日覆盖次数 int
            :ad_local_id #广告位id统计 dict
                        :local_id_num #在某个广告位id出的出现次数 int
"""

for expo_dates in range(410, 423):
    ad_log = {}
    i = 0
    flog = open(input_data_dir + '/track_log/track_log_20190%d.out' % expo_dates,'r')
    
    while 1:
        line = flog.readline()
        if not line:
            break
        line = line.split('\t')
        ad_local_id = int(line[3])
        competitive_queue = line[-1].split(';')
        for cq in competitive_queue:
            cq = cq.split(',')
            if cq[5] == '1':
                continue
            adid = int(cq[0])
            ad_log.setdefault(adid,{})
            ad_log[adid].setdefault('cover_range',1)
            ad_log[adid]['cover_range'] += 1
            
            ad_log[adid].setdefault('ad_local_id',{})
            if ad_local_id not in ad_log[adid]['ad_local_id'].keys():
                ad_log[adid]['ad_local_id'].setdefault(ad_local_id,1)
            else:
                ad_log[adid]['ad_local_id'][ad_local_id] += 1
            
            if cq[6] == '1':
                ad_log[adid].setdefault('expo_times',1)
                ad_log[adid]['expo_times'] += 1
                
        i += 1
            
    flog.close()
    pickle.dump(ad_log, open('../tmp_data/ad_log_%d.pkl' % expo_dates, 'wb'))
    print(expo_dates,i)


ad_log_423 = {}
i = 0
flog = open(input_data_dir + '/test_tracklog_20190423.last.out','r')

while 1:
    line = flog.readline()
    if not line:
        break
    line = line.split('\t')
    ad_local_id = int(line[3])
    competitive_queue = line[-1].split(';')
    for cq in competitive_queue:
        cq = cq.split(',')
        if cq[5] == '1':
            continue
        adid = int(cq[0])
        ad_log_423.setdefault(adid,{})
        ad_log_423[adid].setdefault('cover_range',1)
        ad_log_423[adid]['cover_range'] += 1
        
        ad_log_423[adid].setdefault('ad_local_id',{})
        if ad_local_id not in ad_log_423[adid]['ad_local_id'].keys():
            ad_log_423[adid]['ad_local_id'].setdefault(ad_local_id,1)
        else:
            ad_log_423[adid]['ad_local_id'][ad_local_id] += 1
            
    i += 1
        
flog.close()
# pickle.dump(ad_log_423, open('./data/total_data/ad_log_423.pkl', 'wb'))
print(expo_dates,i)


ad_log_testA = {}
i = 0
flog = open(input_data_dir + '/final_select_test_request.out','r')
while 1:
    line = flog.readline()
    if not line:
        break
    line = line.split('\t')
    adid = int(line[0])
    request_queue = line[-1].split('|')
    for rq in request_queue:
        rq = rq.split(',')
        ad_local_id = int(rq[1])
        ad_log_testA.setdefault(adid,{})
        ad_log_testA[adid].setdefault('cover_range',1)
        ad_log_testA[adid]['cover_range'] += 1

        ad_log_testA[adid].setdefault('ad_local_id',{})
        if ad_local_id not in ad_log_testA[adid]['ad_local_id'].keys():
            ad_log_testA[adid]['ad_local_id'].setdefault(ad_local_id,1)
        else:
            ad_log_testA[adid]['ad_local_id'][ad_local_id] += 1

    i += 1

flog.close()
# pickle.dump(ad_log_testA, open('./data/total_data/ad_log_testA.pkl', 'wb'))
print(expo_dates,i)

ad_log_423.update(ad_log_testA)
pickle.dump(ad_log_423, open('../tmp_data/ad_log_423.pkl', 'wb'))


ad_log_testB = {}
i = 0
flog = open(input_data_dir + '/BTest/Btest_select_request_20190424.out','r')
while 1:
    line = flog.readline()
    if not line:
        break
    line = line.split('\t')
    adid = int(line[0])
    request_queue = line[-1].split('|')
    for rq in request_queue:
        rq = rq.split(',')
        ad_local_id = int(rq[1])
        ad_log_testB.setdefault(adid,{})
        ad_log_testB[adid].setdefault('cover_range',1)
        ad_log_testB[adid]['cover_range'] += 1

        ad_log_testB[adid].setdefault('ad_local_id',{})
        if ad_local_id not in ad_log_testB[adid]['ad_local_id'].keys():
            ad_log_testB[adid]['ad_local_id'].setdefault(ad_local_id,1)
        else:
            ad_log_testB[adid]['ad_local_id'][ad_local_id] += 1

    i += 1

flog.close()
pickle.dump(ad_log_testB, open('../tmp_data/ad_log_testB.pkl', 'wb'))
print(expo_dates,i)


ad_static = pd.read_table(input_data_dir + '/map_ad_static.out',sep='\t',header=None,low_memory=False)
ad_static.columns = ['广告id','创建时间','广告账户id','商品id','商品类型','广告行业id','素材尺寸']
ad_static.fillna(-1,inplace=True)
# ad_static['创建时间'] = ad_static['创建时间'].map(lambda x: ivt2date(x))

ad_operation = pd.read_table(input_data_dir + '/final_map_bid_opt.out',sep='\t',header=None,low_memory=False)
ad_operation.columns = ['广告id','创建/修改时间','操作类型','目标转化类型','计费类型','出价']


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
          20190423: pickle.load(open('../tmp_data/ad_log_423.pkl','rb'))}


adid_list, adid_date = [], []
        
for date in [d for d in range(20190411,20190423)]:
    for ids in ad_operation['广告id'].unique():
        if ids in ad_logs[date].keys():
            # if 'expo_times' in ad_logs[date][ids].keys():
            adid_list.append(ids)
            adid_date.append(date-1)
    print(date)


ad_expo_df = pd.DataFrame()
ad_expo_df['广告id'] = adid_list
ad_expo_df['创建/修改时间'] = adid_date
print(ad_expo_df.shape)

ad_operation['创建/修改时间'] = ad_operation['创建/修改时间'] // 1000000
ad_operation = pd.merge(ad_operation,ad_expo_df,on=['广告id','创建/修改时间'],how='outer')
ad_operation = ad_operation.sort_values(by=['广告id','创建/修改时间'])

ad_operation_copy = []

for adid in ad_operation['广告id'].unique():
    copy_op = ad_operation[ad_operation['广告id']==adid].copy()
    copy_op.fillna(method='ffill',inplace=True)
    ad_operation_copy.extend(copy_op.values)

ad_operation_copy = pd.DataFrame(ad_operation_copy)
ad_operation_copy.columns = ad_operation.columns
ad_operation_copy.drop_duplicates(inplace=True)
ad_operation_copy = ad_operation_copy.loc[ad_operation_copy['出价'].notnull()]

train = pd.merge(ad_operation_copy, ad_static,how='inner', on=['广告id'])
# train = train.loc[train['创建/修改时间'].notnull()]
train['创建/修改时间'] = train['创建/修改时间'].astype(int)
print(train.shape)
train = train.loc[train['创建/修改时间']!=20190422,:]
print(train.shape)

def add_label(date,adid):
    dates = date+1
    if dates > 20190422 or dates < 20190410:
        return 0
    if adid in ad_logs[dates].keys():
        try:
            return ad_logs[dates][adid]['expo_times']
        except:
            return 0
    else:
        return 0

train['label'] = train.apply(lambda x: add_label(x['创建/修改时间'], x['广告id']),axis=1)

def date2uct(dataint):
    dataint = str(dataint)
    year = int(dataint[:4])
    month = int(dataint[4:6])
    day = int(dataint[6:8])
    if day == 0:
        day = 1
    if month == 2 and day > 28:
        day = 28
    dateC1=datetime.datetime(year,month,day,23,59,59)
    timestamp2=time.mktime(dateC1.timetuple())
    return timestamp2

train['创建/修改时间'] = train['创建/修改时间'].map(lambda x :date2uct(x))
# train['创建时间'] = train['创建时间'].map(lambda x :date2uct(x))

del train['操作类型']
train.to_csv('../tmp_data/train_data.csv',index=False)
print(train.loc[train['label']>-1,'label'].describe())
print(train.head())

test_sample = pd.read_table(input_data_dir + '/BTest/Btest_sample_bid.out',sep='\t',header=None,low_memory=False)
test_sample.columns = ['样本id','广告id','目标转化类型','计费类型','出价']
test_sample = pd.merge(test_sample,ad_static, how='left', on=['广告id'])
# test_sample['创建时间'] = test_sample['创建时间'].map(lambda x :date2uct(x))

test_sample.to_csv('../tmp_data/testB_data.csv',index=False)









