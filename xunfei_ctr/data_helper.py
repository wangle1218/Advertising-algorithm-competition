
import pandas as pd
import numpy as np
import datetime
import os
import re


train1_path = '../data/round1_iflyad_train.txt'
train2_path = '../data/round2_iflyad_train.txt'
test_path = '../data/round2_iflyad_test_feature.txt'

def remove_lowcase(se):
    count = dict(se.value_counts())
    se = se.map(lambda x : -1 if count[x]<5 else x)
    return se

def clearn_make(make):
    if 'oppo' in make:
        return 'oppo'
    elif 'vivo' in make:
        return 'vivo'
    elif 'huawei' in make or 'honor' in make:
        return 'huawei'
    elif 'redmi' in make:
        return 'xiaomi'

    strs = make.split()
    if len(strs) > 1:
        s = strs[0]
        if s == 'mi' or s == 'm1' or s == 'm2' or s == 'm3' or s == 'm6':
            s = 'xiaomi'
        return s
    return make

def clearn_model(model):
    if '%' in model:
        return model.split('%')[0]
    return model

def load_and_clearn_data():
    train = pd.concat([pd.read_table(train1_path),
                    pd.read_table(train2_path)],
                    axis=0, ignore_index=True)
    train.drop_duplicates(subset='instance_id',keep='first',inplace=True)
    test = pd.read_table(test_path)
    trn_num, te_num = len(train), len(test)
    data = pd.concat([train, test], axis=0, ignore_index=True)
    data['click'] = data['click'].fillna(-1)
    drop_col = ['os_name','advert_name','creative_is_js',
                'creative_is_voicead','app_paid']
    data.drop(drop_col,axis=1,inplace=True)

    bool_col = ['creative_is_jump', 'creative_is_download','creative_has_deeplink']
    data.loc[:,bool_col] = data.loc[:,bool_col].astype(np.int8)
    data['bool_sum'] = data[bool_col].sum(1)
    data['app_cate_id'].fillna(100,inplace=True)
    data['app_id'].fillna(-1,inplace=True)
    data['f_channel'].fillna('cha_nan',inplace=True)
    data['user_tags'].fillna('',inplace=True)
    data['make'].fillna('N',inplace=True)
    data['model'].fillna('N',inplace=True)

    data['osv'].fillna('-1.0',inplace=True)
    data['osv'] = data['osv'].map(lambda x : re.sub('[a-zA-Z]+','',x))
    data['osv'] = data['osv'].map(lambda x : re.sub('_','.',x))
    data['osv'] = data['osv'].map(lambda x : x.strip().split('.')[0])

    data['make'] = data['make'].map(lambda x : str(x).lower())
    pattern = re.compile(r'iphone.*')
    data['make'] = data['make'].map(lambda x : pattern.sub('iphone',x))
    data['make'] = data['make'].map(lambda x : re.sub(',',' ',x))
    data['make'] = data['make'].map(lambda x : clearn_make(x))

    data['model'] = data['model'].map(lambda x : str(x).lower())
    data['model'] = data['model'].map(lambda x : re.sub('[+-]',' ',x))
    data['model'] = data['model'].map(lambda x : clearn_model(x))

    data['time'] = data['time'].map(lambda x : datetime.datetime.fromtimestamp(x))
    data['day'] = data['time'].map(lambda x : int(x.day))
    data['hour'] = data['time'].map(lambda x : int(x.hour))
    del data['time']
    print(data.shape)

    return data,trn_num,te_num



