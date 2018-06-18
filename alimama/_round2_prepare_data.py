# -*- encoding:utf-8 -*-

import pandas as pd
import numpy as np
from datetime import datetime
import gc

test = pd.read_csv("./data/round2_ijcai_18_test_a_20180425.txt", sep="\s+")
# B 榜测试集，如果是 A 榜，则下面一段代码可以注释掉
testb = pd.read_csv("./data/round2_ijcai_18_test_b_20180510.txt", sep="\s+")
print(testb.shape, test.shape)
test = pd.concat([test,testb],axis=0,ignore_index=True)
del testb
gc.collect()

# --------

test['context_day'] = test['context_timestamp'].map(lambda x : datetime.fromtimestamp(x).day)


# ------------------------提取商品表-----------------------------

item_feat = ['item_id','item_category_list','item_property_list','item_brand_id','context_timestamp',\
            'item_city_id','item_price_level','item_sales_level','item_collected_level','item_pv_level']
item_feat2 = ['item_id','context_day',\
            'item_price_level','item_sales_level','item_collected_level','item_pv_level']
reader = pd.read_csv("./data/round2_train.txt", sep="\s+",iterator=True)
chunks = []
loop = True
while loop:
    try:
        chunk = reader.get_chunk(500000)[item_feat]
        chunks.append(chunk)
    except StopIteration:
        loop = False
        print("Iteration is stopped")

df_item = pd.concat(chunks,axis=0, ignore_index=True)
chunks = []
df_item = pd.concat([df_item, test[item_feat]],axis=0)
df_item['context_day'] = df_item['context_timestamp'].map(lambda x : datetime.fromtimestamp(x).day)
del df_item['context_timestamp']
df_item = df_item[df_item['item_sales_level']!=-1]

df_item['context_day'] = df_item['context_day'].replace(31,0)
df_item = df_item.sort_values(by=['item_id','context_day'])

item = df_item.drop_duplicates(subset='item_id',keep='last')
del item['context_day']
item.to_csv('./data/item_file.csv',index=False)
print('item_file',item.shape)
del item
gc.collect()

df_item = df_item[item_feat2]
df_item = df_item.drop_duplicates(subset=['item_id','item_price_level','item_sales_level','item_collected_level',\
                                'item_pv_level'],keep='last')
df_item.to_csv('./data/item_day.csv',index=False)
print('item_day', df_item.shape)

del df_item
gc.collect()

# -------------------------用户表----------------------------

user_feat = ['user_id','user_gender_id','user_age_level','user_occupation_id','user_star_level']
reader = pd.read_csv("./data/round2_train.txt", sep="\s+",iterator=True)
chunks = []
loop = True
while loop:
    try:
        chunk = reader.get_chunk(500000)[user_feat]
        chunks.append(chunk)
    except StopIteration:
        loop = False
        print("Iteration is stopped")
df_user = pd.concat(chunks,axis=0, ignore_index=True)

df_user = pd.concat(chunks,axis=0, ignore_index=True)
df_user = pd.concat([df_user, test[user_feat]],axis=0)
df_user.drop_duplicates(subset='user_id',keep='first',inplace=True)
df_user.to_csv('./data/user_file.csv',index=False)
print('user_file', df_user.shape)

del df_user
gc.collect()

# -------------------------店铺----------------------------

shop_feat = ['shop_id','shop_review_num_level','shop_review_positive_rate',\
            'shop_star_level','shop_score_service','shop_score_delivery',\
            'shop_score_description','context_timestamp']
reader = pd.read_csv("./data/round2_train.txt", sep="\s+",iterator=True)
chunks = []
loop = True
while loop:
    try:
        chunk = reader.get_chunk(500000)[shop_feat]
        chunks.append(chunk)
    except StopIteration:
        loop = False
        print("Iteration is stopped")
df_shop = pd.concat(chunks,axis=0, ignore_index=True)

df_shop['context_day'] = df_shop['context_timestamp'].map(lambda x : datetime.fromtimestamp(x).day)
shop_feat.remove('context_timestamp')
shop_feat.append('context_day')
del df_shop['context_timestamp']
df_shop = pd.concat([df_shop, test[shop_feat]],axis=0)
df_shop.drop_duplicates(inplace=True)
df_shop.to_csv('./data/shop_file.csv',index=False)
print('shop_file', df_shop.shape)

del df_shop
gc.collect()

# -------------------------上下文----------------------------

context_feat = ['instance_id','shop_id','item_id','context_id','user_id','context_timestamp',\
                'context_page_id','predict_category_property','is_trade']
reader = pd.read_csv("./data/round2_train.txt", sep="\s+",iterator=True)
chunks = []
loop = True
while loop:
    try:
        chunk = reader.get_chunk(500000)[context_feat]
        chunks.append(chunk)
    except StopIteration:
        loop = False
        print("Iteration is stopped")
df_context = pd.concat(chunks,axis=0, ignore_index=True)
chunks = []

context_feat.append('context_day')
context_feat.remove('is_trade')
df_context['context_day'] = df_context['context_timestamp'].map(lambda x : datetime.fromtimestamp(x).day)
df_context = pd.concat([df_context,test[context_feat]],axis=0)
data = df_context[df_context['context_day']==7]
data.to_csv('./data/train_test_context_file.csv',index=False)
print('train_test_context_file', data.shape)

data = df_context[df_context['context_day']!=7]
data.to_csv('./data/history_file.csv',index=False)
print('history_file', data.shape)

del data
del df_context

gc.collect()







