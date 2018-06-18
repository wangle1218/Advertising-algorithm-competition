# -*- encoding:utf-8 -*-

import pandas as pd
import numpy as np 
from datetime import datetime
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.model_selection import KFold
import time
import os
import pickle
import gc
import warnings
warnings.filterwarnings("ignore")

lbl = LabelEncoder()

def remove_lowcase(se):
    count = dict(se.value_counts())
    se = se.map(lambda x : -1 if count[x]<5 else x)
    return se

def gen_user_feat():
    """
    组合用户的性别、年龄、职业、星级属性
    """
    user = pd.read_csv("./data/user_file.csv",header=0)
    user['gender_age'] = user['user_gender_id'].astype('str') + '_' + user['user_age_level'].astype('str')
    user['gender_occp'] = user['user_gender_id'].astype('str') + '_' + user['user_occupation_id'].astype('str')
    user['gender_star'] = user['user_gender_id'].astype('str') + '_' + user['user_star_level'].astype('str')
    user['age_occp'] = user['user_age_level'].astype('str') + '_' + user['user_occupation_id'].astype('str')
    user['age_star'] = user['user_age_level'].astype('str') + '_' + user['user_star_level'].astype('str')
    user['occp_star'] = user['user_occupation_id'].astype('str') + '_' + user['user_star_level'].astype('str')

    for col in ['gender_age','gender_occp','gender_star','age_occp','age_star','occp_star','user_age_level',\
                'user_occupation_id','user_star_level']:
        user[col] = lbl.fit_transform(user[col])
        user[col] = remove_lowcase(user[col])

    return user

def gen_item_diff(df,feat):
    stat_feat = ['mean','sum','std','var']
    df['diff'] = df[feat].diff(periods=1)
    df.loc[df['diff_item']!=0,'diff'] = np.nan
    df['diff'] = df['diff'].fillna(method='bfill')
    df_diff = df.groupby('item_id')['diff'].agg(stat_feat).reset_index()
    df_diff.columns = ['item_id'] + ['%s_%s_' % (feat,'diff') + col for col in stat_feat]
    df_diff.fillna(0,inplace=True)

    return df_diff

def gen_item_feat():
    """
    提取广告商品的固有属性
    """
    items = pd.read_csv("./data/item_file.csv",header=0)
    for i in range(1, 3):
        items['item_category_list_' + str(i)] = lbl.fit_transform(items['item_category_list'].map(
            lambda x: str(str(x).split(';')[i]) if len(str(x).split(';')) > i else '')) 
    for i in range(10):
        items['item_property_list_' + str(i)] = lbl.fit_transform(items['item_property_list'].map(
            lambda x: str(str(x).split(';')[i]) if len(str(x).split(';')) > i else ''))
    del items['item_category_list']
    del items['item_property_list']

    for col in ['item_brand_id', 'item_city_id']:
        items[col] = lbl.fit_transform(items[col])

    item_day = pd.read_csv('./data/item_day.csv',header=0)
    sale_level = item_day.groupby('item_id')['item_sales_level'].agg({'item_sale_level_std':'std'}).reset_index()
    price_level = item_day.groupby('item_id')['item_price_level'].agg({'item_price_level_std':'std'}).reset_index()
    coll_level = item_day.groupby('item_id')['item_collected_level'].agg({'item_coll_level_std':'std'}).reset_index()
    pv_level = item_day.groupby('item_id')['item_pv_level'].agg({'item_pv_level_std':'std'}).reset_index()

    items = pd.merge(items,sale_level,how='left',on='item_id')
    items = pd.merge(items,price_level,how='left',on='item_id')
    items = pd.merge(items,coll_level,how='left',on='item_id')
    items = pd.merge(items,pv_level,how='left',on='item_id')
    items.fillna(0,inplace=True)

    # ------------差分统计-----------------
    item_day['context_day'] = item_day['context_day'].replace(31,0)
    item_day = item_day.sort_values(by=['item_id','context_day'])
    item_day['diff_item'] = item_day['item_id'].diff(periods=1)

    df_diff = gen_item_diff(item_day.copy(),'item_sales_level')
    items = pd.merge(items,df_diff,how='left',on='item_id')

    df_diff = gen_item_diff(item_day.copy(),'item_price_level')
    items = pd.merge(items,df_diff,how='left',on='item_id')

    df_diff = gen_item_diff(item_day.copy(),'item_collected_level')
    items = pd.merge(items,df_diff,how='left',on='item_id')

    df_diff = gen_item_diff(item_day.copy(),'item_pv_level')
    items = pd.merge(items,df_diff,how='left',on='item_id')

    del df_diff
    del item_day
    gc.collect()

    return items

def gen_shop_feat():

    shop = pd.read_csv("./data/shop_file.csv",header=0)
    shop['context_day'] = shop['context_day'].replace(31,0)
    shop.replace(-1,shop.median(),inplace=True)
    shop2 = shop.copy()
    shop = shop.sort_values(by=['shop_id','context_day'])
    shop.drop_duplicates(subset='shop_id',keep='last',inplace=True)

    stat_feat = ['mean','sum','std']
    df_stat = shop2.groupby('shop_id')['shop_score_delivery'].agg(stat_feat).reset_index()
    df_stat.columns = ['shop_id'] + ['%s_%s_' % ('shop','deli') + col for col in stat_feat]
    shop = pd.merge(shop,df_stat,how='left',on='shop_id')

    df_stat = shop2.groupby('shop_id')['shop_review_positive_rate'].agg(stat_feat).reset_index()
    df_stat.columns = ['shop_id'] + ['%s_%s_' % ('shop','pos') + col for col in stat_feat]
    shop = pd.merge(shop,df_stat,how='left',on='shop_id')

    df_stat = shop2.groupby('shop_id')['shop_score_service'].agg(stat_feat).reset_index()
    df_stat.columns = ['shop_id'] + ['%s_%s_' % ('shop','serv') + col for col in stat_feat]
    shop = pd.merge(shop,df_stat,how='left',on='shop_id')

    df_stat = shop2.groupby('shop_id')['shop_score_description'].agg(stat_feat).reset_index()
    df_stat.columns = ['shop_id'] + ['%s_%s_' % ('shop','des') + col for col in stat_feat]
    shop = pd.merge(shop,df_stat,how='left',on='shop_id')

    for col in ['shop_review_num_level', 'shop_star_level']:
        shop[col] = lbl.fit_transform(shop[col])

    score_col = ['shop_review_positive_rate','shop_score_service','shop_score_delivery','shop_score_description']
    shop.loc[:,'shop_score_mean'] = shop[score_col].mean(1)
    shop.loc[:,'shop_score_std'] = shop[score_col].std(1)
    shop.loc[:,'shop_score_sum'] = shop[score_col].sum(1)

    scaler = StandardScaler()
    for col in score_col + ['shop_score_mean','shop_score_sum','shop_deli_mean','shop_deli_sum',\
                            'shop_pos_mean','shop_pos_sum','shop_serv_mean','shop_serv_sum',
                            'shop_des_mean','shop_des_sum']:
        shop[col] = scaler.fit_transform(shop[col].values.reshape(-1, 1))

    del shop['context_day']
    return shop


def gen_history_context_feat():
    context = pd.read_csv("./data/history_file.csv",header=0)
    # del context['instance_id']
    del context['context_id']
    del context['predict_category_property']
    gc.collect()

    user_click = context.groupby(['user_id','context_day'])['context_day'].agg({'click_times':'count'}).reset_index()
    user_click = user_click.pivot(index='user_id', columns='context_day', values='click_times').reset_index()
    user_click.columns = ['user_id','click_day1','click_day2','click_day3','click_day4','click_day5',
                        'user_click_day6','click_day31']
    
    user_click_col = ['click_day1','click_day2','click_day3','click_day4','click_day5','click_day31']
    user_click['user_click_day_count'] = user_click[user_click_col].count(1)
    user_click.fillna(0,inplace=True)
    user_click['user_click_day_mean'] = user_click[user_click_col].mean(1)
    user_click['user_click_day_sum'] = user_click[user_click_col].sum(1)
    user_click['user_click_day_std'] = user_click[user_click_col].std(1)
    user_click['user_click_all'] = user_click['user_click_day_sum'] + user_click['user_click_day6']
    user_click['user_click_diff_pre_6'] = user_click['user_click_day6'] - user_click['user_click_day_mean']
    user_click['user_click_day6_prob'] = user_click['user_click_day6'] / user_click['user_click_all']
    user_click.drop(user_click_col,axis=1,inplace=True)

    item_imp = context.groupby(['item_id','context_day'])['context_day'].agg({'imp_times':'count'}).reset_index()
    item_imp = item_imp.pivot(index='item_id', columns='context_day', values='imp_times').reset_index()
    item_imp.columns = ['item_id','imp_day1','imp_day2','imp_day3','imp_day4','imp_day5',
                        'item_imp_day6','imp_day31']
    
    item_imp_col = ['imp_day1','imp_day2','imp_day3','imp_day4','imp_day5','imp_day31']
    item_imp.fillna(0,inplace=True)
    item_imp['item_imp_day_mean'] = item_imp[item_imp_col].mean(1)
    item_imp['item_imp_day_sum'] = item_imp[item_imp_col].sum(1)
    item_imp['item_imp_day_std'] = item_imp[item_imp_col].std(1)
    item_imp.drop(item_imp_col,axis=1,inplace=True)
    item_imp['item_imp_all'] = item_imp['item_imp_day_sum'] + item_imp['item_imp_day6']
    item_imp['item_imp_diff_pre_6'] = item_imp['item_imp_day6'] - item_imp['item_imp_day_mean']
    item_imp['item_imp_day6_prob'] = item_imp['item_imp_day6'] / item_imp['item_imp_all']


    item_ctr = context[context['context_day']!=6].groupby(['item_id'])['is_trade'].agg(
                                                    {'item_imp_times':'count','item_trade_num':'sum'}).reset_index()
    item_ctr['item_ctr_proba'] = 100*(item_ctr['item_trade_num']+0.001) / (item_ctr['item_imp_times']+0.001)
    item_imp = pd.merge(item_imp,item_ctr[['item_id','item_ctr_proba','item_trade_num']],how='left',on='item_id')
    del item_ctr
    gc.collect()

    shop_imp = context.groupby(['shop_id','context_day'])['context_day'].agg({'shop_imp_day':'count'}).reset_index()
    shop_imp = shop_imp.pivot(index='shop_id', columns='context_day', values='shop_imp_day').reset_index()
    shop_imp.columns = ['shop_id','shop_imp_day1','shop_imp_day2','shop_imp_day3','shop_imp_day4','shop_imp_day5',
                        'shop_imp_day6','shop_imp_day31']
    
    shop_imp_col = ['shop_imp_day1','shop_imp_day2','shop_imp_day3','shop_imp_day4','shop_imp_day5','shop_imp_day31']
    # shop_imp['shop_imp_day_count'] = shop_imp[shop_imp_col].count(1)
    shop_imp.fillna(0,inplace=True)
    shop_imp['shop_imp_day_mean'] = shop_imp[shop_imp_col].mean(1)
    shop_imp['shop_imp_day_sum'] = shop_imp[shop_imp_col].sum(1)
    shop_imp['shop_imp_day_std'] = shop_imp[shop_imp_col].std(1)
    shop_imp.drop(shop_imp_col,axis=1,inplace=True)
    shop_imp['shop_imp_all'] = shop_imp['shop_imp_day_sum'] + shop_imp['shop_imp_day6']
    shop_imp['shop_imp_diff_pre_6'] = shop_imp['shop_imp_day6'] - shop_imp['shop_imp_day_mean']
    shop_imp['shop_imp_day6_prob'] = shop_imp['shop_imp_day6'] / shop_imp['shop_imp_all']


    shop_ctr = context[context['context_day']!=6].groupby(['shop_id'])['is_trade'].agg(
                                                    {'shop_imp_times':'count','shop_trade_num':'sum'}).reset_index()
    shop_ctr['shop_ctr_proba'] = 100*(shop_ctr['shop_trade_num']+0.001) / (shop_ctr['shop_imp_times']+0.001)
    shop_imp = pd.merge(shop_imp,shop_ctr[['shop_id','shop_ctr_proba','shop_trade_num']],how='left',on='shop_id')
    del shop_ctr
    gc.collect()

    user_item = context[context['is_trade']==0].groupby(['user_id','item_id'])['context_day'].agg({\
                                                                'user_item_times':'count'}).reset_index()
    user_shop = context[context['is_trade']==0].groupby(['user_id','shop_id'])['context_day'].agg({\
                                                                'user_shop_times':'count'}).reset_index()

    return user_click, item_imp, shop_imp,user_item,user_shop


def gen_context_feat():

    context = pd.read_csv("./data/train_test_context_file.csv",header=0)
    context = context.sort_values(by=['user_id','context_timestamp'])
    context['context_hour'] = context['context_timestamp'].map(lambda x : datetime.fromtimestamp(x).hour)
    # 在某个商品上浏览的时长
    context['diff_time'] = context['context_timestamp'].diff(periods=1)
    # 是否有翻页，翻页导致前后两个浏览商品之间有较长时间空隙
    context['diff_page'] = context['context_page_id'].diff(periods=1)
    context['diff_user'] = context['user_id'].diff(periods=1)
    context.loc[context['diff_user']!=0,'diff_time'] = 0
    context.loc[context['diff_user']!=0,'diff_page'] = 0

    context['user_click_times_k'] = 0
    context.loc[context['diff_user']!=0,'user_click_times_k'] = 1
    user_click_times_k = context['user_click_times_k'].tolist()
    for k in range(1,len(user_click_times_k)):
        if user_click_times_k[k] == 0:
            user_click_times_k[k] = user_click_times_k[k-1] + 1
    context['user_click_times_k'] = user_click_times_k

    context['diff_time'] = context['diff_time'].map(lambda x: 0 if x > 1800 else x)

    item_look_time = context.groupby(['item_id'])['diff_time'].agg({'item_look_time':'sum','item_look_count':'count'}).reset_index()
    item_look_time.columns = ['item_id','item_look_time','item_look_count']
    item_look_time['item_look_per_time'] = item_look_time['item_look_time']/item_look_time['item_look_count']
    context = pd.merge(context,item_look_time,how='left',on='item_id')

    context['diff_page'] = context['diff_page'].map(lambda x: 0 if x < 0 else x)
    context['duration'] = (context['diff_time']+ 1) / (context['diff_page']+ 1)

    context = context.sort_values(by=['user_id','shop_id','context_timestamp'])
    context['click_shop_k'] = 0
    context['diff_user'] = context['user_id'].diff(periods=1)
    context.loc[context['diff_user']!=0,'click_shop_k'] = 1
    click_shop_k = context['click_shop_k'].tolist()
    for k in range(1,len(click_shop_k)):
        if click_shop_k[k] == 0:
            click_shop_k[k] = click_shop_k[k-1] + 1
    context['click_shop_k'] = click_shop_k

    context = context.sort_values(by=['user_id','item_id','context_timestamp'])
    context['u_click_item_k'] = 0
    context['diff_user'] = context['user_id'].diff(periods=1)
    context.loc[context['diff_user']!=0,'u_click_item_k'] = 1
    u_click_item_k = context['u_click_item_k'].tolist()
    for k in range(1,len(u_click_item_k)):
        if u_click_item_k[k] == 0:
            u_click_item_k[k] = u_click_item_k[k-1] + 1
    context['u_click_item_k'] = u_click_item_k

    # 用户点击商品店铺之后还会不会点击（倒序）
    context = context.sort_values(by=['user_id','shop_id','context_timestamp'],ascending=False)
    context['click_shop_k_last'] = 0
    context['diff_user'] = context['user_id'].diff(periods=1)
    context.loc[context['diff_user']!=0,'click_shop_k_last'] = 1
    click_shop_k_last = context['click_shop_k_last'].tolist()
    for k in range(1,len(click_shop_k_last)):
        if click_shop_k_last[k] == 0:
            click_shop_k_last[k] = click_shop_k_last[k-1] + 1
    context['click_shop_k_last'] = click_shop_k_last

    context = context.sort_values(by=['user_id','item_id','context_timestamp'],ascending=False)
    context['u_click_item_k_last'] = 0
    context['diff_user'] = context['user_id'].diff(periods=1)
    context.loc[context['diff_user']!=0,'u_click_item_k_last'] = 1
    u_click_item_k_last = context['u_click_item_k_last'].tolist()
    for k in range(1,len(u_click_item_k_last)):
        if u_click_item_k_last[k] == 0:
            u_click_item_k_last[k] = u_click_item_k_last[k-1] + 1
    context['u_click_item_k_last'] = u_click_item_k_last

    del context['diff_user']
    context = context.sample(frac=1)

    context['predict_hit'] = context['predict_category_property'].map(
                                        lambda x : 1 - x.count('-1')/len(x.split(';')))
    for i in range(5):
        context['predict_category_property_' + str(i)] = lbl.fit_transform(context['predict_category_property'].map(
                                                            lambda x: str(str(x).split(';')[i]) if len(str(x).split(';')) > i else ''))

    context['context_page_id'] = lbl.fit_transform(context['context_page_id'])
    # del context['instance_id']
    del context['context_id']
    del context['context_timestamp']
    del context['predict_category_property']

    return context

def static_feat(df,df_val, feature):
    df = df.groupby(feature)['is_trade'].agg(['sum','count']).reset_index()

    new_feat_name = feature + '_stas'
    df.loc[:,new_feat_name] = 100 * (df['sum'] + 0.001) / (df['count'] + 0.001)

    df_stas = df[[feature,new_feat_name]]
    df_val = pd.merge(df_val, df_stas, how='left', on=feature)

    return df_val

def gen_cross_static_feat(df):
    ad_feat_list = ['item_brand_id','item_city_id','item_price_level','item_sales_level','item_collected_level',\
                    'item_pv_level','shop_review_num_level','shop_star_level','context_page_id','item_category_list_1',\
                    'item_category_list_2','diff_time','user_click_times_k','duration','click_shop_k','user_click_day6',\
                    'user_click_day_sum','user_click_all','user_click_diff_pre_6','user_click_day_mean',\
                    'user_hour_cnt_1','user_hour_cnt_all','click_shop_k_last','u_click_item_k_last',\
                    'u_click_item_k','user_item_times','user_shop_times']
    user_feat_list = ['user_gender_id','user_age_level','user_occupation_id','user_star_level']
    df['index'] = list(range(df.shape[0]))
    for afeat in ad_feat_list:
        for ufeat in user_feat_list:

            concat_feat = afeat + '_' + ufeat
            df[concat_feat] = df[afeat].astype('str') + '_' +df[ufeat].astype('str')
            df[concat_feat] = lbl.fit_transform(df[concat_feat])
            df[concat_feat] = remove_lowcase(df[concat_feat])
            df[concat_feat] = df[concat_feat].astype(np.int32)
            print('labelencoder done')

            data = df[['index',concat_feat,'is_trade']].copy()

            training = data[~data.is_trade.isnull()]  
            training = training.reset_index(drop=True)
            predict = data[data.is_trade.isnull()]
            del data
            gc.collect()

            df_stas_feat = None
            kf = KFold(n_splits = 5,random_state=2018,shuffle=True)
            for train_index, val_index in kf.split(training):
                X_train = training.loc[train_index,:]
                X_val = training.loc[val_index,:]

                X_val = static_feat(X_train, X_val, concat_feat)
                df_stas_feat = pd.concat([df_stas_feat,X_val],axis=0)

            X_pred = static_feat(training, predict, concat_feat)
            df_stas_feat = pd.concat([df_stas_feat,X_pred],axis=0)

            del df_stas_feat['is_trade']
            del df_stas_feat[concat_feat]
            del training
            del predict
            gc.collect()

            df = pd.merge(df, df_stas_feat,how='left',on='index')
            new_feat_name = concat_feat + '_stas'
            df[new_feat_name] = pd.cut(df[new_feat_name], bins=int(0.4*len(df[concat_feat].unique())),labels=False)
            df[new_feat_name] = df[new_feat_name].astype(np.float32)

            print(afeat, ufeat,'done!')

    del df['index']
    return df

def gen_x2x_stas_feat(df):
    """
    统计每一个城市、品牌、类目。。。的配送服务、好评率。。。评分特征
    """
    stat_feat = ['min','mean','max','median','sum','std']
    df_city = df.drop_duplicates(subset=['item_city_id','shop_id'], keep='last')

    # ----------城市 to 配送服务-----------------
    df_stat = df_city.groupby('item_city_id')['shop_score_delivery'].agg(stat_feat).reset_index()
    df_stat.columns = ['item_city_id'] + ['%s_%s_' % ('city','delivery') + col for col in stat_feat]
    df = pd.merge(df,df_stat,how='left',on='item_city_id')

    # ----------城市 to 好评率-----------------
    df_stat = df_city.groupby('item_city_id')['shop_review_positive_rate'].agg(stat_feat).reset_index()
    df_stat.columns = ['item_city_id'] + ['%s_%s_' % ('city','positive') + col for col in stat_feat]
    df = pd.merge(df,df_stat,how='left',on='item_city_id')

    df_brand = df.drop_duplicates(subset=['item_brand_id','shop_id'], keep='last')
    # ----------品牌 to 配送服务-----------------
    df_stat = df_brand.groupby('item_brand_id')['shop_score_delivery'].agg(stat_feat).reset_index()
    df_stat.columns = ['item_brand_id'] + ['%s_%s_' % ('brand','delivery') + col for col in stat_feat]
    df = pd.merge(df,df_stat,how='left',on='item_brand_id')

    # ----------品牌 to 好评率-----------------
    df_stat = df_brand.groupby('item_brand_id')['shop_review_positive_rate'].agg(stat_feat).reset_index()
    df_stat.columns = ['item_brand_id'] + ['%s_%s_' % ('brand','positive') + col for col in stat_feat]
    df = pd.merge(df,df_stat,how='left',on='item_brand_id')

    # ----------品牌 to 描述-----------------
    df_stat = df_brand.groupby('item_brand_id')['shop_score_description'].agg(stat_feat).reset_index()
    df_stat.columns = ['item_brand_id'] + ['%s_%s_' % ('brand','des') + col for col in stat_feat]
    df = pd.merge(df,df_stat,how='left',on='item_brand_id')

    return df


def stas_x_click_times_hour(df):
    """
    统计每个用户、商店、广告在当前日期的前一天和当前日期之前的所有天的出现次数
    """
    # -----------当前时刻的前一小时------------------
    for h in range(1, 24): 
        previous = df.loc[df['context_hour'] == h - 1,['user_id','item_id','shop_id','instance_id']].copy()
        current = df.loc[df['context_hour'] == h,:].copy()
        user_cnt = previous.groupby(by='user_id').count()['instance_id'].to_dict()
        item_cnt = previous.groupby(by='item_id').count()['instance_id'].to_dict()
        shop_cnt = previous.groupby(by='shop_id').count()['instance_id'].to_dict()
        current.loc[:,'user_hour_cnt_1'] = current['user_id'].apply(lambda x: user_cnt.get(x))
        current.loc[:,'item_hour_cnt_1'] = current['item_id'].apply(lambda x: item_cnt.get(x))
        current.loc[:,'shop_hour_cnt_1'] = current['shop_id'].apply(lambda x: shop_cnt.get(x))
        current = current[['user_hour_cnt_1', 'item_hour_cnt_1', 'shop_hour_cnt_1', 'instance_id']]
        if h == 1:
            Curr_hour = current
        else:
            Curr_hour = pd.concat([Curr_hour, current])
    df = pd.merge(df, Curr_hour, on=['instance_id'], how='left')

    # -----------当前时刻之前的所有小时------------------
    for h in range(1, 24):
        previous_all_hour = df.loc[df['context_hour'] < h,['user_id','item_id','shop_id','instance_id']].copy()
        current = df.loc[df['context_hour'] == h,:].copy()
        user_cnt = previous_all_hour.groupby(by='user_id').count()['instance_id'].to_dict()
        item_cnt = previous_all_hour.groupby(by='item_id').count()['instance_id'].to_dict()
        shop_cnt = previous_all_hour.groupby(by='shop_id').count()['instance_id'].to_dict()
        current.loc[:,'user_hour_cnt_all'] = current['user_id'].apply(lambda x: user_cnt.get(x))
        current.loc[:,'item_hour_cnt_all'] = current['item_id'].apply(lambda x: item_cnt.get(x))
        current.loc[:,'shop_hour_cnt_all'] = current['shop_id'].apply(lambda x: shop_cnt.get(x))
        current = current[['instance_id','user_hour_cnt_all', 'item_hour_cnt_all', 'shop_hour_cnt_all']]
        if h == 1:
            Curr_hour = current
        else:
            Curr_hour = pd.concat([Curr_hour, current])
    df = pd.merge(df, Curr_hour, on=['instance_id'], how='left')
    
    return df



if __name__ == '__main__':
    # gen_user_feat()
    # gen_shop_feat()
    # gen_history_context_feat()
    # gen_context_feat()
    item = gen_item_feat()
    item.to_csv('item_feat.csv',index=False)













