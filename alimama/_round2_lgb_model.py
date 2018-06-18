# -*- coding:utf-8 -*-

import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
import pickle
import os
import gc
import numpy as np
import _round2_gen_features
import warnings
warnings.filterwarnings("ignore")

lbl = LabelEncoder()

def get_xtype_col(train_df,xtype):
    dtype_df = train_df.dtypes.reset_index()
    dtype_df.columns = ['col','type']
    return dtype_df[dtype_df.type==xtype].col.values.tolist()

def transf_data(x):
    if x >= 2:
        return np.ceil(np.log(x*x))
    elif(x>=0 and x <2):
        return np.ceil(x)
    elif x < -2:
        return - np.ceil(np.log(x*x))
    elif (x >=-2 and x < 0):
        return - np.ceil(x)
    else:
        return 0


def concat_data():
    data_path = './data/training.txt'
    if os.path.exists(data_path):
        training = pd.read_csv('./data/training.txt',sep=',')
        predict = pd.read_csv('./data/predict.txt',sep=',')
        label = pickle.load(open('./data/label.pkl','rb'))
    else:
        """
        user = _round2_gen_features.gen_user_feat()
        item = _round2_gen_features.gen_item_feat()
        shop = _round2_gen_features.gen_shop_feat()
        user_click, item_imp, shop_imp ,user_item ,user_shop= _round2_gen_features.gen_history_context_feat()

        user = pd.merge(user,user_click,how='left',on='user_id')
        user.fillna(0,inplace=True)
        item = pd.merge(item,item_imp,how='left',on='item_id')
        item.fillna(0,inplace=True)
        shop = pd.merge(shop,shop_imp,how='left',on='shop_id')
        shop.fillna(shop.median(),inplace=True)
        del user_click
        del item_imp
        del shop_imp
        gc.collect()

        print(user.shape, item.shape, shop.shape, user_item.shape, user_shop.shape)

        data = _round2_gen_features.gen_context_feat()
        data = pd.merge(data, user, how='left', on='user_id')
        del user
        gc.collect()
        data = pd.merge(data, item, how='left', on='item_id')
        del item
        gc.collect()
        data = pd.merge(data, shop, how='left', on='shop_id')
        del shop
        gc.collect()
        data = pd.merge(data, user_item, how='left', on=['user_id','item_id'])
        del user_item
        gc.collect()
        data = pd.merge(data, user_shop, how='left', on=['user_id','shop_id'])
        del user_shop
        gc.collect()

        data = _round2_gen_features.gen_x2x_stas_feat(data)
        print('gen_x2x_stas_feat done')

        data = _round2_gen_features.stas_x_click_times_hour(data) 
        print('stas_x_click_times_hour done')

        # B 榜测试集，如果是 A 榜，则下面一段代码可以注释掉
        testb = pd.read_csv("data/round2_ijcai_18_test_b_20180510.txt", sep = "\s+")
        testb = testb[['instance_id','item_id']]

        predict = data[data.is_trade.isnull()]
        predict = predict[predict['instance_id'].isin(testb['instance_id'].values)]
        training = data[~data.is_trade.isnull()]
        data = pd.concat([training, predict],axis=0, ignore_index=True)
        del training
        del predict
        del testb
        gc.collect()
        # -------------
        """
        # data.to_csv('./data/data.txt',sep=',',index=False)
        data = pd.read_csv('./data/data.txt',sep=',')

        for col in ['user_id','item_id','shop_id']:
            data[col] = lbl.fit_transform(data[col])

        transfer_col = ['diff_time','user_click_times_k','duration','click_shop_k','user_click_day6','user_click_day_mean',\
                        'user_click_day_sum','user_click_all','user_click_diff_pre_6','item_imp_day6','item_imp_day_mean',\
                        'item_imp_day_sum','item_imp_all','item_imp_diff_pre_6','item_trade_num','shop_imp_day6',\
                        'shop_imp_day_mean','shop_imp_day_sum','shop_imp_all','shop_imp_day_std','shop_imp_diff_pre_6',\
                        'shop_trade_num','user_hour_cnt_1','item_hour_cnt_1','shop_hour_cnt_1','user_hour_cnt_all',\
                        'item_hour_cnt_all','shop_hour_cnt_all','item_look_time','item_look_count','item_look_per_time',\
                        'u_click_item_k','item_imp_day_std','user_item_times','user_shop_times','click_shop_k_last','u_click_item_k_last']
        transfer_col = [col for col in transfer_col if col in data.columns]
        for col in transfer_col:
            data[col] = data[col].map(lambda x : transf_data(x))
            data[col] = data[col].astype(np.int32)

        print('transfer done')

        int_col = ['diff_page','item_brand_id',\
                'item_city_id','item_price_level','item_sales_level','item_collected_level','item_pv_level',\
                'item_category_list_1','item_category_list_2','item_property_list_0','item_property_list_1',\
                'item_property_list_2','item_property_list_3','item_property_list_4','item_property_list_5',\
                'item_property_list_6','item_property_list_7','item_property_list_8','item_property_list_9',\
                ]
        for col in int_col:
            try:
                data[col] = data[col].astype(np.int32)
            except:
                continue

        print()

        float64_col = get_xtype_col(data,'float64')
        for col in float64_col:
            data[col] = data[col].astype(np.float32)
            print(col)

        data = _round2_gen_features.gen_cross_static_feat(data)

        training = data[~data.is_trade.isnull()]
        training = training.reset_index(drop=True)
        label = training.is_trade.values
        predict = data[data.is_trade.isnull()]

        del training['is_trade']
        del predict['is_trade']
        del data
        gc.collect()

        training.fillna(0, inplace=True)
        predict.fillna(0, inplace=True)
        
        training.to_csv('./data/training.txt',sep=',',index=False)
        predict.to_csv('./data/predict.txt',sep=',',index=False)
        pickle.dump(label,open('./data/label.pkl','wb'))


    print("training, predict shape:",training.shape, predict.shape)

    return training, label, predict

def lgb_predict(training,label,predict):
    drop_list = ['instance_id','index','context_day','user_id','item_id','shop_id','city_delivery_max',\
                'city_positive_max','item_price_level_std']
    col = [c for c in training.columns if c not in drop_list]
    training = training[col]
    # print(training['context_hour'].value_counts())

    val_x = training.loc[training['context_hour']==11,:]
    val_y = label[training.loc[training['context_hour']==11,:].index]

    train_x = training.loc[training['context_hour']!=11,:]
    train_y = label[training.loc[training['context_hour']!=11,:].index]

    del training
    del train_x['context_hour']
    del val_x['context_hour']
    gc.collect()

    test = predict[col]
    del test['context_hour']

    clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=51, reg_alpha=0.0, reg_lambda=1.,
        subsample_for_bin=60000,
        max_depth=5, n_estimators=1500, objective='binary',
        subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
        learning_rate=0.03, min_child_samples=30, random_state=142, n_jobs=-1
    )

    feature_list = train_x.columns.tolist()
    clf.fit(train_x, train_y, eval_set=[(train_x, train_y),(val_x, val_y)], feature_name=feature_list,\
            eval_metric='logloss',early_stopping_rounds=50)

    feat_imp = pd.Series(clf.feature_importance(), feature_list).sort_values(ascending=False)
    feat_imp.to_csv("tmp/train_feat_sort.txt", sep=' ',index=True,header=False)

    pred = clf.predict_proba(test)[:,1]
    predict['predicted_score'] = pred
    predict = predict[['instance_id', 'predicted_score']]

    sub = pd.read_csv("data/round2_ijcai_18_test_b_20180510.txt", sep = "\s+")
    sub = sub[['instance_id','item_id']]
    sub = pd.merge(sub,predict, on = 'instance_id', how = 'left')
    sub = sub.fillna(0)

    print(sub.shape)
    sub[['instance_id', 'predicted_score']].to_csv('sub/res8.txt',sep=" ",index=False, float_format='%.8f')
    print(sub['predicted_score'].describe())

    sub = pd.DataFrame()
    sub['true'] = val_y
    sub['pred'] = clf.predict_proba(val_x)[:,1]
    sub = sub.sort_values(by=['true','pred'])
    sub.to_csv('tmp/validationRes.txt',sep=" ",index=False)

    return pred


def lgbcv_predict(training,label,predict):
    drop_list = ['instance_id','context_hour','index','context_day','city_delivery_max',\
                'city_positive_max','item_price_level_std','user_gender_id','user_id',\
                'item_price_level_diff_std','item_price_level_diff_sum','item_price_level_diff_mean',\
                'item_price_level_diff_var','user_shop_times_user_occupation_id_stas',\
                'user_click_day_mean_user_occupation_id_stas','user_click_diff_pre_6_user_occupation_id_stas',\
                'user_click_day_mean_user_star_level_stas','user_click_day_mean','user_click_day6']
    col = [c for c in training.columns if c not in drop_list]
    training = training[col]
    test = predict[col]

    clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=41, reg_alpha=0.0, reg_lambda=3.,
        subsample_for_bin=50000,
        max_depth=-1, n_estimators=3000, objective='binary',
        subsample=0.6, colsample_bytree=0.6, subsample_freq=1,
        learning_rate=0.025, min_child_samples=30, random_state=42, n_jobs=-1
    )

    kf = KFold(n_splits = 5,random_state=222,shuffle=True)
    best_logloss = []
    pred_list = []
    for train_index, val_index in kf.split(training):
        X_train = training.loc[train_index,:]
        y_train = label[train_index]
        X_val = training.loc[val_index,:]
        y_val = label[val_index]

        clf.fit(X_train, y_train, eval_set=[(X_train, y_train),(X_val, y_val)],\
                eval_metric='logloss',early_stopping_rounds=80)
        
        predictors = [i for i in X_train.columns]
        feat_imp = pd.Series(clf.feature_importance(), predictors).sort_values(ascending=False)
        feat_imp.to_csv("tmp/feat_importance.txt", sep=' ',index=True,header=False)

        predi = clf.predict_proba(test)[:,1]
        pred_list.append(predi)
        best_logloss.append(clf.best_score_['valid_1']['binary_logloss'])
        gc.collect()

    pred = np.mean(np.array(pred_list),axis=0)
    predict['predicted_score'] = pred
    predict = predict[['instance_id', 'predicted_score']]

    sub = pd.read_csv("data/round2_ijcai_18_test_b_20180510.txt", sep = "\s+")
    sub = sub[['instance_id','item_id']]
    sub = pd.merge(sub,predict, on = 'instance_id', how = 'left')
    sub = sub.fillna(0)

    print(sub.shape)
    sub[['instance_id', 'predicted_score']].to_csv('sub/lgb_b6.txt',sep=" ",index=False, float_format='%.8f')

    print(best_logloss,'\n',np.mean(best_logloss),np.std(best_logloss))
    print(sub['predicted_score'].describe())

    return pred


if __name__ == '__main__':
    training,label,predict = concat_data()
    # label = label.astype(int)
    # len_train = training.shape[0]
    # training = pd.concat([training,predict],axis=0)

    # transfer_col = ['u_click_item_k_last','click_shop_k_last']
    # for col in transfer_col:
    #     training[col] = training[col].map(lambda x : transf_data(x))
    #     training[col] = training[col].astype(np.int32)

    # ad_feat_list = ['diff_time','user_click_times_k','duration','click_shop_k','user_click_day6','user_click_day_mean',\
    #                 'user_click_day_sum','user_click_all','user_click_diff_pre_6',\
    #                 'user_hour_cnt_1','user_hour_cnt_all','click_shop_k_last','u_click_item_k_last',\
    #                 'u_click_item_k','user_item_times','user_shop_times']
    # ad_feat_list = [col for col in ad_feat_list if col in training.columns]

    # user_feat_list = ['user_gender_id','user_age_level','user_occupation_id','user_star_level']
    # for afeat in ad_feat_list:
    #     for ufeat in user_feat_list:
    #         concat_feat = afeat + '_' + ufeat
    #         training[concat_feat] = training[afeat].astype('str') + '_' +training[ufeat].astype('str')
    #         training[concat_feat] = lbl.fit_transform(training[concat_feat])
    #         training[concat_feat] = _round2_gen_features.remove_lowcase(training[concat_feat])

    # predict = training[len_train:]
    # training = training[:len_train]


    lgbcv_predict(training,label,predict)





