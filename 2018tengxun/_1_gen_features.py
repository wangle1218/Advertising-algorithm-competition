# coding=utf-8

"""
creativeId:和aid一一对应，有173个。
可以不用的 ID 类广告属性：advertiserId,campaignId,creativeId,productId
adFeature categroy 属性：creativeSize,adCategoryId,productType,aid

userFeature
house:填充 0 ; 
appIdAction,appIdInstall,interest1,interest2,interest3,interest4,interest5,topic1,topic2,topic3,
kw1,kw2,kw3 填充' '字符串
categroy 属性：LBS,age,carrier,consumptionAbility,education,gender,house
存在多种状态的属性：ct,marriageStatus,os
"""

import pandas as pd 
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans, MiniBatchKMeans,DBSCAN
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import KFold
from scipy import sparse
import os
import gc
import math
import fasttext
import warnings
warnings.filterwarnings("ignore")


def gen_adfeat(df):
    lbl = LabelEncoder()
    df = df[['advertiserId','campaignId','creativeSize','adCategoryId','productType','productId','aid']]
    for col in ['advertiserId','campaignId','creativeSize','adCategoryId','productType','productId']:
        df[col] = lbl.fit_transform(df[col].astype('int'))

    return df

def remove_lowcase(se):
    count = dict(se.value_counts())
    se = se.map(lambda x : -1 if count[x]<10 else x)
    return se

def remove_low(se):
    row_list = se.tolist()
    dicts = {}
    for row in row_list:
        row = str(row).strip().split()
        for case in row:
            if case not in dicts.keys():
                dicts[case] = 1
            else:
                dicts[case] += 1
    def _map(x,dicts):
        x = str(x).split()
        x = [xx for xx in x if dicts[xx]>10]
        x = ' '.join(x)
        return x
    se = se.map(lambda x : _map(x,dicts))
    return se


def static_feat(df,df_val, feature):
    df = df.groupby(feature)['label'].agg(['sum','count']).reset_index()

    new_feat_name = feature + '_stas'
    df.loc[:,new_feat_name] = 100 * (df['sum'] + 0.001) / (df['count'] + 0.001)
    df.loc[:,new_feat_name] = df.loc[:,new_feat_name].astype(np.float16)
    df_stas = df[[feature,new_feat_name]]
    df_val = pd.merge(df_val, df_stas, how='left', on=feature)

    return df_val


def gen_combian_feat(df,gp):
    """
    ['aid','campaignId']
    ['age','gender','education','consumptionAbility','LBS']
    """
    lbl = LabelEncoder()
    path = './data/gen_train_file/combian_feat_%d.txt' % gp
    if os.path.exists(path):
        # df = pickle.load(open(path,'rb'))
        df = pd.read_csv(path)
    else:
        ad_feat_list = ['advertiserId','productType','campaignId','adCategoryId','productId']
        user_feat_list = ['LBS','age','consumptionAbility','house','education','gender']
        df['index2'] = list(range(df.shape[0]))
        for afeat in ad_feat_list:
            for ufeat in user_feat_list:

                concat_feat = afeat + '_' + ufeat
                df[concat_feat] = df['aid'].astype('str') + df[afeat].astype('str') + '_' +df[ufeat].astype('str')
                df[concat_feat] = lbl.fit_transform(df[concat_feat])
                df.loc[:,concat_feat] = df.loc[:,concat_feat].astype(np.int32)
                df[concat_feat] = remove_lowcase(df[concat_feat])
                # print("concat ")

                data = df[['index2',concat_feat,'label']].copy()

                training = data[data.label!=-1]  
                training = training.reset_index(drop=True)
                predict = data[data.label==-1]
                del data
                gc.collect()

                df_stas_feat = None
                kf = KFold(n_splits = 3,random_state=2018,shuffle=True)
                for train_index, val_index in kf.split(training):
                    X_train = training.loc[train_index,:]
                    X_val = training.loc[val_index,:]

                    X_val = static_feat(X_train,X_val, concat_feat)
                    df_stas_feat = pd.concat([df_stas_feat,X_val],axis=0)

                X_pred = static_feat(training, predict, concat_feat)
                df_stas_feat = pd.concat([df_stas_feat,X_pred],axis=0)

                del df_stas_feat['label']
                del df_stas_feat[concat_feat]
                del training
                del predict
                gc.collect()

                df = pd.merge(df, df_stas_feat,how='left',on='index2')
                print(afeat, ufeat,'done!')


        # for col in ['LBS','age','consumptionAbility','education','gender','os','ct','marriageStatus',\
        #             'house','carrier']:
        #     del df[col]
        del df['index2']
        gc.collect()

        # pickle.dump(df,open(path,'wb'))
        df.to_csv(path,index=False)


    return df


def gen_userfeat(df,gp):
    path = './data/gen_train_file/userfeat_feat_%d.txt' % gp
    if os.path.exists(path):
        # df = pickle.load(open(path,'rb'))
        df = pd.read_csv(path)
    else:
        # text_col = ['interest1','interest2','interest3','interest4',\
        #             'interest5','topic1','topic2','topic3','kw1','kw2','kw3']

        df['house'] = df['house'].fillna(0)
        df['LBS'] = df['LBS'].fillna(-1)
        # df['LBS'] = df['LBS'].astype(int)
        # df['LBS'] = remove_lowcase(df['LBS'])
        
        # for col in text_col:
        #     df[col] = df[col].fillna(' ')
        #     df[col + '_num'] = df[col].map(lambda x: len(str(x).split()))
        #     del df[col]
        #     gc.collect()

        df['interest_sum'] = df[['interest1_num','interest2_num','interest3_num','interest4_num','interest5_num']].sum(1)
        df['interest_mean'] = df[['interest1_num','interest2_num','interest3_num','interest4_num','interest5_num']].mean(1)
        df['interest_std'] = df[['interest1_num','interest2_num','interest3_num','interest4_num','interest5_num']].std(1)

        df['topic_sum'] = df[['topic1_num','topic2_num','topic3_num']].sum(1)
        df['topic_mean'] = df[['topic1_num','topic2_num','topic3_num']].mean(1)
        df['topic_std'] = df[['topic1_num','topic2_num','topic3_num']].std(1)

        df['kw_sum'] = df[['kw1_num','kw2_num','kw3_num']].sum(1)
        df['kw_mean'] = df[['kw1_num','kw2_num','kw3_num']].mean(1)
        df['kw_std'] = df[['kw1_num','kw2_num','kw3_num']].std(1)

        for col in ['ct','marriageStatus','os']:
            df[col + '_num'] = df[col].map(lambda x: len(str(x).split()))
        
        df['ct_0'] = df['ct'].astype('str').map(lambda x : 1 if '0' in x else 0)
        df['ct_1'] = df['ct'].astype('str').map(lambda x : 1 if '1' in x else 0)
        df['ct_2'] = df['ct'].astype('str').map(lambda x : 1 if '2' in x else 0)
        df['ct_3'] = df['ct'].astype('str').map(lambda x : 1 if '3' in x else 0)
        del df['ct']

        df['os_0'] = df['os'].astype('str').map(lambda x : 1 if '0' in x else 0)
        df['os_1'] = df['os'].astype('str').map(lambda x : 1 if '1' in x else 0)
        df['os_2'] = df['os'].astype('str').map(lambda x : 1 if '2' in x else 0)
        del df['os']

        
        for i in [0,1,2,3,4,5,6,9,10,11,12,13,14,15]:
            df['marriS_%d' % i] = df['marriageStatus'].map(lambda x : 1 if str(i) in str(x).split() else 0)
        del df['marriageStatus']

        print("user feature stat finish!")

    return df 


def filterfeat(feat,feat_dict,i):
    feat = str(feat).split()
    # print(feat)
    # feat0 = ['%s_' % i +str(w) for w in feat if w not in feat_dict.keys()]
    # feat0 = ['%sn'%i] * len(feat0)
    feat = ['%s_' % i +str(w) for w in feat if w in feat_dict.keys()]
    # feat.extend(feat0)
    feat = ' '.join(feat)
    return feat


def remove_nonfeat(df,feature,i):
    feat_dict = {}
    for line in df.loc[df['label']==1,feature]:
        # print(line)
        line = str(line).split()
        for w in line:
            if w not in feat_dict.keys():
                feat_dict[w] =1
            else:
                feat_dict[w] += 1
    big_dim = ['kw1','kw2','topic1','topic2']
    if feature in big_dim:
        renum = 60
    else:
        renum = 20
    feat_dict = [w for w in feat_dict.keys() if feat_dict[w] >= renum]
    feat_dict = {w:i for i,w in enumerate(feat_dict)}
    # feat_dict = pickle.load(open('./data/posfeat/%s.pkl' %feature ,'rb'))
    df[feature] = df[feature].map(lambda x : filterfeat(x,feat_dict,i))
    return df[feature]


def get_user_text_feat(df):
    text_col = ['interest1','interest2','interest5','kw1','kw2','topic1','topic2']
    big_dim = ['kw1','kw2','topic1','topic2']
    # for col in text_col:
    #     df[col] = remove_low(df[col])
        
    df_text = df[['uid'] + text_col].copy()
    df = df[['uid','interest1']]
    
    tfidf = CountVectorizer(ngram_range=(1, 1), max_df=0.8, min_df=5)
    normalizer = Normalizer(copy=False)
    for i,feature in enumerate(text_col):
        df_feat = df_text[['uid',feature]].copy()
        df_feat = remove_nonfeat(df_feat,feature,i)
        # df_feat = df_feat[~df_feat[feature].isnull()]
        # print('%s row:' % feature ,df_feat.shape[0])
        df_feat = df_feat.fillna(' ')
        X = tfidf.fit_transform(df_feat[feature])
        print('X.shape:',X.shape)
        df_feat = df_feat['uid']

        if feature in big_dim:
            n_components = 55
        else:
            n_components =10# int(0.35*(X.shape[1]))
        svd = TruncatedSVD(n_components)
        lsa = make_pipeline(svd, normalizer)
        X = lsa.fit_transform(X)
        X = np.round(X,6)
        X = pd.DataFrame(X)
        X.columns = [feature + '_lsa' + str(i) for i in range(X.shape[1])]
        # X[feature + '_lsa_sum'] = X.sum(1)
        # X[feature + '_lsa_mean'] = X.mean(1)
        X = X.astype(np.float32)
        df_feat = pd.concat([df_feat,X],axis=1)

        df = pd.merge(df, df_feat ,how='left', on='uid')

    df.fillna(0,inplace=True)
    del df['interest1']
    del df_text
    gc.collect()

    return df

def mapp(i,x):
    x = str(x).split()
    x = ['%s_'%i+v for v in x]
    return ' '.join(x)

def get_user_cv_feat(df):
    text_col = ['interest1','interest2','interest3','interest4','interest5','kw1','kw2','kw3',\
                'topic1','topic2','topic3']
    
    df['index'] = ''
    for i,feature in enumerate(text_col):
        print(feature)
        df[feature] = remove_nonfeat(df,feature,i) #修改函数remove_nonfeat返回形式为df[feature]

    for i,feature in enumerate(text_col):
        # df[feature] = df[feature].map(lambda x: mapp(i,x))
        df['index'] = df['index'] + ' ' + df[feature]
        df[feature + '_num'] = df[feature].map(lambda x: len(str(x).split()))
        del df[feature]
        gc.collect()

    # df = df.fillna()

    return df

def get_w2v(text,model,feat_dict):
    text = text.split()
    text = [w for w in text if w in feat_dict.keys()]
    w2v = np.zeros(50)
    for word in text:
        try:
            w2v += model[word]
        except:
            continue
    if len(text) == 0:
        return w2v
    else:
        return w2v/len(text)

def gen_w2v_feat(df):
    w2v_feat = ['kw1','kw2','topic1','topic2']

    df_w2v = df[['uid'] + w2v_feat]
    df = df['uid']
    for feat in w2v_feat:
        feat_dict = pickle.load(open('./data/posfeat/%s.pkl' % feat, 'rb'))
        w2v_mar = []
        model = fasttext.load_model('./w2v/%s-model.bin' % feat)
        for row in df_w2v[feat]:
            w2v_mar.append(get_w2v(str(row),model,feat_dict))

        w2v_mar = np.round(np.array(w2v_mar),6)
        w2v_mar = pd.DataFrame(w2v_mar)
        w2v_mar.columns = [feat + '_w2v_' + str(i) for i in range(50)]
        w2v_mar[feat + '_w2v_sum'] = w2v_mar.sum(1)
        w2v_mar[feat + '_w2v_mean'] = w2v_mar.mean(1)
        w2v_mar = w2v_mar.astype(np.float32)
        df = pd.concat([df, w2v_mar], axis=1)

        print(feat,"w2v done!")
    del model
    gc.collect()

    return df

















