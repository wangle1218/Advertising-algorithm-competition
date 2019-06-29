# -*- coding:utf-8 -*-

import pandas as pd
import pickle
import gc

def merge_data(train,ad,user):
    train = pd.merge(train,ad,how='left',on='aid')
    train = pd.merge(train,user,how='left',on='uid')
    return train

def remove_lowcase(se):
    count = dict(se.value_counts())
    se = se.map(lambda x : -1 if count[x]<10 else x)
    return se

def filterfeat(feat,feat_dict):
    feat = str(feat).split()
    feat = [str(w) for w in feat if w in feat_dict.keys()]
    feat = ' '.join(feat)
    return feat

def remove_nonfeat(train,test,feature):
    aid_list = train['aid'].unique()
    for aid in aid_list:
        data = train.loc[train['aid']==aid,feature].tolist()
        num = int(len(data) * 0.0003)
        if num < 30:
            num = 30
        feat_dict = {}
        for line in data:
            line = str(line).split()
            for f in line:
                if f not in feat_dict.keys():
                    feat_dict[f] = 1
                else:
                    feat_dict[f] += 1

        feat_dict = [f for f in feat_dict.keys() if feat_dict[f]>num]
        feat_dict = {f:i for i,f in enumerate(feat_dict)}
        
        train.loc[train['aid']==aid,feature] = train.loc[train['aid']==aid,feature].map(lambda x : filterfeat(x,feat_dict))
        test.loc[test['aid']==aid,feature] = test.loc[test['aid']==aid,feature].map(lambda x : filterfeat(x,feat_dict))
    return train[feature],test[feature]

def transform_feat(train,test):
    text_col = ['interest1','interest2','interest3','interest4','interest5','kw1','kw2','kw3',\
                'topic1','topic2','topic3','appIdAction', 'appIdInstall']

    for i,feature in enumerate(text_col):
        print(feature)
        train[feature].fillna('',inplace=True)
        test[feature].fillna('',inplace=True)

        train[feature], test[feature] = remove_nonfeat(train,test,feature)

    return train,test

if __name__ == '__main__':
    train = pd.read_csv('./data/final_competition_data/train.csv')
    test = pd.read_csv('./data/final_competition_data/test2.csv')
    df_ad = pd.read_csv('./data/final_competition_data/adFeature.csv')

    i =1
    aid_group = pickle.load(open('./data/tmp/aid_group.pkl','rb'))
    for gp in [[1,2],[3,4],[5,6],[7,8],[9,10],[11,12],[13,14],[15,16],[17,18],[19,20],[21,22],[23,24]]:
        aid_list = aid_group[gp[0]]
        aid_list2 = aid_group[gp[1]]
        aid_list.extend(aid_list2)

        userFeat_path1 = './data/final_competition_data/userFeature_%d.txt' % gp[0]
        df_user0 = pd.read_csv(userFeat_path1)
        userFeat_path2 = './data/final_competition_data/userFeature_%d.txt' % gp[1]
        df_user = pd.read_csv(userFeat_path2)
        df_user = pd.concat([df_user0,df_user],axis=0)
        del df_user0
        gc.collect()

        train_gp = train[train['aid'].isin(aid_list)]
        test_gp = test[test['aid'].isin(aid_list)]
        df_ad_gp = df_ad[df_ad['aid'].isin(aid_list)]

        train_gp.loc[train_gp['label']==-1,'label']=0
        test_gp['label']=-1

        train_gp = merge_data(train_gp,df_ad_gp,df_user)
        test_gp = merge_data(test_gp,df_ad_gp,df_user)

        train_gp, test_gp= transform_feat(train_gp,test_gp)

        new_train_path = './data/combine/merge_train%s.txt' % i
        new_test_path = './data/combine/merge_test%s.txt' % i

        train_gp.to_csv(new_train_path,sep=',',index=False)
        test_gp.to_csv(new_test_path,sep=',',index=False)
        i += 1

        print(gp,'done')





