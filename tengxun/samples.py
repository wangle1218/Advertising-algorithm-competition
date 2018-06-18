# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import pickle
import os

if not os.path.exists('./data/combine'):
    os.makedirs('./data/combine')

def sample(data_path,frac):
    df = pd.read_csv(data_path)
    df_ad = pd.read_csv('./data/final_competition_data/adFeature.csv')

    # sample_data = None
    aid_group = pickle.load(open('./data/aid_group.pkl','rb'))
    for gp in aid_group.keys():
        aid_list = aid_group[gp]

        userFeat_path = './data/final_competition_data/userFeature_%d.txt' % gp
        df_user = pd.read_csv(userFeat_path,header=0)

        df_gp = df[df['aid'].isin(aid_list)]
        if frac < 1:
            df_gp = df_gp.sample(frac=frac)

        df_ad_gp = df_ad[df_ad['aid'].isin(aid_list)]

        sub_df = pd.merge(df_gp,df_ad_gp,how='left',on='aid')
        sub_df = pd.merge(sub_df,df_user,how='left',on='uid')

        if frac < 1:
            sub_df = sub_df.sample(frac=1)
            train_num = int(0.5*len(sub_df))
            sub_df[:train_num].to_csv('./data/combine/train_combine_%s.txt' % gp, sep=',',index=False)
            sub_df[train_num:].to_csv('./data/combine/val_combine_%s.txt' % gp, sep=',',index=False)
        else:
            sub_df.to_csv('./data/combine/test1_combine_%s.txt' % gp, sep=',',index=False)

        # sample_data = pd.concat([sample_data,sub_df],axis=0)
        print(gp," finish!")

    # return sample_data

def filterfeat(feat,feat_dict,i):
    feat = str(feat).split()
    feat = [str(w) for w in feat if w in feat_dict.keys()]
    feat = ' '.join(feat)
    return feat

def concat_data(data):
    df = None
    f = open('./data/%s_combine.txt' % data, 'a+')
    drop_feat = ['appIdAction', 'appIdInstall','topic3','interest3','interest4']
    for i in range(1,25):
        sub_df = pd.read_csv('./data/combine/%s_combine_%s.txt' % (data,i), sep=',')
        sub_df.drop(drop_feat,axis=1,inplace=True)
        if data == 'test1':
            sub_df['label'] = -1

        for j ,col in enumerate(['topic1','topic2','interest1','interest2','interest5','kw1','kw2']):
            feat_dict = pickle.load(open('./data/posfeat/%s.pkl' % col,'rb'))
            sub_df[col] = sub_df[col].map(lambda x : filterfeat(x,feat_dict,j))

        if i == 1:
            f.write(','.join(sub_df.columns.tolist()))
            f.write('\n')

        for k in range(sub_df.shape[0]):
            fo = sub_df.loc[k,:]
            fo = [str(w) for w in fo]
            f.write(','.join(fo))
            f.write('\n')

        df = pd.concat([df,sub_df[['aid','uid']]],axis=0)
    
        # df = pd.concat([df,sub_df],axis=0)
        print(i)

    f.close()

    # for j ,col in enumerate(['topic1','topic2','interest1','interest2','interest5','kw1','kw2']):
    #     feat_dict = pickle.load(open('./data/posfeat/%s.pkl' % col,'rb'))
    #     df[col] = df[col].map(lambda x : filterfeat(x,feat_dict,j))

    # df = df.sample(frac=1)

    # if data == 'test1':
    #     test = pd.read_csv('./data/final_competition_data/test1.csv')
    #     df = pd.merge(test, df, how='left', on=['aid','uid'])
    if data == 'test1':
        df.to_csv('./sub/%s_combine_sort.txt' % data, sep=',',index=False)

    print(df.shape)

def combine_train_val():
    fo = open('./example/data/train_combine_1.txt','a+')
    num = 0
    fi = open('./example/data/val_combine_1.txt', 'r')
    print(fi.readline())
    for line in fi:
        line = line.strip()
        fo.write(line)
        fo.write('\n')
        num += 1

    fo.close()
    fi.close()
    print(num)

def split_data():
    fi = open('./data/train_combine.txt','r')
    header = fi.readline().strip()
    num = 0
    k = 1
    fo = open('./data/train_combine%s.txt' % k, 'a+')
    for line in fi:
        if k ==1:
            fo.write(header)
            fo.write('\n')
        line = line.strip()
        fo.write(line)
        fo.write('\n')
        num += 1
        if num % 8500000 ==0:
            fo.close()
            k += 1
            fo = open('./data/train_combine%s.txt' % k, 'a+')
            fo.write(header)
            fo.write('\n')

    fo.close()
    fi.close()
    print(num)


def merge_sub():
    sub = None
    for i in range(1,13):
        path = './sub/res%s.csv' % i
        subi = pd.read_csv(path,header=None)

        path = './data/combian/merge_test%s.txt' %i
        testi = pd.read_csv(path)
        testi = testi[['aid','uid']] 
        testi['score'] = subi[0].values
        sub = pd.concat([sub,testi],axis=0)
    sub = sub.drop_duplicates(subset=['aid','uid'], keep='last')
    print(sub.describe())
    sub.to_csv('./sub/submission.csv',index=False,header=True,float_format='%.6f')

def merge():
    df1=pd.read_csv('./sub/7457.csv',header=0)
    df2 = pd.read_csv('./sub/746b.csv')
    df3 = pd.read_csv('./sub/7461.csv')
    df4 = pd.read_csv('./sub/7466.csv')
    df5 = pd.read_csv('./sub/746a.csv')
    df6 = pd.read_csv('./sub/744a.csv')
    df7 = pd.read_csv('./sub/7403.csv')



    a = df1['score'].values
    b = df2['score'].values
    c = df3['score'].values
    d = df4['score'].values
    e = df5['score'].values
    f = df6['score'].values
    g = df7['score'].values


    print('b,e corr',np.corrcoef(b,e))
    print('e,d corr',np.corrcoef(e,d))
    print('f,d corr',np.corrcoef(f,d))
    print('a,d corr',np.corrcoef(a,d))
    print('b,d corr',np.corrcoef(b,d))
    print('b,e corr',np.corrcoef(b,e))



    # res = 0.35*a + 0.2*b + 0.45*c
    res = 0.7*(b+c+d+e)/4 + 0.3*(a+f+g)/3
    print('res,d corr',np.corrcoef(res,d))
    # res = (b+c+d+e)/4
    df2['score'] = res
    df2.to_csv('./sub/submission.csv',index=False,header=True,float_format='%.6f')
    print(df2.describe())


if __name__ == '__main__':
    # data_path = './data/final_competition_data/train.csv'
    # sample(data_path, 0.6)

    # data_path = './data/final_competition_data/test1.csv'
    # sample(data_path, 1)

    # datas = ['test1']
    # for data in datas:
    #     concat_data(data)
    #     print(data,"finish!")

    merge()
    # split_data()




