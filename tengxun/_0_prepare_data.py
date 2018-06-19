
# coding=utf-8
# author:WL

import pandas as pd 
import numpy as np
import pickle
import os
import gc


def get_seed_group():
    train = pd.read_csv('../data/final_competition_data/train.csv',header = 0)
    test = pd.read_csv('../data/final_competition_data/test2.csv',header = 0)
    data = pd.concat([train,test],axis=0)
    print(data.shape)
    del train
    del test
    gc.collect()

    seed = pd.DataFrame(data['aid'].value_counts())
    seed['aid1'] = seed.index
    seed.columns = ['num','aid1']

    seed = seed.reset_index(drop=True)
    # print(seed.head(20))
    num_cut = []
    n = 1
    num = 0
    for i in range(seed.shape[0]):
        num += seed.loc[i,'num']
        if num > 2500000:
            print('num',num)
            num = seed.loc[i,'num']
            n += 1
        print('num',num)
        num_cut.append(n)
        
    seed['num_cut'] = num_cut
    print(seed['num_cut'].value_counts())

    aid_group = {}
    group = seed.num_cut.values
    aid = seed.aid1.values

    for i in range(len(aid)):
        if group[i] not in aid_group.keys():
            aid_group[group[i]] = [aid[i]]
        else:
            aid_group[group[i]].append(aid[i])

    pickle.dump(aid_group,open('../data/tmp/aid_group.pkl','wb'))

    for aid_id in aid_group.keys():
        aid_list = aid_group[aid_id]
        aid_uid = data.loc[data['aid'].isin(aid_list),'uid'].unique()
        pickle.dump(aid_uid,open('../data/tmp/aid_uid_%s.pkl' % aid_id, 'wb'))


def get_user_feature(uid_list,group_num):
    uid_list = {uid:i for i,uid in enumerate(uid_list)}
    userFeature_data = []
    with open('../data/final_competition_data/userFeature.data', 'r') as f:
        for i, line in enumerate(f):
            line = line.strip().split('|')
            if int(line[0].split()[-1]) in uid_list.keys():
                userFeature_dict = {}
                for each in line:
                    each_list = each.split(' ')
                    userFeature_dict[each_list[0]] = ' '.join(each_list[1:])
                userFeature_data.append(userFeature_dict)

        user_feature = pd.DataFrame(userFeature_data)
        user_feature.to_csv('../data/final_competition_data/userFeature_%d.txt' % group_num,sep=',' ,index=False)

        # with open('../data/final_competition_data/userFeature_%d.txt' % group_num, 'a+') as f:
        #     f.write(','.join(user_feature.columns.tolist()))
        #     f.write('\n')
        #     for i in range(user_feature.shape[0]):
        #         sample = user_feature.loc[i,:].tolist()
        #         sample = [str(w) for w in sample]
        #         sample = ','.join(sample)
        #         f.write(sample)
        #         f.write('\n')

    gc.collect()


if __name__ == '__main__':
    get_seed_group()

    aid_group = pickle.load(open('../data/tmp/aid_group.pkl','rb'))
    for gn in aid_group.keys():
        uid_list = pickle.load(open('../data/tmp/aid_uid_%s.pkl' % gn,'rb'))

        get_user_feature(uid_list,gn)
        print(gn,"done!")





