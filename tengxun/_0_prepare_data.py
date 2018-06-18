
# coding=utf-8
# author:WL

"""
开源一个转CSV的代码。不需要大内存，只提取训练集和测试集中出现过得用户ID，分开存储，减小数据写入读取压力。。
做训练时直接全部 concat就行，或者分开训练
"""
from gensim.models.word2vec import Word2Vec
import pandas as pd 
import numpy as np
import pickle
import os
import gc
import fasttext


def get_seed_group():
    if os.path.exists('./data/aid_group.pkl'):
        aid_group = pickle.load(open('./data/aid_group.pkl','rb'))
        aid2uid = pickle.load(open('./data/aid2uid.pkl','rb'))
    else:
        train = pd.read_csv('./data/final_competition_data/train.csv',header = 0)
        test = pd.read_csv('./data/final_competition_data/test2.csv',header = 0)
        data = pd.concat([train,test],axis=0)
        del train
        del test
        gc.collect()

        seed = pd.DataFrame(data['aid'].value_counts())
        seed['aid1'] = seed.index
        seed.columns = ['num','aid1']

        seed = seed.reset_index(drop=True)
        num_cut = []
        n = 1
        num = 0
        for i in range(seed.shape[0]):
            num += seed.loc[i,'num']
            if num > 3500000:
                num = seed.loc[i,'num']
                n += 1
            num_cut.append(n)
            
        seed['num_cut'] = num_cut

        aid_group = {}
        group = seed.num_cut.values
        aid = seed.aid1.values

        for i in range(len(aid)):
            if group[i] not in aid_group.keys():
                aid_group[group[i]] = [aid[i]]
            else:
                aid_group[group[i]].append(aid[i])

        pickle.dump(aid_group,open('./data/aid_group.pkl','wb'))

        for aid_id in aid_group.keys():
            aid_list = aid_group[aid_id]
            aid_uid = data.loc[data['aid'].isin(aid_list),'uid'].unique()
            pickle.dump(aid_uid,open('./data/aid_uid_%s.pkl' % aid_id, 'wb'))

    # return aid_group,aid2uid


def get_user_feature(uid_list,group_num):
    uid_list = {uid:i for i,uid in enumerate(uid_list)}
    userFeature_data = []
    with open('./data/final_competition_data/userFeature.data', 'r') as f:
        for i, line in enumerate(f):
            line = line.strip().split('|')
            if int(line[0].split()[-1]) in uid_list.keys():
                userFeature_dict = {}
                for each in line:
                    each_list = each.split(' ')
                    userFeature_dict[each_list[0]] = ' '.join(each_list[1:])
                userFeature_data.append(userFeature_dict)

        user_feature = pd.DataFrame(userFeature_data)
        user_feature.to_csv('./data/final_competition_data/userFeature_%d.txt' % group_num,sep=',' ,index=False)
    gc.collect()


def train_w2v(file_path,features):
    model = fasttext.skipgram(file_path,'./w2v/%s-model' % features, dim=30, lr=0.1)
    return model 


# aid_group,aid2uid = get_seed_group()
aid_group = pickle.load(open('./data/aid_group.pkl','rb'))
for gn in aid_group.keys():
    # if gn != 11:
    # aid_list = aid_group[gn]
    # uid_list = []
    # for aid in aid_list:
    #     uid_list.extend(aid2uid[aid])
    uid_list = pickle.load(open('./data/aid_uid_%s.pkl' % gn,'rb'))

    get_user_feature(uid_list,gn)
    print(gn,"done!")

"""
w2v_feat = ['kw1','kw2','topic1','topic2']

aid_group = pickle.load(open('./data/aid_group.pkl','rb'))
for feat in w2v_feat:
    corpus = []
    for gn in aid_group.keys():
        aid_list = aid_group[gn]
        userFeat_path = './data/final_competition_data/userFeature_%d.csv' % gn
        df_user = pd.read_csv(userFeat_path)
        df_user = df_user[['uid',feat]]
        df_user.drop_duplicates(subset=feat,keep='last',inplace=True)
        for text in df_user[feat].tolist():
            if len(str(text).split()) > 1:
                corpus.append(str(text))

    with open('./w2v/w2vtext_%s.txt' % feat,'w') as f:
        for line in corpus:
            f.write(line)
            f.write('\n')

    train_w2v('./w2v/w2vtext_%s.txt' % feat, feat)
    print(feat," train done!")

"""



'悲伤', '悲惨', '悲戚', '悲痛', '悲切', '悲叹', '悲观', '悲悯', '哀叹', '哀愁' ,'哀怨', '哀痛', '哀思' ,'哀鸣', 
'怅然', '凄切', '痛苦', '痛切' ,'伤心', '伤感' ,'心如刀割' ,'痛不欲生', '痛心疾首', '悲痛欲绝' ,'欲哭无泪' ,'乐极生悲' ,'慷慨悲歌' 


'焦虑', '紧张', '沮丧' ,'痛苦' ,'愤怒' ,'抱怨', '自责', '悔恨', '担忧', '不安' ,'郁闷' ,'伤心',
 '难过', '失望' ,'堕落', '烦躁' ,'生气'


'忧郁','烦闷' ,'悲愤','忧伤','悲哀','怅惘','失落','惆怅','无聊','苦闷','酸楚','黯然','孤独','自卑','垂头丧气' ,'愁雾漫漫' ,'忧愁满腹' ,'满腔心事' ,'满腹愁肠',
'愁肠百结' ,'愁肠欲断', '愁肠寸断', '九回肠断', '回肠九转'
'百愁在心' ,'心事重重' ,'心情阴郁' ,'忧心忡忡', '忧心如焚'
'忧心如煎', '愁绪满怀', '忧愁担心', '食不下咽' ,'越肠百折'
'茶饭不思', '肚里泪下', '愁眉苦脸' ,'悲戚悲凉','悲怆'

sad_word = ['悲伤', '悲惨', '悲戚', '悲痛', '悲切', '悲叹', '悲观', '悲悯', '哀叹', '哀愁' ,'哀怨', '哀痛',\
            '哀思' ,'哀鸣', '怅然', '凄切', '痛苦', '痛切' ,'伤心', '伤感' ,'心如刀割' ,'痛不欲生', '痛心疾首',\
            '悲痛欲绝' ,'欲哭无泪' ,'乐极生悲' ,'慷慨悲歌' ,'苦恼','心酸','哀伤','心碎','惊恐','焦虑', '紧张', '沮丧' ,\
            '痛苦' ,'愤怒' ,'抱怨', '自责', '悔恨', '担忧', '不安' ,'郁闷' ,'伤心', '难过', '失望' ,'堕落', '烦躁' ,\
            '生气','忧郁','烦闷' ,'悲愤','忧伤','悲哀','怅惘','失落','惆怅','无聊','苦闷','酸楚','黯然','孤独',\
            '自卑','垂头丧气' ,'愁雾漫漫' ,'忧愁满腹' ,'满腔心事' ,'满腹愁肠','愁肠百结' ,'愁肠欲断', '愁肠寸断',\
            '九回肠断', '回肠九转','百愁在心' ,'心事重重' ,'心情阴郁' ,'忧心忡忡', '忧心如焚','忧心如煎', '愁绪满怀',\
            '忧愁担心', '食不下咽' ,'越肠百折','茶饭不思', '肚里泪下', '愁眉苦脸' ,'悲戚悲凉','悲怆']

happy_word = ['愉快', '畅快', '大喜', '狂喜', '欣喜', '高兴', '开心' ,'喜悦' ,'快慰' , '尽情', '快乐',
'美丽' ,'大方','可爱','帅气','无私','自律','恒心','创新','雄心','幽默','理解','虚心','自信',
'执着','挑战','热情','奉献','激情','爱心','自豪','渴望','信赖','卿卿我我','如胶似漆','情有独钟','干柴烈火','一往情深','珠联璧合',
'心心相印','比翼双飞','青梅竹马','琴瑟和鸣','志同道合','形影不离','情投意合','天作之合','比翼双飞','鸳鸯戏水','海枯石烂','海誓山盟','形影不离',
'两情相悦','至死靡它','至死不渝','持子之手','与尔偕老','恋爱','爱情','萌','萌萌的','女神','憨笑','爱心','爱上',
'兴奋','快乐','喜悦','愉快','畅快','欢畅','欢喜','欢腾','欢快','欣喜','潇洒',
'甜美','亲亲','灿烂','缠绵','恬豁','乐观','开朗','忻乐','乐呵','悠然','宝贝',
'美好','宠乐','乐怀','惬意','安心','舒心','温馨','玲珑','甜蜜','温馨','愉悦',
'比翼','连理','并蒂','合欢','良缘','天合','地设','同心','鸳鸯','意合','璧合',
'吉祥','白首','齐眉','良宵','甘甜','甜美','甘美','浪漫','舒服','开心','欣忭',
'怡悦','乐意','欢跃','雀跃','得意','快活','高兴','欢心','欢欣','温存','温暖',
'温和','微笑','欢笑','嬉笑','欢颜','思念','相思','婵娟','心悦','美满','幸福',
'可爱','美丽','思春','思韵','开怀','笑颜','笑容','喜笑','欢声','笑语','眉开',
'眷恋','爱恋','思恋','热恋','恬静','舒适','恬荡','恬雅','恬适','恬旷','恬逸',
'安然','坦然','祥和','甜润','甜适','甜爽','闺蜜','心系','唯爱','独钟','美滋',
'达令','心爱','心慕','亲爱','密爱','喜爱','如意','如愿','飘逸','逸致','宠爱']



