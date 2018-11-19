"""
PAI平台训练，处理特征大概一个小时，训练大概半小时，比mac笔记本快几倍
使用data_vali.txt作为线下验证集结果如下：
0.42744262121111853 0.0012049467161769245（logloos mean std）
-------验证集 F1 ----------
threshold: 0.37, f1-value: 0.7712, 1:0 numbers 21194:28806
threshold: 0.4, f1-value: 0.7662, 1:0 numbers 19196:30804
threshold: 0.43, f1-value: 0.7649, 1:0 numbers 18096:31904
threshold: 0.45, f1-value: 0.7617, 1:0 numbers 17235:32765
threshold: 0.47, f1-value: 0.7620, 1:0 numbers 16280:33720
threshold: 0.5, f1-value: 0.7562, 1:0 numbers 15317:34683
--------------------------
0    124266
1     75734 （阈值取的0.4）
"""

import pandas as pd
import numpy as np
import pickle,os,jieba,time,gc,re
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.feature_selection import chi2, SelectPercentile
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import lightgbm as lgb
import datetime
import warnings
warnings.filterwarnings("ignore")

def remove_lowcase(se):
    """过滤掉低频属性"""
    count = dict(se.value_counts())
    se = se.map(lambda x : -1 if count[x]<5 else x)
    return se

def statis_feat(train, test, col):
    """计算转化率、曝光次数、点击次数"""
    temp = train.groupby(col,as_index=False)['label'].agg(
        {col+'_click':'sum', col+'_count':'count'})
    temp[col+'_ctr'] = temp[col+'_click']/(temp[col+'_count']+3)
    test = pd.merge(test, temp, on=col, how='left')
    test[col+'_ctr'].fillna(test[col+'_ctr'].mean(), inplace=True)
    test[col+'_click'].fillna(test[col+'_click'].median(), inplace=True)
    test[col+'_count'].fillna(test[col+'_count'].median(), inplace=True)
    return test

def is_title_proba_maximum(query_dict,title):
    title = str(title)
    c = {}
    for i,j in query_dict.items():
        c[str(i).lower()] = float(j)

    if title.lower() in c:
        if float(c[title.lower()])>0.1:
            return 1
        else:
            return 0
    else:
        return 0

def getNumOfCommonSubstr(str1, str2):
    """计算最长公共子串，用于对prefix改写"""
    lstr1 = len(str1)
    lstr2 = len(str2)
    record = [[0 for i in range(lstr2+1)] for j in range(lstr1+1)]
    maxNum = 0
    p = 0
 
    for i in range(lstr1):
        for j in range(lstr2):
            if str1[i] == str2[j]:
                record[i+1][j+1] = record[i][j] + 1
                if record[i+1][j+1] > maxNum:
                    maxNum = record[i+1][j+1]
                    p = i + 1
    return str1[p-maxNum:p], maxNum

def min_edit(str1, str2):
    """计算两句子的最小编辑距离，用于对prefix改写"""
    matrix = [[i+j for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]
    for i in range(1,len(str1)+1):
        for j in range(1,len(str2)+1):
            if str1[i-1] == str2[j-1]:
                d = 0
            else:
                d = 1
            matrix[i][j] = min(matrix[i-1][j]+1,matrix[i][j-1]+1,matrix[i-1][j-1]+d)
    return matrix[len(str1)][len(str2)]

def get_prefix_loc_in_title(prefix,title):
    """计算查询词prefix出现在title中的那个位置，前、后、中、没出现"""
    if prefix not in title:
        return -1
    lens = len(prefix)
    if prefix == title[:lens]:
        return 0
    elif prefix == title[-lens:]:
        return 1
    else:
        return 2
    
def get_rp_prefix_in_title(prefix,title,mode='char'):
    """计算title对prefix的词、字级别的召回率、精确率"""
    if mode == 'char':
        prefix = list(prefix)
        title = list(title)
    else:
        prefix = list(jieba.cut(prefix))
        title = list(jieba.cut(title))  
    len_title = len(title)
    len_prefix = len(prefix)
    len_comm_xx = len(set(prefix) & set(title))
    
    recall = len_comm_xx / (len_prefix + 0.01)
    precision = len_comm_xx / (len_title + 0.01)
    acc = len_comm_xx / (len_title + len_prefix - len_comm_xx)
    return [recall,precision,acc]

def get_ngram_rp_prefix_in_title(prefix,title,mode='char'):
    """计算title对prefix的词、字级别的召回率、精确率（1-2gram）"""
    if mode == 'char':
        prefix = list(prefix)
        title = list(title)
    else:
        prefix = list(jieba.cut(prefix))
        title = list(jieba.cut(title))
    prefix_2gram = []
    for i in range(len(prefix)-1):
        prefix_2gram.append(prefix[i]+prefix[i+1])
    prefix.extend(prefix_2gram)
    
    title_2gram = []
    for i in range(len(title)-1):
        title_2gram.append(title[i]+title[i+1])
    title.extend(title_2gram)
    
    len_title = len(title)
    len_prefix = len(prefix)
    len_comm_xx = len(set(prefix) & set(title))
    
    recall = len_comm_xx / (len_prefix + 0.01)
    precision = len_comm_xx / (len_title + 0.01)
    acc = len_comm_xx / (len_title + len_prefix - len_comm_xx)
    return [recall,precision,acc]
    
def get_rp_query_in_title(query,title,mode='char'):
    """计算title对query中概率最大句子的词、字级别的召回率、精确率"""
    if len(query) == 0:
        return [-1,-1,-1]
    query = sorted(query.items(),key = lambda x:x[1],reverse = True)
    query_str = query[0][0]
    if float(query[0][1]) < 0.1:
        return [0,0,0]
    else:
        return get_rp_prefix_in_title(query_str,title,mode=mode)
        
def get_ngram_rp_query_in_title(query,title,mode='char'):
    """计算title对query中概率最大句子的词、字级别的召回率、精确率（1-2gram）"""
    if len(query) == 0:
        return [-1,-1,-1]
    query = sorted(query.items(),key = lambda x:x[1],reverse = True)
    query_str = query[0][0]
    if float(query[0][1]) < 0.1:
        return [0,0,0]
    else:
        return get_ngram_rp_prefix_in_title(query_str,title,mode=mode)
    
def get_semantic_vector():
    pass

def correct_prefix(prefix, query_list):
    """根据query_predition中的句子对prefix进行改写"""
    if len(prefix) > 2 and (prefix[0] == '.' or prefix[0] == ':'):
        prefix = prefix[1:]
    if prefix[:3] == r'%2c':
        prefix = prefix[3:]
    if len(query_list) == 0:
        return prefix
    else:
        query_list = sorted(query_list.items(),key = lambda x:x[1],reverse = True)
    if len(query_list) >= 2:
        query1 = query_list[0][0]
        query2 = query_list[1][0]
        comm_str,lens = getNumOfCommonSubstr(query1,query2)
        edit = min_edit(prefix,comm_str)
        if edit <= 1:
            prefix = comm_str
        elif re.search('[a-z]',prefix) and edit < 5:
            prefix = comm_str
        else:
            if len(prefix) < len(comm_str):
                k = comm_str.find(prefix)
                if k > -1:
                    comm_str = comm_str[k:]
                    comm_str_cut = list(jieba.cut(comm_str))
                    new_comm_str = comm_str_cut[0]
                    for w in comm_str_cut[1:]:
                        new_comm_str += w
                        if min_edit(prefix,new_comm_str) <= 2:
                            prefix = new_comm_str
                            break
    else:
        query = query_list[0][0]
        k = query.find(prefix)
        if k > -1:
            query = query[k:]
            query_cut = list(jieba.cut(query))
            new_query = query_cut[0]
            for w in query_cut[1:]:
                new_query += w
                if min_edit(prefix,new_query) <= 2 and len(prefix)<len(new_query):
                    prefix = new_query
                    break
        else:
            pass
    return prefix

def combian_tag_to_prefix(prefix_cut, tag):
    prefix_cut = str(prefix_cut).split()
    prefix_cut = [w+str(tag) for w in prefix_cut]
    return ' '.join(prefix_cut)

###########################################
#            读取数据，预处理                #
###########################################

def load_clearn_data():
    train = pd.read_table('./DataSets/oppo_data_ronud2_20181107/data_train.txt', 
        names= ['prefix','query_prediction','title','tag','label'],
        header= None, encoding='utf-8',low_memory=False,quoting=3).astype(str)
    val = pd.read_table('./DataSets/oppo_data_ronud2_20181107/data_vali.txt', 
        names = ['prefix','query_prediction','title','tag','label'],
        header = None, encoding='utf-8',low_memory=False,quoting=3).astype(str)
    test = pd.read_table('./DataSets/oppo_data_ronud2_20181107/data_test.txt',
            names = ['prefix','query_prediction','title','tag'],
            header = None, encoding='utf-8',low_memory=False,quoting=3).astype(str)
    train = train[train['label'] != '音乐' ]

    train['label'] = train['label'].astype(int)
    val['label'] = val['label'].astype(int)
    print(train['label'].value_counts())
    print(val['label'].value_counts())

    # print(train.loc[train['query_prediction']=='nan',:])
    train['query_prediction'].replace('nan','{}',inplace=True)
    val['query_prediction'].replace('nan','{}',inplace=True)
    test['query_prediction'].replace('nan','{}',inplace=True)

    train['prefix'] = train['prefix'].map(lambda x : str(x).lower().strip())
    val['prefix'] = val['prefix'].map(lambda x : str(x).lower().strip())
    test['prefix'] = test['prefix'].map(lambda x : str(x).lower().strip())
    # 将字符串转换成字典格式
    train['query_prediction'] = train['query_prediction'].map(lambda x : eval(str(x)))
    val['query_prediction'] = val['query_prediction'].map(lambda x : eval(str(x)))
    test['query_prediction'] = test['query_prediction'].map(lambda x : eval(str(x)))

    train['len_prefix'] = train['prefix'].map(lambda x: len(x))
    train['num_pred'] = train['query_prediction'].map(lambda x: len(x.keys()))

    test['len_prefix'] = test['prefix'].map(lambda x: len(x))
    test['num_pred'] = test['query_prediction'].map(lambda x: len(x.keys()))

    val['len_prefix'] = val['prefix'].map(lambda x: len(x))
    val['num_pred'] = val['query_prediction'].map(lambda x: len(x.keys()))
    
    pre_tag_ctr = train.groupby(['len_prefix','tag'])['label'].agg(
        {'pre_tag_ctr':'mean','pre_tag_sum':'sum'}).reset_index()
    pred_tag_ctr = train.groupby(['num_pred','tag'])['label'].agg(
        {'pred_tag_ctr':'mean','pred_tag_sum':'sum'}).reset_index()

    train['flag'] = 1
    val['flag'] = 2
    test['flag'] = 3
    data = pd.concat([train,val,test],axis=0, ignore_index=True)

    data = pd.merge(data,pre_tag_ctr,how='left',on=['len_prefix','tag'])
    data = pd.merge(data,pred_tag_ctr,how='left',on=['num_pred','tag'])

    print(data.shape)

    return data

###########################################
#            特征工程                      #
###########################################

def generate_features():
    
    data = load_clearn_data()
    train_num = data[data['flag']==1].shape[0]
    vali_num = data[data['flag']==2].shape[0]
    test_num = data[data['flag']==3].shape[0]

    train_y = data.loc[data['flag']==1,'label'].values
    vali_y = data.loc[data['flag']==2,'label'].values

    data['min_edit'] = data.apply(lambda x :
        min_edit(x['prefix'],x['title']), axis=1)
    data['len_pred'] = data['query_prediction'].map(lambda x: len(x))
    data['in_query_big'] = data.apply(lambda x:is_title_proba_maximum(
        x['query_prediction'],x['title']),axis=1)

    prefix_count = data['prefix'].value_counts()
    data['prefix'] = data.apply(lambda x :
        correct_prefix(x['prefix'],x['query_prediction']), axis=1) #if prefix_count[x['prefix']]< 10 else x['prefix']
    data['prefix_loc'] = data.apply(lambda x :
        get_prefix_loc_in_title(x['prefix'],x['title']), axis=1)
 
    char_level_prefix = data.apply(lambda x :
        get_rp_prefix_in_title(x['prefix'],x['title'],mode='char'), axis=1)
    char_level_prefix = [kk for kk in char_level_prefix]
    char_level_prefix = np.array(char_level_prefix)
    data['prefix_t_recall_char'] = char_level_prefix[:,0].tolist()
    data['prefix_t_precision_char'] = char_level_prefix[:,1].tolist()
    data['prefix_t_acc_char'] = char_level_prefix[:,2].tolist()
    
    word_level_prefix = data.apply(lambda x :
        get_rp_prefix_in_title(x['prefix'],x['title'],mode='word'), axis=1)
    word_level_prefix = [kk for kk in word_level_prefix]
    word_level_prefix = np.array(word_level_prefix)
    data['prefix_t_recall_word'] = word_level_prefix[:,0].tolist()
    data['prefix_t_precision_word'] = word_level_prefix[:,1].tolist()
    data['prefix_t_acc_word'] = word_level_prefix[:,2].tolist()
    
    char_ngram_level_prefix = data.apply(lambda x :
        get_ngram_rp_prefix_in_title(x['prefix'],x['title'],mode='char'), axis=1)
    char_ngram_level_prefix = [kk for kk in char_ngram_level_prefix]
    char_ngram_level_prefix = np.array(char_ngram_level_prefix)
    data['prefix_t_recall_char_ngram'] = char_ngram_level_prefix[:,0].tolist()
    data['prefix_t_precision_char_ngram'] = char_ngram_level_prefix[:,1].tolist()
    data['prefix_t_acc_char_ngram'] = char_ngram_level_prefix[:,2].tolist()
    
    word_ngram_level_prefix = data.apply(lambda x :
        get_ngram_rp_prefix_in_title(x['prefix'],x['title'],mode='word'), axis=1)
    word_ngram_level_prefix = [kk for kk in word_ngram_level_prefix]
    word_ngram_level_prefix = np.array(word_ngram_level_prefix)
    data['prefix_t_recall_word_ngram'] = word_ngram_level_prefix[:,0].tolist()
    data['prefix_t_precision_word_ngram'] = word_ngram_level_prefix[:,1].tolist()
    data['prefix_t_acc_word_ngram'] = word_ngram_level_prefix[:,2].tolist()
    
    char_level_query = data.apply(lambda x :
        get_rp_query_in_title(x['query_prediction'],x['title'],mode='char'), axis=1)
    char_level_query = [kk for kk in char_level_query]
    char_level_query = np.array(char_level_query)
    data['query_t_recall_char'] = char_level_query[:,0].tolist()
    data['query_t_precision_char'] = char_level_query[:,1].tolist()
    data['query_t_acc_char'] = char_level_query[:,2].tolist()
    
    word_level_query = data.apply(lambda x :
        get_rp_query_in_title(x['query_prediction'],x['title'],mode='word'), axis=1)
    word_level_query = [kk for kk in word_level_query]
    word_level_query = np.array(word_level_query)
    data['query_t_recall_word'] = word_level_query[:,0].tolist()
    data['query_t_precision_word'] = word_level_query[:,1].tolist()
    data['query_t_acc_word'] = word_level_query[:,2].tolist()
    
    char_ngram_level_query = data.apply(lambda x :
        get_ngram_rp_query_in_title(x['query_prediction'],x['title'],mode='char'), axis=1)
    char_ngram_level_query = [kk for kk in char_ngram_level_query]
    char_ngram_level_query = np.array(char_ngram_level_query)
    data['query_t_recall_char_ngram'] = char_ngram_level_query[:,0].tolist()
    data['query_t_precision_char_ngram'] = char_ngram_level_query[:,1].tolist()
    data['query_t_acc_char_ngram'] = char_ngram_level_query[:,2].tolist()
    
    word_ngram_level_query = data.apply(lambda x :
        get_ngram_rp_query_in_title(x['query_prediction'],x['title'],mode='word'), axis=1)
    word_ngram_level_query = [kk for kk in word_ngram_level_query]
    word_ngram_level_query = np.array(word_ngram_level_query)
    data['query_t_recall_word_ngram'] = word_ngram_level_query[:,0].tolist()
    data['query_t_precision_word_ngram'] = word_ngram_level_query[:,1].tolist()
    data['query_t_acc_word_ngram'] = word_ngram_level_query[:,2].tolist()

    data['prefix_list'] = data['prefix'].map(lambda x : ' '.join(jieba.cut(str(x))))
    data['title_list'] = data['title'].map(lambda x : ' '.join(jieba.cut(str(x))))
    
    # data['prefix_list'] = data.apply(lambda x: 
    #     combian_tag_to_prefix(x['prefix_list'],x['tag']), axis=1)
    # data['title_list'] = data.apply(lambda x: 
    #     combian_tag_to_prefix(x['title_list'],x['tag']), axis=1)

    # 组合字段转化率
    data['prefix_tag'] = data['prefix'].astype(str)+'_' + data['tag'].astype(str)
    data['prefix_title'] = data['prefix'].astype(str)+'_' + data['title'].astype(str)
    data['title_tag'] = data['title'].astype(str)+'_' + data['tag'].astype(str)
    data['prefix_title_tag'] = data['prefix'].astype(str)+'_'+data['title'].astype(str)+'_' + data['tag'].astype(str)

    data['tag'] = data['tag'].map(dict(
        zip(data['tag'].unique(), range(0, data['tag'].nunique()))))
    data['prefix'] = data['prefix'].map(dict(
        zip(data['prefix'].unique(), range(0, data['prefix'].nunique()))))
    data['title'] = data['title'].map(dict(
        zip(data['title'].unique(), range(0, data['title'].nunique()))))
    data['prefix_tag'] = data['prefix_tag'].map(dict(
        zip(data['prefix_tag'].unique(), range(0, data['prefix_tag'].nunique()))))
    data['prefix_title'] = data['prefix_title'].map(dict(
        zip(data['prefix_title'].unique(), range(0, data['prefix_title'].nunique()))))
    data['title_tag'] = data['title_tag'].map(dict(
        zip(data['title_tag'].unique(), range(0, data['title_tag'].nunique()))))
    data['prefix_title_tag'] = data['prefix_title_tag'].map(dict(
        zip(data['prefix_title_tag'].unique(), range(0, data['prefix_title_tag'].nunique()))))

    del data['query_prediction']
    data.to_csv('./DataSets/dataaa.txt',sep='\t',index=False)
    """
    data = pd.read_csv('./DataSets/dataaa.txt',sep='\t')
    print(data.columns)
    train_num = data[data['flag']==1].shape[0]
    vali_num = data[data['flag']==2].shape[0]
    test_num = data[data['flag']==3].shape[0]
    train_y = data.loc[data['flag']==1,'label'].values
    vali_y = data.loc[data['flag']==2,'label'].values
    """
    # 计算下列字段的交叉转化率
    items = ['prefix','title','prefix_tag','tag','prefix_title','title_tag','prefix_title_tag']
    data['index'] = list(range(data.shape[0]))
    for col in items:
        data[col] = remove_lowcase(data[col])
        df_cv = data[['index',col,'label','flag']].copy()

        train = df_cv.loc[df_cv['flag'] == 1,:].reset_index(drop=True)
        test = df_cv.loc[df_cv['flag'] != 1,:]

        temp = train.groupby(col,as_index=False)['label'].agg(
            {col+'_click':'sum', col+'_count':'count'})
        temp[col+'_ctr'] = temp[col+'_click']/(temp[col+'_count']+3)
        test = pd.merge(test, temp, on=col, how='left')
        test[col+'_ctr'].fillna(test[col+'_ctr'].mean(), inplace=True)
        test[col+'_click'].fillna(test[col+'_click'].median(), inplace=True)
        test[col+'_click'] = test[col+'_click'].map(lambda x: int(0.8*x))
        test[col+'_count'].fillna(test[col+'_count'].median(), inplace=True)
        test[col+'_count'] = test[col+'_count'].map(lambda x: int(0.8*x))

        df_stas_feat = None
        kf = KFold(n_splits=5,random_state=2018,shuffle=True)
        for train_index, val_index in kf.split(train):
            X_train = train.loc[train_index,:]
            X_val = train.loc[val_index,:]

            X_val = statis_feat(X_train,X_val, col)
            df_stas_feat = pd.concat([df_stas_feat,X_val],axis=0)

        df_stas_feat = pd.concat([df_stas_feat,test],axis=0)
        df_stas_feat.drop([col,'label','flag'], axis=1, inplace=True)
        data = pd.merge(data, df_stas_feat,how='left',on='index')

    del train; gc.collect()
    print(data.shape)

    # 计算prefix,title的奇异值分解向量乘积做相似性度量
    data['prefix_list'].fillna(' ', inplace=True)
    data['title_list'].fillna(' ', inplace=True)
    print(data[['prefix_list','title_list']].info())
    prefix_vec = data['prefix_list'].tolist()
    title_vec = data['title_list'].tolist()
    corpus = prefix_vec + title_vec

    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_df=0.8, min_df=5)
    normalizer = Normalizer(copy=False)
    svd = TruncatedSVD(30)
    lsa = make_pipeline(tfidf, normalizer, svd)

    lsa.fit(corpus)
    title_vec = lsa.transform(title_vec)
    prefix_vec = lsa.transform(prefix_vec)
    # print(prefix_vec[:5])

    dot_sim = []
    consin_sim = []
    for i in range(title_vec.shape[0]):
        dot_sim.append(np.dot(title_vec[i], prefix_vec[i].T))
        consin_sim.append(cosine_similarity([title_vec[i], prefix_vec[i]])[0][1])
    data['dot_sim'] = dot_sim
    data['consin_sim'] = consin_sim


    ## 语义向量交互，相应位置相减取绝对值,点乘
    search_query1 = np.abs(prefix_vec - title_vec)
    search_query2 = prefix_vec * title_vec
    data['svd1_sim_mean'] = search_query1.mean(1)
    data['svd1_sim_sum'] = search_query1.sum(1)
    data['svd2_sim_mean'] = search_query2.mean(1)
    data['svd2_sim_sum'] = search_query2.sum(1)

    
    # 提取1-2gram 使用卡方检验找出重要的词和词组(感觉，这个可能是造成过拟合的原因)
    cv = CountVectorizer(ngram_range=(1, 2), max_df=0.8, min_df=4)
    prefix_word = cv.fit_transform(data['prefix_list'].tolist())
    print(prefix_word.shape)

    feature_select = SelectPercentile(chi2, percentile=10)
    feature_select.fit(prefix_word[:train_num], train_y)
    ##
    # 将 prefix 剩下的语义词组进行 bag of word ，作为特征
    prefix_word = feature_select.transform(prefix_word)
    
    drop_col = ['flag','prefix_list','title_list','label','index']
    ##
    data.drop(drop_col, axis=1, inplace=True)
    data.fillna(-1,inplace=True)

    # 将类别特征 onehot
    cate_col = items + ['prefix_loc']
    one_hot_csr = sparse.csr_matrix((len(data), 0))

    enc = OneHotEncoder()
    for col in cate_col:
        data[col].replace(-1,999999,inplace=True)
        one_hot_csr = sparse.hstack((one_hot_csr,
            enc.fit_transform(data[col].values.reshape(-1, 1))),'csr')

    data.drop(cate_col, axis=1, inplace=True)
    print(data.shape, search_query1.shape, one_hot_csr.shape)

    # 合并所有特征
    print(data.info())
    data = np.c_[data.values,search_query1,search_query2]
    # data = sparse.hstack((data, prefix_word), 'csr')
    data = sparse.hstack((data, one_hot_csr), 'csr')
    print(data.shape)

    return data[:train_num],data[train_num:train_num+vali_num],data[-test_num:],train_y,vali_y

def lgbcv_predict(train,label,vali,vali_y,test):
    training_time = 0
    clf = lgb.LGBMClassifier(
                boosting_type='gbdt', num_leaves=121, reg_alpha=1.2, reg_lambda=1,
                max_depth=-1, n_estimators=5000, objective='binary',
                subsample=0.8, colsample_bytree=0.6, subsample_freq=1,
                learning_rate=0.08, random_state=2018, n_jobs=-1)

    kf = KFold(n_splits=5,random_state=2333,shuffle=True)
    best_logloss = []
    pred_list, sub_list = [],[]
    for i,(train_index, val_index) in enumerate(kf.split(train)):
        t0 = time.time()
        X_train = train[train_index]
        y_train = label[train_index]
        X_val = train[val_index]
        y_val = label[val_index]

        clf.fit(X_train, y_train, eval_set=[(X_train, y_train),(X_val, y_val)],\
                eval_metric='logloss',early_stopping_rounds=50,verbose=100)

        best_logloss.append(clf.best_score_['valid_1']['binary_logloss'])

        predi = clf.predict_proba(vali,num_iteration=clf.best_iteration_)[:,1]
        pred_list.append(predi)
        p = np.where(predi>=0.35,1,0)
        print("{} round offline validation f1 score: {:.4f}".format(i+1,f1_score(vali_y,p)))

        predi = clf.predict_proba(test,num_iteration=clf.best_iteration_)[:,1]
        sub_list.append(predi)

        t = (time.time() - t0) / 60
        training_time += t
        print("This round cost time: {:.2f} minutes".format(t))

    pred_val = np.mean(np.array(pred_list),axis=0)
    pred_test = np.mean(np.array(sub_list),axis=0)

    print(best_logloss,'\n',np.mean(best_logloss),np.std(best_logloss))
    print("Total training time cost: {:.2f} minutes".format(training_time))
    return pred_val, pred_test


if __name__ == '__main__':
    start = time.time()
    data_path = './DataSets/label3.pkl'
    if os.path.exists(data_path):
        train_y,vali_y = pickle.load(open(data_path,'rb'))
        train = sparse.load_npz('./DataSets/lgb2_train_csr.npz').tocsr().astype('float32')
        vali = sparse.load_npz('./DataSets/lgb2_vali_csr.npz').tocsr().astype('float32')
        test = sparse.load_npz('./DataSets/lgb2_test_csr.npz').tocsr().astype('float32')
    else:
        train,vali,test,train_y,vali_y = generate_features()

        sparse.save_npz('./DataSets/lgb2_train_csr.npz', train)
        sparse.save_npz('./DataSets/lgb2_vali_csr.npz', vali)
        sparse.save_npz('./DataSets/lgb2_test_csr.npz', test)
        pickle.dump((train_y,vali_y),open(data_path,'wb'))

    """ 如果不把验证集进行训练，则注释下面两行 """
    #train_y = np.r_[train_y,vali_y]
    #train = sparse.vstack((train, vali), 'csr')
    print(train.shape)
    pred_val, pred_test = lgbcv_predict(train,train_y,vali,vali_y,test)

    print("-------验证集 F1 ----------")
    for v in [0.30,0.32,0.35,0.36,0.37,0.40,0.43]:
        p = np.where(pred_val>=v,1,0)
        print("threshold: {}, f1-score: {:.4f}, 1:0 numbers {}:{}".format(
            v,f1_score(vali_y,p), sum(p), len(p)-sum(p)))
    print("--------------------------")

    df_test = pd.DataFrame()
    df_test['predict'] = pred_test
    df_test.to_csv('sub-proba.csv' ,index=False,header=False)
    # 自己选一个阈值，提交，不满意还可以手动改上面的 sub-proba.csv 文件
    df_test['predict'] = df_test['predict'].map(lambda x: 1 if x>=0.36 else 0)
    print(df_test['predict'].value_counts())

    df_test.to_csv('result.csv' ,index=False,header=False)
    os.system('zip subfile_all.zip result.csv')
    
    t = (time.time() - start)/60
    print()
    print("running time: {:.2f} minutes\n".format(t))




