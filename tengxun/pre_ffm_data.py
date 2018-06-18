# coding=utf-8

import pandas as pd
import os
import gc

import hashlib, csv, math, os, pickle, subprocess

def hashstr(str, nr_bins):
    return int(hashlib.md5(str.encode('utf8')).hexdigest(), 16)%(nr_bins-1)+1

def gen_hashed_fm_feats(feats, nr_bins = int(1e+6)):
    feats = ['{0}:{1}:1'.format(field-1, hashstr(feat, nr_bins)) for (field, feat) in feats]
    return feats

def gen_ffm_data(fileIN_path, fileOUT_path):
    one_hot_feature=['LBS','age','carrier','consumptionAbility','education','gender','creativeSize',\
                    'house','aid','advertiserId','campaignId',\
                    'adCategoryId', 'productId', 'productType']
    combine_feature=['interest1','interest2','interest4','interest3','interest5']
    vector_feature = ['topic1','topic2','topic3','os','ct','marriageStatus','kw1','kw2','kw3']

    print("reading data")
    f = open(fileIN_path,'r')
    features = f.readline().strip().split(',')
    print(features)
    data_dict = {}
    num = 0
    for line in f:
        datas = line.strip().split(',')
        for i,d in enumerate(datas):
            if features[i] not in data_dict.keys():
                data_dict[features[i]] = []
            data_dict[features[i]].append(d)
        num += 1
        # if num >10:
        #     break

    f.close()

    print(num)
    print("transforming data")
    ftrain =  open(fileOUT_path,'a+')

    for i in range(num):
        feats = []
        for j, f in enumerate(one_hot_feature,1):
            field = j
            feats.append((field, f+'_'+data_dict[f][i]))

        for j, f in enumerate(vector_feature,1):
            field = j + len(one_hot_feature)
            xs = data_dict[f][i].split(' ')
            # feat_dict = pickle.load(open('../data/posfeat/%s.pkl' % f, 'rb'))
            # xs = [x for x in xs if x in feat_dict.keys()]
            # print(xs)
            for x in xs:
                feats.append((field, f+'_'+x))

        for j, f in enumerate(combine_feature,1):
            field = j + len(one_hot_feature) + len(vector_feature)
            xs = data_dict[f][i].split(' ')
            # if f not in ['os','ct','marriageStatus']:
                # feat_dict = pickle.load(open('../data/posfeat/%s.pkl' % f, 'rb'))
            # xs = [x for x in xs if x in feat_dict.keys()]
            for x in xs:
                feats.append((field, 'aid_'+ data_dict['aid'][i]+'_'+f+'_'+x))
                feats.append((field, 'adCategoryId_'+ data_dict['adCategoryId'][i]+'_'+f+'_'+x))
                feats.append((field, 'productType_'+ data_dict['productType'][i]+'_'+f+'_'+x))


        feats = gen_hashed_fm_feats(feats)
        ftrain.write(data_dict['label'][i] + ' ' + ' '.join(feats) + '\n')

        if i % 500000 == 0:
            print(i)

    del data_dict
    gc.collect()
    ftrain.close()

def split_test():
    fi = open('./data/libffm/test.txt','r')
    num = 0
    fo = open('./data/libffm/test%s.txt' % num, 'a+')
    for line in fi:
        line = line.strip()
        fo.write(line)
        fo.write('\n')
        num += 1

        if num % 2500000 ==0:
            fo.close()
            fo = open('./data/libffm/test%s.txt' % num, 'a+')

    fo.close()
    fi.close()
    print(num)

def combine_data(inpath, outpath):
    fo = open(outpath,'a+')
    num = 0
    fi = open(inpath, 'r')
    for line in fi:
        line = line.strip()
        fo.write(line)
        fo.write('\n')
        num += 1

    fo.close()
    fi.close()
    print(num)



if __name__ == '__main__':
    for i in range(1,13):
        gen_ffm_data('./data/combine/merge_test%s.txt' % i, './data/tmp/test%s.txt' %i)

    # combine_data('./data/libffm/train2.txt', './data/libffm/train1.txt')
    # combine_data('./data/libffm/train3.txt', './data/libffm/train1.txt')


    # gen_ffm_data('../data/val_combine.txt', '../data/libffm/validation.txt')

    # gen_ffm_data('./data/test1_combine.txt', './data/libffm/test.txt')

    # split_test()















