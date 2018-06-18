# -*- encoding:utf-8 -*-
# author: wangle
# GitHub: https://github.com/wangle1218

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import gc
import warnings
warnings.filterwarnings("ignore")


class CVStatistic(object):
    """
    K 折交叉统计交叉特征
    """
    def __init__(self, adFeat_list, userFeat_list, label):
        self.adFeat_list = adFeat_list
        self.userFeat_list = userFeat_list
        self.label = label

    def CVS(self, df):
        """
        df为DataFrame格式数据，包含训练集和测试集，
        """
        lbl = LabelEncoder()
        df['index'] = list(range(df.shape[0]))
        for afeat in self.adFeat_list:
            for ufeat in self.userFeat_list:
                concat_feat = afeat + '_' + ufeat
                df[concat_feat] = df[afeat].astype('str') + '_' +df[ufeat].astype('str')
                df[concat_feat] = lbl.fit_transform(df[concat_feat])
                df[concat_feat] = self._remove_lowcase(df[concat_feat])

                df_cv = df[['index',concat_feat, self.label]].copy()

                training = df_cv[~df_cv[self.label].isnull()]  
                training = training.reset_index(drop=True)
                predict = df_cv[df_cv[self.label].isnull()]
                del df_cv
                gc.collect()

                df_stas_feat = None
                kf = KFold(n_splits = 5,random_state=2018,shuffle=True)
                for train_index, val_index in kf.split(training):
                    X_train = training.loc[train_index,:]
                    X_val = training.loc[val_index,:]

                    X_val = self._statis_feat(X_train,X_val, concat_feat)
                    df_stas_feat = pd.concat([df_stas_feat,X_val],axis=0)

                X_pred = self._statis_feat(training, predict, concat_feat)
                df_stas_feat = pd.concat([df_stas_feat,X_pred],axis=0)

                del df_stas_feat[self.label]
                del df_stas_feat[concat_feat]
                del training
                del predict
                gc.collect()

                df = pd.merge(df, df_stas_feat,how='left',on='index')
                print(afeat, ufeat,'done!')

        del df['index']
        return df
                
    def _remove_lowcase(self, se):
        count = dict(se.value_counts())
        se = se.map(lambda x : -1 if count[x]<5 else x)
        return se


    def _statis_feat(self, df,df_val, feature):
        df[self.label] = df[self.label].replace(-1,0)

        df = df.groupby(feature)[self.label].agg(['sum','count']).reset_index()

        new_feat_name = feature + '_stas'
        df.loc[:,new_feat_name] = 100 * (df['sum'] + 0.001) / (df['count'] + 0.001)
        df[new_feat_name] = pd.cut(df[new_feat_name], bins=100,labels=False)
        df[new_feat_name] = df[new_feat_name].astype(np.int32)
        df_stas = df[[feature,new_feat_name]]
        df_val = pd.merge(df_val, df_stas, how='left', on=feature)

        return df_val


def test():
    df = pd.read_csv('train_test_data.csv', header=0)
    # 广告/商品特征（类别离散特征，不是连续特征）
    ad_feat_list = ['aid','advertiserId','campaignId','creativeSize','adCategoryId','productType','productId']
    # 用户特征（类别离散特征，不是连续特征）
    user_feat_list = ['LBS','age','consumptionAbility','education','gender','os','ct','marriageStatus','house','carrier']
    # 数据集中标签（y）的名称, y 取值为1或者-1/0，1为正样本，-1/0 为负样本
    label = 'label' 

    cvst = CVStatistic(ad_feat_list, user_feat_list, label)
    df = cvst.CVS(df)
    df.to_csv('training_test.csv',index=False)
    print(df.head())


if __name__ == '__main__':
    test()











