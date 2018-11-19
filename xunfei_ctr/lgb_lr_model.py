import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from scipy import sparse
from sklearn.metrics import log_loss
from sklearn.preprocessing import OneHotEncoder, scale, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold,KFold

from keras.layers import Input, Dense
from keras.models import Model
from keras.layers.core import Flatten, Dense, Dropout,Activation
from keras.callbacks import EarlyStopping
from keras import backend  as KK
from keras.regularizers import l2

import pickle
import warnings
import gc
import os
warnings.filterwarnings("ignore")


def apply_xgb(train,test,label):
    train = train.tocsr()
    test = test.tocsr()
    clf = xgb.XGBClassifier(max_depth=5, learning_rate=0.05, n_estimators=60, silent=True,
                            objective='binary:logistic', booster='gbtree', n_jobs=4, gamma=0, 
                            min_child_weight=1, max_delta_step=5, subsample=0.8, colsample_bytree=0.8, 
                            colsample_bylevel=1, reg_alpha=0.1, reg_lambda=3, scale_pos_weight=1, 
                            base_score=0.5, random_state=2018, seed=2018)

    clf.fit(train,label,eval_set=[(train,label),(train,label)], eval_metric="logloss")
    train_leaf= clf.apply(train)
    test_leaf= clf.apply(test)

    del train
    del test
    gc.collect()

    print(train_leaf.shape, test_leaf.shape)
    print(train_leaf[:5])

    enc = OneHotEncoder()
    transformed_training_matrix = enc.fit_transform(train_leaf)
    transformed_testing_matrix = enc.transform(test_leaf)

    print(transformed_testing_matrix.shape)

    return transformed_training_matrix,transformed_testing_matrix

def apply_lgb(train,test,label):
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'binary_logloss'},
        'num_leaves': 31,
        'num_boost_round': 40,
        'learning_rate': 0.1,
        'feature_fraction': 0.75,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'verbose': 0
    }

    lgb_train = lgb.Dataset(train,label=label)

    print('Start training...')
    # train
    gbm = lgb.train(params=params,
                    train_set=lgb_train,
                    valid_sets=lgb_train)

    train_leaf= gbm.predict(train, pred_leaf=True)
    test_leaf= gbm.predict(test, pred_leaf=True)

    print(train_leaf.shape, test_leaf.shape)
    print(train_leaf[:5])

    enc = OneHotEncoder()
    transformed_training_matrix = enc.fit_transform(train_leaf)
    transformed_testing_matrix = enc.transform(test_leaf)

    print(transformed_testing_matrix.shape)

    return transformed_training_matrix,transformed_testing_matrix

# -----------------------------

def combian_feat():
    if os.path.exists('../data/lgblr_train_csr.npz'):
        train_csr = sparse.load_npz('../data/lgblr_train_csr.npz')
        predict_csr = sparse.load_npz('../data/lgblr_predict_csr.npz')
        label = pickle.load(open('../data/lgblr_label.pkl','rb'))
        predict_result = pickle.load(open('../data/lgblr_test_id.pkl','rb'))
    else:
        # model1 生成的特征文件，用于生成 lgb/xgb 叶子节点特征，用LR 模型中作为组合特征
        data_path = '../data/predict_result.pkl'
        label,predict_result = pickle.load(open(data_path,'rb'))
        train = sparse.load_npz('../data/lgb1_train_csr.npz').tocsr()
        test = sparse.load_npz('../data/lgb1_predict_csr.npz').tocsr()

        # model 生成的特征文件，作为 LR 模型中的单特征
        base_train_csr = sparse.load_npz('../data/lgb2_train_csr.npz')
        base_predict_csr = sparse.load_npz('../data/lgb2_predict_csr.npz')

        # 使用 lgb 来生成叶子节点特征
        transformed_training_matrix,transformed_testing_matrix = apply_lgb(train,test,label)

        # 拼接两组特征
        train_csr = sparse.hstack(
            (sparse.csr_matrix(transformed_training_matrix), base_train_csr), 'csr').astype(
            'float32')
        predict_csr = sparse.hstack(
            (sparse.csr_matrix(transformed_testing_matrix), base_predict_csr), 'csr').astype('float32')
        print(train_csr.shape)

        del transformed_training_matrix,base_train_csr,base_predict_csr
        del transformed_testing_matrix
        gc.collect()

        scaler = StandardScaler(with_mean=False)
        train_csr = scaler.fit_transform(train_csr)
        predict_csr = scaler.transform(predict_csr)

        sparse.save_npz('../data/lgblr_train_csr.npz', train_csr)
        sparse.save_npz('../data/lgblr_predict_csr.npz', predict_csr)
        pickle.dump(predict_result, open('../data/lgblr_test_id.pkl','wb'))
        pickle.dump(label, open('../data/lgblr_label.pkl','wb'))

    return train_csr, predict_csr, label, predict_result

def log_loss(y_true, y_pred):
    logloss = KK.sum(KK.binary_crossentropy(y_true,y_pred), axis=-1)
    return logloss

def MLP_model(X_train,y_train,X_val,y_val,predict_csr):
    x_dim = X_train.shape[1]
    inputs = Input(shape=(x_dim,),sparse=True)
    x = Dense(5000, kernel_regularizer=l2(0.001))(inputs)
    x = Activation('relu')(x)
    x = Dropout(0.3, seed=24)(x)
    x = Dense(1000, kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.3, seed=24)(x)
    y_out = Dense(1, activation='sigmoid')(x)

    early_stopping = EarlyStopping(monitor='val_loss', patience=2, verbose=2)
    model = Model(input=inputs, output=y_out)
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=[log_loss])
    model.fit(X_train, y_train, nb_epoch=30, validation_data=(X_val,y_val),
                batch_size=256, verbose=1, callbacks=[early_stopping])
    pred_test = model.predict(predict_csr)

    return pred_test

if __name__ == '__main__':

    train_csr, predict_csr, label, predict_result = combian_feat()

    n_splits=5
    skf = StratifiedKFold(n_splits=n_splits, random_state=2018, shuffle=True)
    pred_list = []
    for i,(train_index, val_index) in enumerate(skf.split(train_csr,label)):
        print(i)
        X_train = train_csr[train_index]
        y_train = label[train_index]
        X_val = train_csr[val_index]
        y_val = label[val_index]

        y_testi = MLP_model(X_train,y_train,X_val,y_val,predict_csr)
        pred_list.append(y_testi)
        print("test mean:",np.mean(y_testi))

    pred = np.mean(np.array(pred_list),axis=0)
    predict_result['predicted_score'] = pred
    predict_result = predict_result.fillna(0)
    print(predict_result['predicted_score'].describe())

    predict_result[['instance_id', 'predicted_score']].to_csv('../output/lgb_mlp.csv',sep=",",
        index=False, float_format='%.8f')




