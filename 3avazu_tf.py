# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch
import sys
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
from deepctr.models import *

import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
sess = tf.Session(config=config)
KTF.set_session(sess)

if __name__ == "__main__":
    epochs=2
    batch_size=512

    # data = pd.read_csv('/HDD_sdb/wyw/zsx/avazu/avazu_first_3d.csv')
    # data = pd.read_csv('/HDD_sdb/wyw/zsx/avazu/avazu_first_3d.csv',nrows=50)

    sparse_features = ['hour', 'C1', 'banner_pos', 'site_id', 'site_domain',
                       'site_category', 'app_id', 'app_domain', 'app_category', 'device_id',
                       'device_model', 'device_type', 'device_conn_type',  # 'device_ip', 
                       'C14',
                       'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', ]
    print('len(sparse_features)',len(sparse_features))  # 去掉id click device_ip   不用day  25-4=21

    # data[sparse_features] = data[sparse_features].fillna('-1', )
    target = ['click']


    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    # for feat in sparse_features:
    #     lbe = LabelEncoder()
    #     data[feat] = lbe.fit_transform(data[feat])

    # data.to_pickle('data_avazu_first_3d')
    # input('to pickle ok')

    data=pd.read_pickle('data_avazu_first_3d')
    print('read pickle ok')
    
    print(data[:5])
    print(data['day'].unique())

    # 2.count #unique features for each sparse field,and record dense feature field name

    fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique(), embedding_dim=8)
                              for feat in sparse_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(
        linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model

    # train, test = train_test_split(data, test_size=0.2,random_state=2020)
    train=data[data['day']<23]
    test=data[data['day']==23]
    val, test = train_test_split(test, test_size=0.8,random_state=2020)

    train_model_input = {name: train[name] for name in feature_names}
    val_model_input = {name: val[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

    # 4.Define Model,train,predict and evaluate

    model = DeepFM(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
                   task='binary')
    print('model',model)

    model.compile("adam", "binary_crossentropy",
                  metrics=["binary_crossentropy", ], )

    pred_ans = model.predict(test_model_input, batch_size)

    print("")    
    print('pred_ans.shape', pred_ans.shape)
    print("pred_ans tf float32 memory usage: %.3f MB" % (sys.getsizeof(pred_ans)/1024/1024))
    print("pred_ans tf float64 memory usage: %.3f MB" % (sys.getsizeof(pred_ans.astype(np.float64))/1024/1024))
    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))

    for epoch in range(epochs):
        print('epoch',epoch)
        model.fit(train_model_input, train[target].values,
                  batch_size=batch_size, epochs=1, validation_data=(val_model_input,val[target]), verbose=1)

        pred_ans = model.predict(test_model_input, batch_size)

        print("")
        print('pred_ans.shape', pred_ans.shape)
        print("pred_ans tf float32 memory usage: %.3f MB" % (sys.getsizeof(pred_ans)/1024/1024))
        print("pred_ans tf float64 memory usage: %.3f MB" % (sys.getsizeof(pred_ans.astype(np.float64))/1024/1024))
        print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
        print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))