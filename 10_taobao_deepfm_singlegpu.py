# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch
import sys
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import *
from torch.optim import Adam


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == "__main__":
    epochs=10
    batch_size=256
    sparse_features = ['user', 'adgroup_id' ]
    print('len(sparse_features)',len(sparse_features))  # 去掉id click device_ip   不用day  25-4=21

    # data[sparse_features] = data[sparse_features].fillna('-1', )
    target = ['clk']

    try:
        # read data from pkl directly
        data=pd.read_pickle('taobao1kw.pkl')
        print('read_pickle ok')
    except:
        print('read data')
        data_folder='/HDD_sdb/wyw/xun/deepctr/alimama/'
        data = pd.read_csv(data_folder+'taobao1m.txt')
    
    print(data[:5])
    print(data.shape)
    # for feat in sparse_features:
    #     lbe = LabelEncoder()
    #     data[feat] = lbe.fit_transform(data[feat])

    # 2.count #unique features for each sparse field,and record dense feature field name

    fixlen_feature_columns = [SparseFeat(feat, data[feat].max()+1, embedding_dim=16)
                              for feat in sparse_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(
        linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model
    train=data[:800000]
    test=data[800000:]
    print('train.shape',train.shape)
    print('test.shape',test.shape)

    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

    # 4.Define Model,train,predict and evaluate

    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'

    model = DeepFM(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
                   task='binary',
                   l2_reg_embedding=1e-5, device=device)
    print('model',model)

    model.compile(Adam(model.parameters(),5e-5), "binary_crossentropy",
                  # metrics=["binary_crossentropy", ], )
                  metrics=["binary_crossentropy", "auc"], )

    for epoch in range(epochs):
        print('epoch',epoch)
        model.fit(train_model_input, train[target].values,
                  batch_size=batch_size, epochs=1, verbose=1)

        pred_ans = model.predict(test_model_input, batch_size*20)
        print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
        print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))
