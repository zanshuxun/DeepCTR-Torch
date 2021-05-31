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

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == "__main__":
    epochs = 2
    batch_size = 512

    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]
    target = ['label']

    try:
        # read data from pkl directly
        data = pd.read_pickle('data_criteo.pkl')
        print('read_pickle ok')
    except:
        # data = pd.read_csv('/HDD_sdb/wyw/xun/deepctr/criteo/criteo_12m.txt', delimiter='\t',nrows=100000)
        data = pd.read_csv('/HDD_sdb/wyw/xun/deepctr/criteo/criteo_12m.txt', delimiter='\t')
        data.columns = 'label,I1,I2,I3,I4,I5,I6,I7,I8,I9,I10,I11,I12,I13,C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11,C12,C13,C14,C15,C16,C17,C18,C19,C20,C21,C22,C23,C24,C25,C26'.split(
            ',')
        data[sparse_features] = data[sparse_features].fillna('-1', )
        data[dense_features] = data[dense_features].fillna(0, )
        for feat in sparse_features:
            lbe = LabelEncoder()
            data[feat] = lbe.fit_transform(data[feat])
        mms = MinMaxScaler(feature_range=(0, 1))
        data[dense_features] = mms.fit_transform(data[dense_features])

        data.to_pickle('data_criteo.pkl')
        print('to_pickle ok')

    print(data[:5])

    # 2.count #unique features for each sparse field,and record dense feature field name

    fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique(), embedding_dim=8)
                              for feat in sparse_features] + [DenseFeat(feat, 1, )
                                                              for feat in dense_features]
    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(
        linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model
    train, test = data[:10000000], data[10000000:]
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

    # 4.Define Model,train,predict and evaluate

    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'

    model = xDeepFM(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
                    task='binary', dnn_hidden_units=(),
                    l2_reg_embedding=1e-5, device=device)
    print('model', model)

    model.compile("adam", "binary_crossentropy",
                  # metrics=["binary_crossentropy", ], )
                  metrics=["binary_crossentropy", "auc"], )

    for epoch in range(epochs):
        print('epoch', epoch)
        model.fit(train_model_input, train[target].values,
                  batch_size=batch_size, epochs=1, verbose=1)

        pred_ans = model.predict(test_model_input, batch_size * 20)
        print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
        print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))
