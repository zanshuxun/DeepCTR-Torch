# -*- coding: utf-8 -*-
import pytest

from deepctr_torch.models import DIFM
from ..utils import get_test_data, SAMPLE_SIZE, check_model, get_device


@pytest.mark.parametrize(
    'att_head_num,dnn_hidden_units,sparse_feature_num',
    [(1, (4,), 2), (2, (4, 4,), 2), (1, (4,), 1)]
)
def test_DIFM(att_head_num, dnn_hidden_units, sparse_feature_num):
    model_name = "DIFM"
    sample_size = SAMPLE_SIZE
    x, y, feature_columns = get_test_data(
        sample_size, sparse_feature_num=sparse_feature_num, dense_feature_num=sparse_feature_num)

    model = DIFM(linear_feature_columns=feature_columns, dnn_feature_columns=feature_columns,
                    att_head_num=att_head_num,
                    dnn_hidden_units=dnn_hidden_units, dnn_dropout=0.5, device=get_device())
    check_model(model, model_name, x, y)


if __name__ == "__main__":
    pass
