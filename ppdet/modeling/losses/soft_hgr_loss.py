from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.core.workspace import register, serializable
from ppdet.modeling import ops

'''
Maximize the correlations across multimodal-fused features 
extracted from MultiAttn through Soft-HGR loss.
'''
__all__ = ['SoftHGRLoss']


@register
@serializable
class SoftHGRLoss(nn.Layer):

    def __init__(self):
        super().__init__()

    '''
    Calculate the inner products between feature mappings.
    '''

    def feature_mapping(self, feature_X, feature_Y):
        feature_mapping_X_Y = paddle.mean(paddle.sum(feature_X * feature_Y, axis=-1), axis=0)

        return feature_mapping_X_Y

    '''
    Calculate the inner products between feature covariances. 
    '''

    def feature_covariance(self, feature_X, feature_Y):
        cov_feature_X = compute_covariance_matrix(feature_X)
        cov_feature_Y = compute_covariance_matrix(feature_Y)
        # We empirically find that scaling the feature covariance by a factor of 1 / num_samples
        # leads to enhanced training stability and improvements in model performances.
        feature_covariance_X_Y = paddle.trace(paddle.matmul(cov_feature_X, cov_feature_Y)) / self.num_samples
        return feature_covariance_X_Y

    def forward(self, f_t, f_a):
        self.num_samples = f_t.shape[0]

        feature_mapping = self.feature_mapping(f_t, f_a)
        feature_covariance = self.feature_covariance(f_t, f_a)
        soft_hgr_loss = feature_mapping - feature_covariance / 2

        loss = - soft_hgr_loss / self.num_samples

        return paddle.sum(-loss / self.num_samples, axis=-1)


def compute_covariance_matrix(data):
    """
    计算三维张量的协方差矩阵。

    参数:
    data (paddle.Tensor): 形状为 (batchsize, numfeatures, numsamples) 的三维张量。

    返回:
    paddle.Tensor: 协方差矩阵。
    """
    # 合并batchsize和numsamples维度
    data_reshaped = data.transpose((1, 0, 2)).reshape((data.shape[1], -1))

    # 计算均值
    mean = paddle.mean(data_reshaped, axis=0, keepdim=True)

    # 中心化数据
    centered_data = data_reshaped - mean

    # 计算协方差矩阵
    N = data_reshaped.shape[0]  # 观测值的数量
    cov_matrix = paddle.matmul(centered_data.T, centered_data) / (N - 1)

    return cov_matrix
