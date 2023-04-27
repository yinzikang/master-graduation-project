#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""自定义的rnn网络，用于提取叠加的状态组成的状态中的信息

Write detailed description here

Write typical usage example here

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
4/18/23 8:16 PM   yinzikang      1.0         None
"""

import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class LSTMFeatureExtractor(BaseFeaturesExtractor):
    """
    rnn的特征提取器
    observation_space不用给定，他会自己调用，设置好后面两个参数即可
    observation_space.shape[-1]维升维到features_dim再送入lstm
    """

    def __init__(self, observation_space, features_dim, num_layers):
        super().__init__(observation_space, features_dim)
        self.input_dense = nn.Linear(in_features=observation_space.shape[-1], out_features=features_dim)
        self.lstm = nn.LSTM(input_size=features_dim,
                            hidden_size=features_dim,  # 输出层维度等于隐藏层维度
                            num_layers=num_layers,
                            batch_first=True)

    def forward(self, x):
        # x shape: (env_num, observation_range, observation_space.shape)
        output, _ = self.lstm(self.input_dense(x))
        # output shape: (env_num, observation_range, hidden_dim)
        feature = output[:, -1, :]
        # feature shape: (env_num, hidden_dim)
        return feature
