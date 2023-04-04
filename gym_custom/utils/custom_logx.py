#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""保存各种数据的logger

来源自spinup/utils/logx，负责保存各种超参数文件、训练指标、环境状态变化
嫌弃针对一个episode的env内的信息保存不好，自己改

Write typical usage example here

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
1/26/23 12:07 PM   yinzikang      1.0         None
"""
import atexit
import json
import os
import os.path as osp
import time
import warnings

import joblib
import numpy as np
import torch
from spinup.utils.mpi_tools import proc_id, mpi_statistics_scalar
# from spinup.utils.serialization_utils import convert_json
from gym_custom.utils.custom_serialization_utils import convert_json  # 修改为自定义转换


class EpisodeLogger():
    """
    仿照EpochLogger的自定义EpisodeLogger，用于保存一个episode中的状态变化
    """

    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.episode_dict = dict()

    def store_buffer(self, **kwargs):
        if proc_id() == 0:
            for k, v in kwargs.items():
                if not (k in self.episode_dict.keys()):
                    self.episode_dict[k] = []
                self.episode_dict[k].append(v)

    def dump_buffer(self):
        if proc_id() == 0:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            for k, v in self.episode_dict.items():
                np.save(self.output_dir + '/' + k, np.array(v))
            print('log has been dumped to ' + self.output_dir)

    def reset_buffer(self):
        if proc_id() == 0:
            self.episode_dict = dict()
