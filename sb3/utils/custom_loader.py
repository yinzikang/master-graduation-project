#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""eval结果的读取


@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
1/20/23 12:32 PM   yinzikang      1.0         None
"""
import os

import numpy as np


def load_episode(path):
    file_list = os.listdir(path)
    episode_dict = dict()
    for file in file_list:
        if 'npy' in file:
            episode_dict[file.split('.')[0]] = np.load(path + '/' + file)
    return episode_dict
