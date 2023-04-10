#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""基于gym.Env的实现，用于sb3环境

继承自v6,专用来参数优化
初始化关键词参数更加的简单

Write typical usage example here

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
3/2/23 11:08 AM   yinzikang      6.0         None
"""

from gym_custom.envs.env_kwargs import env_kwargs
from gym_custom.envs.jk5_env_v6 import TrainEnvVariableStiffness as VS
from gym_custom.envs.jk5_env_v6 import TrainEnvVariableStiffnessAndPosture as VSAP


class TrainEnvVariableStiffness(VS):
    def __init__(self, task_name):
        super().__init__(**env_kwargs(task_name)[-1])


class TrainEnvVariableStiffnessAndPosture(VSAP):
    def __init__(self, task_name):
        super().__init__(**env_kwargs(task_name)[-1])
