#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Write target here

Write detailed description here

Write typical usage example here

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
3/28/23 9:59 PM   yinzikang      1.0         None
"""
import numpy as np
import gym
import gym_custom
from gym_custom.envs.env_kwargs import env_kwargs

task = "cabinet surface with plan"
env = gym.make('TrainEnvVariableStiffnessAndPosture-v6', **env_kwargs(task)[-1])
# env = gym.make('FetchSlideDense-v1')
# gym.make('Ant-v3')
print('ccc')
