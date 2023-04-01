#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Write target here

Write detailed description here

Write typical usage example here

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
3/29/23 7:56 PM   yinzikang      1.0         None
"""
import numpy as np
from sb3_contrib import RecurrentPPO
from module.jk5_env_v5 import TrainEnv
from module.env_kwargs import load_env_kwargs
from stable_baselines3.common.evaluation import evaluate_policy

_, _, env_kwargs = load_env_kwargs('cabinet surface with plan')
env = TrainEnv(**env_kwargs)
model = RecurrentPPO.load("ppo_recurrent")

obs = env.reset()
# cell and hidden state of the LSTM
lstm_states = None
num_envs = 1
# Episode start signals are used to reset the lstm states
episode_starts = np.ones((num_envs,), dtype=bool)
while True:
    action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    episode_starts = dones
    env.render()
    if True in dones:
        break
