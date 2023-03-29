#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Write target here

Write detailed description here

Write typical usage example here

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
3/8/23 11:21 AM   yinzikang      1.0         None
"""
import torch as th
from stable_baselines3 import SAC
from module.jk5_env_v5 import TrainEnv
from module.env_kwargs import load_env_kwargs
from stable_baselines3.common.evaluation import evaluate_policy

_, _, env_kwargs = load_env_kwargs('cabinet surface with plan')
env = TrainEnv(**env_kwargs)
model = SAC.load("test", env=env)

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# Enjoy trained agent
vec_env = model.get_env()
obs = vec_env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render()
    if True in dones:
        break
print('ccc')