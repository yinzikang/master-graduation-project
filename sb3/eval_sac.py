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
from stable_baselines3.common.evaluation import evaluate_policy
from module.jk5_env_v3 import TrainEnv, load_env_kwargs
from stable_baselines3.common.evaluation import evaluate_policy
env_kwargs = load_env_kwargs('desk')
env = TrainEnv(**env_kwargs)
model = SAC.load("test", env=env)


mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# Enjoy trained agent
vec_env = model.get_env()
obs = vec_env.reset()
for i in range(4*env_kwargs['rl_frequency']):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render()
