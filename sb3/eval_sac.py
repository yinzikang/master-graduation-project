#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Write target here

Write detailed description here

Write typical usage example here

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
3/8/23 11:21 AM   yinzikang      1.0         None
"""
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from module.jk5_env_v6 import TrainEnvVariableStiffness as TrainEnv
from module.env_kwargs import env_kwargs
from utils.custom_loader import load_episode
import matplotlib.pyplot as plt
import numpy as np

test_name = 'cabinet surface with plan'
rl_name = 'SAC'
time_name = '04-01-16-23'
path_name = test_name + '/' + rl_name + '/' + time_name + '/'
itr = 0
# _, _, env_kwargs = load_env_kwargs(test_name)
_, _, env_kwargs = env_kwargs(test_name)
env = TrainEnv(**env_kwargs)
env.logger_init("eval_results/" + path_name + str(itr))
model = SAC.load("train_results/" + path_name + "model", env=env)

mean_reward, std_reward = evaluate_policy(model=model, env=env, n_eval_episodes=1, deterministic=True,
                                          render=False, callback=None, reward_threshold=None,
                                          return_episode_rewards=False, warn=True)
print(mean_reward, std_reward)
result_dict = load_episode("eval_results/" + path_name + str(itr))
for idx, (name, series) in enumerate(result_dict.items()):
    num = series.shape[-1]
    plt.figure(idx + 1)
    plt.plot(series)
    plt.grid()
    plt.legend(np.linspace(1, num, num, dtype=int).astype(str).tolist())
    plt.title(name)
    plt.savefig("eval_results/" + path_name + str(itr) + '/' + name)
# plt.show()
