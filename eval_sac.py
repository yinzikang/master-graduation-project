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
import gym
import gym_custom
from gym_custom.envs.env_kwargs import env_kwargs
from gym_custom.utils.custom_loader import load_episode
import matplotlib.pyplot as plt
import numpy as np

# 任务参数
# 环境加载
env_name = 'TrainEnvVariableStiffnessAndPosture-v6'
test_name = 'cabinet surface with plan'
rl_name = 'SAC'
time_name = '04-04-20-54'
path_name = test_name + '/' + rl_name + '/' + time_name + '/'
itr = 400
mode = 1

if mode == 1:  # 评估中间模型
    logger_path = "eval_results/" + path_name + "model_" + str(itr)
    modeL_path = "train_results/" + path_name + "model_" + str(itr) + '_steps'
elif mode == 2:  # 评估最后模型
    logger_path = "eval_results/" + path_name + "model"
    modeL_path = "train_results/" + path_name + "model"
elif mode == 3:  # 评估最优模型
    logger_path = "eval_results/" + path_name + "best_model"
    modeL_path = "train_results/" + path_name + "best_model"

_, _, rl_kwargs = env_kwargs(test_name, save_flag=False)
env = gym.make(env_name, **rl_kwargs)
env.logger_init(logger_path)
# 模型加载
model = SAC.load(modeL_path, env=env)

# 评估
mean_reward, std_reward = evaluate_policy(model=model, env=env, n_eval_episodes=1, deterministic=True, render=False,
                                          callback=None, reward_threshold=None, return_episode_rewards=False, warn=True)
print(mean_reward, std_reward)
result_dict = load_episode(logger_path)
# plot
for idx, (name, series) in enumerate(result_dict.items()):
    num = series.shape[-1]
    plt.figure(idx + 1)
    plt.plot(series)
    plt.grid()
    plt.legend(np.linspace(1, num, num, dtype=int).astype(str).tolist())
    plt.title(name)
    plt.savefig(logger_path + '/' + name)
# plt.show()
