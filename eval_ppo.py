#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Write target here

Write detailed description here

Write typical usage example here

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
3/8/23 11:21 AM   yinzikang      1.0         None
"""
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import gym
import gym_custom
from gym_custom.envs.env_kwargs import env_kwargs
from gym_custom.utils.custom_loader import load_episode
from stable_baselines3.common.env_util import make_vec_env
from eval_everything import eval_everything
import matplotlib.pyplot as plt
import numpy as np

# 任务参数
# 环境加载
env_name = 'TrainEnvVariableStiffnessAndPostureAndSM_v2-v8'
# test_name = 'cabinet surface with plan v7'
# test_name = 'cabinet drawer open with plan'
test_name = 'cabinet door open with plan'
rl_name = 'PPO'
# time_name = '05-07-13-06'
# time_name = '05-08-09-09'
time_name = '05-09-15-56'
path_name = test_name + '/' + rl_name + '/' + time_name + '/'
itr = 655360
mode = 3

eval_flag = True
save_fig = True
plot_fig = True
render = False
n_eval_episodes = 1

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
if eval_flag:
    env.logger_init(logger_path)
# 模型加载
model = PPO.load(modeL_path)

# 评估
mean_reward, std_reward = evaluate_policy(model=model, env=env, n_eval_episodes=n_eval_episodes, deterministic=True,
                                          render=render, callback=None, reward_threshold=None,
                                          return_episode_rewards=True, warn=True)
print(mean_reward, std_reward)

if eval_flag:
    result_dict = load_episode(logger_path)
    eval_everything(env, result_dict, plot_fig, save_fig, logger_path)
