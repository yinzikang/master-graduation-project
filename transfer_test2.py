#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""用于刚度的迁移：
可以选择刚度、也可以选择采取完全一致的动作

Write detailed description here

Write typical usage example here

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
5/10/23 10:44 PM   yinzikang      1.0         None
"""

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import gym
import os
import gym_custom
from gym_custom.envs.env_kwargs import env_kwargs
from gym_custom.utils.custom_loader import load_episode
from stable_baselines3.common.env_util import make_vec_env
from eval_everything import eval_everything
import matplotlib.pyplot as plt
import numpy as np

def load_npy_files(path):
    results = {}
    for file in os.listdir(path):
        if file.endswith('.npy'):
            name = os.path.splitext(file)[0]
            file_path = os.path.join(path, file)
            data = np.load(file_path)
            results[name] = data
    return results

# 任务参数
# 环境加载
env_name = 'TrainEnvVariableStiffnessAndPostureAndSM_v2-v8'
test_name = 'cabinet surface with plan v7'
time_name = '04-30-17-20'
mode = 3
# test_name = 'cabinet drawer open with plan'
# time_name = '05-07-22-47'
# mode = 3
# test_name = 'cabinet door open with plan'
# time_name = '05-09-15-56'
# mode = 2
rl_name = 'PPO'
path_name = test_name + '/' + rl_name + '/' + time_name + '/'
itr = 655360

save_fig = False
plot_fig = False
render_flag = False

logger_path = None
if mode == 1:  # 评估中间模型
    logger_path = "eval_results/" + path_name + "model_" + str(itr)
    modeL_path = "train_results/" + path_name + "model_" + str(itr) + '_steps'
elif mode == 2:  # 评估最后模型
    logger_path = "eval_results/" + path_name + "model"
    modeL_path = "train_results/" + path_name + "model"
elif mode == 3:  # 评估最优模型
    logger_path = "eval_results/" + path_name + "best_model"
    modeL_path = "train_results/" + path_name + "best_model"

# 加载动作
result_dict = load_npy_files(logger_path)
action_series = result_dict['action']

_, _, rl_kwargs = env_kwargs(test_name, save_flag=False)
env = gym.make(env_name, **rl_kwargs)
env.reset()
if render_flag:
    env.viewer_init()
if plot_fig:
    env.logger_init('ccc')

R = 0
for a in action_series:
    o, r, d, info = env.step(a/env.action_limit)
    R += r
    if d:
        break

print(info['terminal info'], ' ', R)

if plot_fig:
    result_dict = env.logger.episode_dict
    eval_everything(env, result_dict, plot_fig, save_fig, logger_path)
