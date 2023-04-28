#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Write target here

Write detailed description here

Write typical usage example here

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
3/28/23 9:07 PM   yinzikang      1.0         None
"""
import gym
from envs.env_kwargs import env_kwargs
from envs.controller import orientation_error_quat_with_mat
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.env_checker import check_env
from eval_everything import eval_everything

task = 'cabinet surface with plan v7'
algo = 'TrainEnvVariableStiffnessAndPostureAndSM-v8'
logger_path = './rl_test_results/' + task + '/' + algo
_, _, rl_kwargs = env_kwargs(task)
# env = TrainEnv(**rl_kwargs)
env = gym.make(algo, **rl_kwargs)
# env = gym.make(algo, **dict(task_name=task))

if not check_env(env):
    print('check passed')

test_times = 10
eval_flag = True
render_flag = False
plot_fig = True
save_fig = False
zero_action_flag = False

for _ in range(test_times):
    env.reset()
    if render_flag:
        env.viewer_init()
    if plot_fig:
        env.logger_init(logger_path)
    R = 0
    while True:
        # Random action
        a = env.action_space.sample()
        if zero_action_flag:
            if len(a) == 6:
                a[:9] = 0.0
            elif len(a) == 13:
                a[:9] = 0.0
                a[12] = 0.0
            elif len(a) == 17:
                a[:9] = 0.0
                a[12] = 0.0
                a[16] = 0.0

        o, r, d, info = env.step(a)
        R += r
        if d:
            break

    print(info['terminal info'], ' ', R)

    if eval_flag:
        result_dict = env.logger.episode_dict
        eval_everything(env, result_dict, plot_fig, save_fig, logger_path)
