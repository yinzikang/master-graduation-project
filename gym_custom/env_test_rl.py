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
from stable_baselines3.common.env_checker import check_env

from envs.env_kwargs import env_kwargs
from eval_everything import eval_everything

# test_name = 'cabinet surface with plan v7'
test_name = 'cabinet drawer open with plan test'
# test_name = 'cabinet door open with plan'
env_name = 'TrainEnvVariableStiffnessAndPostureAndSM_v2-v8'
logger_path = './rl_test_results/' + test_name + '/' + env_name
_, _, rl_kwargs = env_kwargs(test_name)
# env = TrainEnv(**rl_kwargs)
env = gym.make(env_name, **rl_kwargs)
# env = gym.make(algo, **dict(task_name=task))

if not check_env(env):
    print('check passed')

test_times = 1
render_flag = True
plot_fig = True
save_fig = False
zero_action_flag = True

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

    if plot_fig:
        result_dict = env.logger.episode_dict
        eval_everything(env, result_dict, plot_fig, save_fig, logger_path)
