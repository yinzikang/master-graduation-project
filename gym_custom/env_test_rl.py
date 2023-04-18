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
from envs.jk5_env_v5 import TrainEnv
from envs.jk5_env_v6 import TrainEnvVariableStiffnessAndPosture as TrainEnv
from envs.env_kwargs import env_kwargs
from envs.controller import orientation_error_quat_with_mat
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.env_checker import check_env

task = 'cabinet surface with plan v7'
_, _, rl_kwargs = env_kwargs(task)
# env = TrainEnv(**rl_kwargs)
# env = gym.make('TrainEnvVariableStiffness-v6', **rl_kwargs)
env = gym.make('TrainEnvVariableStiffnessAndPosture-v7', **rl_kwargs)
# env = gym.make('TrainEnvVariableStiffnessAndPosture-v7', **dict(task_name=task))

if not check_env(env):
    print('check passed')

test_times = 1
view_flag = True
plot_flag = True
zero_action_flag = False

for _ in range(test_times):
    env.reset()
    if view_flag:
        env.viewer_init()
    if plot_flag:
        env.logger_init('./rl_test_results/'+task)
    R = 0
    while True:
        # Random action
        a = env.action_space.sample()
        if zero_action_flag:
            # a = np.zeros_like(a)
            a[:9] = 0.0
            a[12] = 0.0
            a[16] = 0.0
        # a[12:] = 0.0
        o, r, d, info = env.step(a)
        R += r
        if d:
            break

    print(info['terminal info'], ' ', R)

    if plot_flag:
        i = 0

        i += 1
        plt.figure(i)
        plt.plot(np.array(env.logger.episode_dict["xpos"]))
        plt.plot(np.array(env.logger.episode_dict["desired_xpos"]))
        plt.plot(np.array(env.init_desired_xposture_list)[:, :3])
        plt.legend(['x', 'y', 'z', 'dx', 'dy', 'dz', 'idx', 'idy', 'idz'])
        plt.title('xpos')
        plt.grid()

        i += 1
        plt.figure(i)
        plt.plot(np.array(env.logger.episode_dict["xpos"]) -
                 np.array(env.logger.episode_dict["desired_xpos"]))
        plt.legend(['x', 'y', 'z'])
        plt.title('xpos error')
        plt.grid()

        i += 1
        plt.figure(i)
        plt.plot(np.array(env.logger.episode_dict["xvel"])[:, :3])
        plt.plot(np.array(env.logger.episode_dict["desired_xvel"])[:, :3])
        plt.legend(['x', 'y', 'z', 'dx', 'dy', 'dz'])
        plt.title('xpos_vel')
        plt.grid()

        i += 1
        plt.figure(i)
        plt.plot(np.array(env.logger.episode_dict["desired_xacc"])[:, :3])
        plt.legend(['dx', 'dy', 'dz'])
        plt.title('xpos_acc')
        plt.grid()

        i += 1
        plt.figure(i)
        plt.plot(np.array(env.logger.episode_dict["xquat"]))
        plt.plot(np.array(env.logger.episode_dict["desired_xquat"]))
        plt.plot(np.array(env.init_desired_xposture_list)[:, 12:16])
        plt.legend(['x', 'y', 'z', 'w', 'dx', 'dy', 'dz', 'idw', 'idx', 'idy', 'idz', 'idw'])
        plt.title('xquat')
        plt.grid()

        i += 1
        plt.figure(i)
        plt.plot(np.array(env.logger.episode_dict["xvel"])[:, 3:])
        plt.plot(np.array(env.logger.episode_dict["desired_xvel"])[:, 3:])
        plt.legend(['x', 'y', 'z', 'dx', 'dy', 'dz'])
        plt.title('xmat_vel')
        plt.grid()

        i += 1
        plt.figure(i)
        orientation_error_buffer = []
        for j in range(len(env.logger.episode_dict["xquat"])):
            orientation_error_buffer.append(
                orientation_error_quat_with_mat(np.array(env.logger.episode_dict["desired_xmat"])[j].reshape([3,3]),
                                                np.array(env.logger.episode_dict["xmat"])[j].reshape([3,3])))
        plt.plot(orientation_error_buffer)
        plt.legend(['x', 'y', 'z'])
        plt.title('orientation_error')
        plt.grid()

        i += 1
        plt.figure(i)
        plt.plot(np.array(env.logger.episode_dict["qpos"]))
        plt.legend(['j1', 'j2', 'j3', 'j4', 'j5', 'j6'])
        plt.title('qpos')
        plt.grid()

        i += 1
        plt.figure(i)
        plt.plot(np.array(env.logger.episode_dict["qvel"]))
        plt.legend(['j1', 'j2', 'j3', 'j4', 'j5', 'j6'])
        plt.title('qvel')
        plt.grid()

        # i += 1
        # plt.figure(i)
        # plt.plot((np.array(xpos_buffer) - np.array(desired_xpos_buffer))[1:, :] /
        #          np.array(contact_force_buffer)[1:, :3])
        # plt.legend(['x', 'y', 'z'])
        # plt.title('1/stiffness')
        # plt.grid()

        i += 1
        plt.figure(i)
        plt.plot(np.array(env.logger.episode_dict["contact_force"]))
        plt.plot(np.array(env.logger.episode_dict["desired_force"])[:, 2] + 2.5)
        plt.plot(np.array(env.logger.episode_dict["desired_force"])[:, 2] - 2.5)
        plt.legend(['x', 'y', 'z', 'rx', 'ry', 'rz'])
        plt.title('force')
        plt.grid()

        i += 1
        plt.figure(i)
        plt.plot(np.array(env.logger.episode_dict["tau"]))
        plt.legend(['j1', 'j2', 'j3', 'j4', 'j5', 'j6'])
        plt.title('tau')
        plt.grid()

        i += 1
        plt.figure(i)
        plt.plot(np.array(env.logger.episode_dict["K"]))
        plt.legend(['x', 'y', 'z', 'rx', 'ry', 'rz'])
        plt.title('K')
        plt.grid()

        i += 1
        plt.figure(i)
        plt.plot(np.array(env.logger.episode_dict["observation"]))
        plt.legend(['px', 'py', 'pz', 'vx', 'vy', 'vz'])
        plt.title('observation')
        plt.grid()

        i += 1
        plt.figure(i)
        plt.plot(np.array(env.logger.episode_dict["action"]))
        # plt.legend(['a0', 'a1', 'a2', 'a3', 'a4', 'a5'])
        plt.title('action')
        plt.grid()

        i += 1
        plt.figure(i)
        plt.plot(np.array(env.logger.episode_dict["reward"]))
        plt.title('reward ' + str(sum(np.array(env.logger.episode_dict["reward"]))))
        plt.grid()

        plt.show()
