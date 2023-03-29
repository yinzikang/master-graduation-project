#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Write target here

Write detailed description here

Write typical usage example here

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
3/28/23 9:07 PM   yinzikang      1.0         None
"""
# from module.jk5_env_v5 import TrainEnv
from module.jk5_env_v6 import TrainEnvVariableStiffness as TrainEnv
from module.env_kwargs import load_env_kwargs
from module.controller import orientation_error_quat_with_mat
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.env_checker import check_env

_, _, rl_kwargs = load_env_kwargs('cabinet surface with plan')
env = TrainEnv(**rl_kwargs)
if not check_env(env):
    print('check passed')

test_times = 1
plot_flag = True
zero_action_flag = False

for _ in range(test_times):
    env.reset()
    env.viewer_init()
    if plot_flag:
        env.logger_init('./env_test_log')
    R = 0
    while True:
        # Random action
        a = env.action_space.sample()
        if zero_action_flag:
            a = np.zeros_like(a)
        o, r, d, info = env.step(a)
        R += r
        if d:
            break

    print(info['terminal info'], ' ', R)

    # env.logger.dump_buffer(itr=0)
    if plot_flag:
        i = 0

        i += 1
        plt.figure(i)
        plt.plot(np.array(env.logger.episode_dict["xpos"]))
        plt.plot(np.array(env.logger.episode_dict["desired_xpos"]))
        plt.legend(['x', 'y', 'z', 'dx', 'dy', 'dz'])
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
        plt.plot(np.array(env.logger.episode_dict["xquat"]))
        plt.plot(np.array(env.logger.episode_dict["desired_xquat"]))
        plt.legend(['x', 'y', 'z', 'w', 'dx', 'dy', 'dz', 'dw'])
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
                orientation_error_quat_with_mat(np.array(env.logger.episode_dict["desired_xmat"])[j],
                                                np.array(env.logger.episode_dict["xmat"])[j]))
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
        plt.plot(np.array(env.logger.episode_dict["action"])[:,7])
        # plt.legend(['a0', 'a1', 'a2', 'a3', 'a4', 'a5'])
        plt.title('action')
        plt.grid()

        i += 1
        plt.figure(i)
        plt.plot(np.array(env.logger.episode_dict["reward"]))
        plt.title('reward ' + str(sum(np.array(env.logger.episode_dict["reward"]))))
        plt.grid()

        plt.show()
