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
from envs.jk5_env_v6 import TrainEnvVariableStiffnessAndPosture as TrainEnv
from envs.env_kwargs import env_kwargs
from envs.controller import orientation_error_quat_with_mat
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.env_checker import check_env

task = 'cabinet surface with plan test'
algo = 'TrainEnvVariableStiffnessAndPostureAndSM-v7'
logger_path = './rl_test_results/' + task + '/' + algo
_, _, rl_kwargs = env_kwargs(task)
# env = TrainEnv(**rl_kwargs)
env = gym.make(algo, **rl_kwargs)
# env = gym.make(algo, **dict(task_name=task))

if not check_env(env):
    print('check passed')

test_times = 1
view_flag = False
plot_flag = True
save_fig = False
zero_action_flag = True

for _ in range(test_times):
    env.reset()
    if view_flag:
        env.viewer_init()
    if plot_flag:
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

    if plot_flag:
        i = 0

        # for _, (name, series) in enumerate(env.logger.episode_dict.items()):
        #     series = np.array(series)
        #     num = series.shape[-1]
        #     plt.figure(i)
        #     if name == 'observation':
        #         plt.plot(series[:, -1, :])
        #     else:
        #         plt.plot(series)
        #     plt.grid()
        #     plt.legend(np.linspace(1, num, num, dtype=int).astype(str).tolist())
        #     plt.title(name)
        #     i += 1
        #     if save_fig:
        #         plt.savefig(logger_path + '/' + name)

        plt.figure(i)
        plt.plot(np.array(env.logger.episode_dict["xpos"]))
        plt.plot(np.array(env.logger.episode_dict["desired_xpos"]))
        plt.plot(np.array(env.init_desired_xposture_list)[:, :3])
        plt.legend(['x', 'y', 'z', 'dx', 'dy', 'dz', 'idx', 'idy', 'idz'])
        plt.title('compare_xpos')
        plt.grid()
        if save_fig:
            plt.savefig(logger_path + '/' + plt.gca().get_title())

        i += 1
        plt.figure(i)
        plt.plot(np.array(env.logger.episode_dict["xpos"]) -
                 np.array(env.logger.episode_dict["desired_xpos"]))
        plt.legend(['x', 'y', 'z'])
        plt.title('compare_xpos_error')
        plt.grid()
        if save_fig:
            plt.savefig(logger_path + '/' + plt.gca().get_title())

        i += 1
        xpos_error = np.array(env.logger.episode_dict["xpos"]) - np.array(env.logger.episode_dict["desired_xpos"])
        table = np.array([[0.9986295, 0, -0.0523360],
                          [0, 1, 0],
                          [0.0523360, 0, 0.9986295]])
        xpos_error_table = np.array([table.transpose() @ xpos_error[i] for i in range(len(xpos_error))])
        plt.figure(i)
        plt.plot(xpos_error_table)
        plt.legend(['x', 'y', 'z'])
        plt.title('compare_xpos_error_table')
        plt.grid()
        if save_fig:
            plt.savefig(logger_path + '/' + plt.gca().get_title())

        i += 1
        plt.figure(i)
        plt.plot(np.array(env.logger.episode_dict["xvel"])[:, :3])
        plt.plot(np.array(env.logger.episode_dict["desired_xvel"])[:, :3])
        plt.legend(['x', 'y', 'z', 'dx', 'dy', 'dz'])
        plt.title('compare_xpos_vel')
        plt.grid()
        if save_fig:
            plt.savefig(logger_path + '/' + plt.gca().get_title())

        i += 1
        plt.figure(i)
        plt.plot(np.array(env.logger.episode_dict["desired_xacc"])[:, :3])
        plt.legend(['dx', 'dy', 'dz'])
        plt.title('compare_xpos_acc')
        plt.grid()
        if save_fig:
            plt.savefig(logger_path + '/' + plt.gca().get_title())

        i += 1
        plt.figure(i)
        plt.plot(np.array(env.logger.episode_dict["xquat"]))
        plt.plot(np.array(env.logger.episode_dict["desired_xquat"]))
        plt.plot(np.array(env.init_desired_xposture_list)[:, 12:16])
        plt.legend(['x', 'y', 'z', 'w', 'dx', 'dy', 'dz', 'idw', 'idx', 'idy', 'idz', 'idw'])
        plt.title('compare_xquat')
        plt.grid()
        if save_fig:
            plt.savefig(logger_path + '/' + plt.gca().get_title())

        i += 1
        plt.figure(i)
        plt.plot(np.array(env.logger.episode_dict["xvel"])[:, 3:])
        plt.plot(np.array(env.logger.episode_dict["desired_xvel"])[:, 3:])
        plt.legend(['x', 'y', 'z', 'dx', 'dy', 'dz'])
        plt.title('compare_xmat_vel')
        plt.grid()
        if save_fig:
            plt.savefig(logger_path + '/' + plt.gca().get_title())

        i += 1
        plt.figure(i)
        orientation_error_buffer = []
        for j in range(len(env.logger.episode_dict["xquat"])):
            orientation_error_buffer.append(
                orientation_error_quat_with_mat(np.array(env.logger.episode_dict["desired_xmat"])[j].reshape([3, 3]),
                                                np.array(env.logger.episode_dict["xmat"])[j].reshape([3, 3])))
        plt.plot(orientation_error_buffer)
        plt.legend(['x', 'y', 'z'])
        plt.title('compare_orientation_error')
        plt.grid()
        if save_fig:
            plt.savefig(logger_path + '/' + plt.gca().get_title())

        i += 1
        plt.figure(i)
        plt.plot(np.array(env.logger.episode_dict["contact_force"]))
        plt.legend(['x', 'y', 'z', 'rx', 'ry', 'rz'])
        plt.title('contact force')
        plt.grid()
        if save_fig:
            plt.savefig(logger_path + '/' + plt.gca().get_title())

        i += 1
        plt.figure(i)
        plt.plot(np.array(env.logger.episode_dict["contact_force_l"]))
        plt.legend(['x', 'y', 'z', 'rx', 'ry', 'rz'])
        plt.title('contact force local')
        plt.grid()
        if save_fig:
            plt.savefig(logger_path + '/' + plt.gca().get_title())

        i += 1
        plt.figure(i)
        contact_force = np.array(env.logger.episode_dict["contact_force"])
        table = np.array([[0.9986295, 0, -0.0523360],
                          [0, 1, 0],
                          [0.0523360, 0, 0.9986295]])
        contact_force_table = np.array(
            [(table.transpose() @ contact_force[i].reshape((3, 2), order="F")).reshape(-1, order="F")
             for i in range(len(contact_force))])
        plt.plot(contact_force_table)
        plt.legend(['x', 'y', 'z', 'rx', 'ry', 'rz'])
        plt.title('contact force table')
        plt.grid()
        if save_fig:
            plt.savefig(logger_path + '/' + plt.gca().get_title())

        delta_pos = np.array(env.logger.episode_dict["xpos"]) - np.array(env.logger.episode_dict["desired_xpos"])
        f = np.array(env.logger.episode_dict["contact_force"])[:, :3]
        stiffness = np.zeros_like(delta_pos)
        for j in range(len(stiffness)):
            if not np.any(delta_pos[j, :] == 0):
                stiffness[j] = f[j, :] / delta_pos[j, :]
        i += 1
        plt.figure(i)
        plt.plot(stiffness[3:, :])
        plt.legend(['x', 'y', 'z'])
        plt.title('stiffness')
        plt.grid()
        if save_fig:
            plt.savefig(logger_path + '/' + plt.gca().get_title())

        plt.show()
