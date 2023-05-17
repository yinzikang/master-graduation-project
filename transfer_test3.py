#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""用于轨迹的迁移：
将轨迹复制出来，将奖励函数复制出来，用不了rl结构，只能用robot结构

Write detailed description here

Write typical usage example here

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
5/11/23 10:06 AM   yinzikang      1.0         None
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from gym_custom.envs.controller import ComputedTorqueController
from gym_custom.envs.controller import orientation_error_quat_with_quat
from gym_custom.envs.env_kwargs import env_kwargs
from gym_custom.envs.jk5_env_v8 import Jk5StickRobotWithController
from eval_everything import eval_robot

def load_npy_files(path):
    results = {}
    for file in os.listdir(path):
        if file.endswith('.npy'):
            name = os.path.splitext(file)[0]
            file_path = os.path.join(path, file)
            data = np.load(file_path)
            results[name] = data
    return results


def get_reward(done, success, failure):
    if 'cabinet surface with plan' in env.task:
        # 奖励范围： - 1.5
        # xposture_error = np.concatenate([env.status['desired_xpos'] - env.status['xpos'],
        #                                  orientation_error_quat_with_quat(env.status['desired_xquat'],
        #                                                                   env.status['xquat'])])
        xposture_error = np.concatenate([env.init_desired_xposture_list[env.current_step, 0:3] -
                                         env.status['xpos'],
                                         orientation_error_quat_with_quat(env.init_desired_xposture_list[
                                                                          env.current_step, 12:16],
                                                                          env.status['xquat'])])
        force_error = env.status['contact_force'] - env.status['desired_force']
        # table = np.eye(3)
        table_rotation = np.array([[0.9986295, 0, -0.0523360],
                                   [0, 1, 0],
                                   [0.0523360, 0, 0.9986295]])
        xposture_error_table = (table_rotation.transpose() @
                                xposture_error.reshape((3, 2), order="F")).reshape(-1, order="F")
        force_error_table = (table_rotation.transpose() @
                             force_error.reshape((3, 2), order="F")).reshape(-1, order="F")

        # 运动状态的奖励
        movement_reward = - np.sum(abs(xposture_error_table)[[0, 1, 3, 4, 5]])
        # 要是力距离期望力较近则进行额外奖励
        fext_reward = - np.sum(abs(force_error_table))
        fext_reward = fext_reward + 10 if fext_reward > -2.5 else fext_reward

        reward = 5 * movement_reward + 0.05 * fext_reward + 1.

    elif 'cabinet drawer open with plan' in env.task:
        # xposture_error = np.concatenate([env.status['desired_xpos'] - env.status['xpos'],
        #                                  orientation_error_quat_with_quat(env.status['desired_xquat'],
        #                                                                   env.status['xquat'])])
        xposture_error = np.concatenate([env.init_desired_xposture_list[env.current_step, 0:3] -
                                         env.status['xpos'],
                                         orientation_error_quat_with_quat(env.init_desired_xposture_list[
                                                                          env.current_step, 12:16],
                                                                          env.status['xquat'])])
        # table = np.eye(3)
        table_rotation = np.array([[0.9986295, 0, -0.0523360],
                                   [0, 1, 0],
                                   [0.0523360, 0, 0.9986295]])
        xposture_error_table = (table_rotation.transpose() @
                                xposture_error.reshape((3, 2), order="F")).reshape(-1, order="F")
        force_table = (table_rotation.transpose() @
                       env.status['contact_force'].reshape((3, 2), order="F")).reshape(-1, order="F")

        # 运动状态的奖励
        movement_reward = - np.sum(abs(xposture_error_table)[0])
        # 要是力距离期望力较近则进行额外奖励
        fext_reward = - np.sum(abs(force_table)[1:])
        fext_reward = fext_reward + 10 if fext_reward > -2.5 else fext_reward
        # 成功则衡量任务的完成度给出对应奖励，失败则给出恒定惩罚
        drawer_reward = 0
        if success:
            drawer_reward = env.data.qpos[-2] / 0.3
        if failure:
            drawer_reward = -1
        # print(env.data.qpos[-2] / 0.3)
        reward = 5 * movement_reward + 0.05 * fext_reward + 5 * drawer_reward + 1.

    elif 'cabinet door open with plan' in env.task:
        cabinet_pos = np.array([0.8, -0.2, 0.3])  # 准确
        radius = np.sqrt(0.34 ** 2 + 0.025 ** 2)  # 准确
        angle_bias = np.arctan(0.025 / 0.34)  # 准确
        center = cabinet_pos + np.array([-0.2 + 0.0075, -0.19, 0.22])  # 准确
        door_angle = env.data.qpos[-1]
        c = np.cos(angle_bias + door_angle - np.pi / 2)
        s = np.sin(angle_bias + door_angle - np.pi / 2)
        door_rotation = np.array([[c, -s, 0],
                                  [s, c, 0],
                                  [0, 0, 1]])
        radius_error = np.linalg.norm(env.status['xpos'] - center) - radius
        force_door = (door_rotation.transpose() @
                      env.status['contact_force'].reshape((3, 2), order="F")).reshape(-1, order="F")
        # 运动状态的奖励
        movement_reward = - np.abs(radius_error)
        # 要是力距离期望力较近则进行额外奖励
        fext_reward = - np.sum(abs(force_door)[[0, 2, 3, 4, 5]])
        fext_reward = fext_reward + 10 if fext_reward > -5 else fext_reward
        # 成功则衡量任务的完成度给出对应奖励，失败则给出恒定惩罚
        door_reward = 0
        if success:
            door_reward = door_angle / np.pi * 2
        if failure:
            door_reward = -1

        reward = 5 * movement_reward + 0.05 * fext_reward + 5 * door_reward + 1.

    return reward


plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10.5
plt.rcParams['lines.linewidth'] = 2.0

# 任务参数
# 环境加载
env_name = 'TrainEnvVariableStiffnessAndPostureAndSM_v2-v8'
# test_name = 'cabinet surface with plan v7'
# time_name = '04-30-17-20'
# mode = 3
test_name = 'cabinet drawer open with plan'
time_name = '05-07-22-47'
mode = 3
# test_name = 'cabinet door open with plan'
# time_name = '05-09-15-56'
# mode = 2
rl_name = 'PPO'
path_name = test_name + '/' + rl_name + '/' + time_name + '/'
itr = 655360

plot_flag = False
render_flag = True
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
desired_xposture_list = np.concatenate((result_dict['xpos'], result_dict['xmat'], result_dict['xquat']), axis=1)
desired_xvel_list = result_dict['xvel']
desired_xacc_list = result_dict['desired_xacc']
desired_force_list = result_dict['desired_force']

desired_xposture_list = np.vstack([desired_xposture_list, desired_xposture_list[-1:, :]])
desired_xvel_list = np.vstack([desired_xvel_list, desired_xvel_list[-1:, :]])
desired_xacc_list = np.vstack([desired_xacc_list, desired_xacc_list[-1:, :]])
desired_force_list = np.vstack([desired_force_list, desired_force_list[-1:, :]])

_, rbt_controller_kwargs, rl_kwargs = env_kwargs(test_name, save_flag=False)
wn = 200
damping_ratio = np.sqrt(2)
kp = wn * wn * np.ones(6, dtype=np.float64)
kd = 2 * damping_ratio * np.sqrt(kp)
controller_parameter = {'kp': kp, 'kd': kd}
controller = ComputedTorqueController
rbt_controller_kwargs.update(controller_parameter=controller_parameter, controller=controller,
                             desired_xposture_list=desired_xposture_list, desired_xvel_list=desired_xvel_list,
                             desired_xacc_list=desired_xacc_list, desired_force_list=desired_force_list)

# env = gym.make(env_name, **rl_kwargs)
env = Jk5StickRobotWithController(**rbt_controller_kwargs)
env.reset()
if render_flag:
    env.viewer_init()

reward = 0
buffer = dict()
for status_name in env.status.keys():
    buffer[status_name] = [env.status[status_name]]
for step in range(2000):

    done = False
    other_info = dict()
    # env.render()
    env.step()
    success, error_K, error_force = False, False, False
    # 时间约束：到达最大时间，视为done
    if env.current_step + 1 == env.step_num:
        success = True
        other_info['is_success'], other_info["TimeLimit.truncated"], other_info[
            'terminal info'] = True, True, 'success'

    failure = error_K or error_force
    done = success or failure

    # 获得奖励
    reward += get_reward(done, success, failure)

    if done:
        break

    for status_name in env.status.keys():
        buffer[status_name].append(env.status[status_name])
for status_name in env.status.keys():
    buffer[status_name] = np.array(buffer[status_name])

print(reward / 25)

eval_robot(test_name, buffer, plot_flag)