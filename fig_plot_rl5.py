#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""用于截刚度椭球的图最优表现

Write detailed description here

Write typical usage example here

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
5/16/23 12:54 PM   yinzikang      1.0         None
"""

import os

import gym
import matplotlib.pyplot as plt
import mujoco as mp
import numpy as np

from gym_custom.envs.env_kwargs import env_kwargs
from gym_custom.envs.transformations import quaternion_about_axis, rotation_matrix, quaternion_multiply


def draw_ellipsoid(k_r_p):
    env.viewer.add_marker(pos=k_r_p[2],  # Position
                          mat=k_r_p[1],
                          label=" ",  # Text beside the marker
                          type=mp.mjtGeom.mjGEOM_ELLIPSOID,  # Geomety type
                          size=k_r_p[0],  # Size of the marker
                          rgba=(84 / 255, 179 / 255, 69 / 255, 0.75),
                          emission=1)  # RGBA of the marker


def load_npy_files(path):
    results = {}
    for file in os.listdir(path):
        if file.endswith('.npy'):
            name = os.path.splitext(file)[0]
            file_path = os.path.join(path, file)
            data = np.load(file_path)
            results[name] = data
    return results


plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10.5
plt.rcParams['lines.linewidth'] = 2.0

# 任务参数
# 环境加载
env_name = 'TrainEnvVariableStiffnessAndPostureAndSM_v2-v8'
# test_name = 'cabinet surface with plan v7'
# time_name = '04-30-17-20'
# test_name = 'cabinet drawer open with plan'
# time_name = '05-07-22-47'
test_name = 'cabinet door open with plan'
time_name = '05-09-15-56'
rl_name = 'PPO'
path_name = test_name + '/' + rl_name + '/' + time_name + '/'
itr = 655360
mode = 3

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
action_series = result_dict['action']

_, _, rl_kwargs = env_kwargs(test_name, save_flag=False)
env = gym.make(env_name, **rl_kwargs)
env.reset()
if render_flag:
    env.viewer_init(pause_start=True)

R = 0
# record_step_list = [0, 30, 50, 70]
record_step_list = [0, 20, 40, 60]
k_r_p_list = []
for step in range(action_series.shape[0]):
    action = action_series[step] / env.action_limit

    # o, r, done, info = env.step(action)

    # 其余状态的初始化
    reward = 0
    done = False
    other_info = dict()
    # 对多次执行机器人控制
    sub_step = 0
    action = env.action_limit * action
    delta_K = action[:6] / env.sub_step_num
    ddelta_pos, ddelta_direction, ddelta_angle = action[6:9] / env.sub_step_num, action[9:12] / np.linalg.norm(
        action[9:12]), action[12] / env.sub_step_num
    ddelta_mat = rotation_matrix(ddelta_angle, ddelta_direction)[:3, :3]
    ddelta_quat = quaternion_about_axis(ddelta_angle, ddelta_direction)
    ellipsoid_direction = action[13:16] / np.linalg.norm(action[13:16])
    ellipsoid_delta_angle = action[16] / env.sub_step_num
    ellipsoid_delta_mat = rotation_matrix(ellipsoid_delta_angle, ellipsoid_direction)[:3, :3]

    if step in record_step_list:
        k_r_p_list.append([env.status["controller_parameter"]["K"][:3] / 50000,
                           env.status["controller_parameter"]["SM"],
                           env.status["xpos"]])

    for sub_step in range(env.sub_step_num):
        # 刚度椭圆轴变化量，进行插值（每一次更新间不同，有叠加效果）
        env.controller_parameter['K'] += delta_K
        M = env.controller_parameter['K'] / (env.wn * env.wn)
        env.controller_parameter['B'] = 2 * env.damping_ratio * np.sqrt(M * env.controller_parameter['K'])
        env.controller_parameter['M'] = M
        # 位姿变化量，进行叠加（每一次更新间不同，无叠加效果）
        env.delta_pos += ddelta_pos
        env.status['desired_xpos'] += env.delta_pos
        env.delta_mat = ddelta_mat @ env.delta_mat
        env.delta_quat = quaternion_multiply(ddelta_quat, env.delta_quat)
        env.status['desired_xmat'] = env.delta_mat @ env.status['desired_xmat']
        env.status['desired_xquat'] = quaternion_multiply(env.delta_quat, env.status['desired_xquat'])
        # 刚度椭球姿态变化量
        env.controller_parameter['SM'] = ellipsoid_delta_mat @ env.controller_parameter['SM']


        # 可视化
        if hasattr(env, 'viewer'):
            env.render()

        for idx in range(len(k_r_p_list)):
            draw_ellipsoid(k_r_p_list[idx])

        env.step2()
        success, error_K, error_force = False, False, False
        # 时间约束：到达最大时间，视为done
        if env.current_step + 1 == env.step_num:
            success = True
            other_info['is_success'], other_info["TimeLimit.truncated"], other_info[
                'terminal info'] = True, True, 'success'
        # 刚度约束：到达最大时间，视为done
        if any(np.greater(env.controller_parameter['K'], env.max_K)) or \
                any(np.greater(env.min_K, env.controller_parameter['K'])):
            error_K = True
            other_info['is_success'], other_info["TimeLimit.truncated"], other_info[
                'terminal info'] = False, False, 'error K'
            # print(env.controller_parameter['K'])
        # 接触力约束：超过范围，视为done
        if any(np.greater(np.abs(env.status['contact_force']), env.max_force)):
            error_force = True
            other_info['is_success'], other_info["TimeLimit.truncated"], other_info[
                'terminal info'] = False, False, 'error force'
            # print(env.status['contact_force'])
        failure = error_K or error_force
        done = success or failure
        env.status.update(done=done, success=success, failure=failure)

        # 获得奖励
        reward += env.get_reward(done, success, failure)

        if done:
            break

    reward = reward / (sub_step + 1)

    if done:
        input()