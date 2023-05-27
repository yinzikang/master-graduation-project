#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""用于评估任务三的最优表现

Write detailed description here

Write typical usage example here

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
5/15/23 12:22 PM   yinzikang      1.0         None
"""
import gym
import matplotlib.pyplot as plt
import numpy as np

from gym_custom.envs.env_kwargs import env_kwargs
from gym_custom.utils.custom_loader import load_episode
from gym_custom.envs.controller import orientation_error_quat_with_quat, mat33_to_quat
from gym_custom.envs.transformations import quaternion_multiply, quaternion_inverse

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10.5
plt.rcParams['lines.linewidth'] = 2.0

env_name = 'TrainEnvVariableStiffnessAndPostureAndSM_v2-v8'
test_name = 'cabinet door open with plan'

rl_name = 'PPO'
# # time_name = '05-10-10-01'
# time_name = '05-09-15-56'
# path_name = test_name + '/' + rl_name + '/' + time_name + '/'
# # logger_path = "eval_results/" + path_name + "best_model"
# logger_path = "eval_results/" + path_name + "model"
# save_dir = './figs/' + test_name
# 优化前
path_name = './gym_custom/rl_test_results/' + test_name + '/' + env_name + '/'
logger_path = path_name
save_dir = logger_path

_, _, rl_kwargs = env_kwargs(test_name, save_flag=False)
env = gym.make(env_name, **rl_kwargs)
result_dict = load_episode(logger_path)

save_flag = True
view_flag = True

duration = 2000

cabinet_pos = np.array([0.8, -0.2, 0.3])
r_bias = np.array([-0.025, 0.34, 0])
angle_init = np.arctan(np.abs(r_bias[0] / r_bias[1]))
radius = np.linalg.norm(r_bias)
center = cabinet_pos + np.array([-0.2 + 0.0075, -0.19, 0.22])
rbt_tool = np.array([-0.011, -0.004, 0])

perfect_door_angle = np.linspace(0, np.pi / 2, duration - 1) + angle_init
pos_door = np.concatenate((-radius * np.sin(perfect_door_angle).reshape(-1, 1),
                           radius * np.cos(perfect_door_angle).reshape(-1, 1),
                           np.zeros_like(perfect_door_angle).reshape(-1, 1)), axis=1) + center.reshape(1, 3)

i = 0

# 位置误差：实际运行result_dict["xpos"]、修正前env.init_desired_xposture_list[:, :3]、修正后result_dict["desired_xpos"]
# 对比修正后的期望轨迹与门的一致性
pos_init = env.init_desired_xposture_list[:duration - 1, 0:3]
pos_adjusted = result_dict["desired_xpos"]
pos_real = result_dict["xpos"]

plt.figure(i)
# 门轨迹
plt.plot(pos_door[:, 0], pos_door[:, 1], label='cabinet door')
plt.plot([pos_door[0, 0], center[0]], [pos_door[0, 1], center[1]])
plt.plot([pos_door[-1, 0], center[0]], [pos_door[-1, 1], center[1]])
# 期望轨迹
plt.plot(pos_init[:, 0] - rbt_tool[0], pos_init[:, 1] - rbt_tool[1], label='desired path')
plt.plot([pos_init[0, 0] - rbt_tool[0], center[0]], [pos_init[0, 1] - rbt_tool[1], center[1]])
plt.plot([pos_init[-1, 0] - rbt_tool[0], center[0]], [pos_init[-1, 1] - rbt_tool[1], center[1]])
# # 修正后轨迹
# plt.plot(pos_adjusted[:, 0] - rbt_tool[0], pos_adjusted[:, 1] - rbt_tool[1], label='path adjusted by action')
# plt.plot([pos_adjusted[0, 0] - rbt_tool[0], center[0]], [pos_adjusted[0, 1] - rbt_tool[1], center[1]])
# plt.plot([pos_adjusted[-1, 0] - rbt_tool[0], center[0]], [pos_adjusted[-1, 1] - rbt_tool[1], center[1]])
# 实际轨迹
plt.plot(pos_real[:, 0] - rbt_tool[0], pos_real[:, 1] - rbt_tool[1], label='actual path')
plt.plot([pos_real[0, 0] - rbt_tool[0], center[0]], [pos_real[0, 1] - rbt_tool[1], center[1]])
plt.plot([pos_real[-1, 0] - rbt_tool[0], center[0]], [pos_real[-1, 1] - rbt_tool[1], center[1]])

plt.legend(loc='upper left')
plt.xlabel(r'X position $\mathrm{(m)}$')
plt.ylabel(r'Y position $\mathrm{(m)}$')
plt.grid()
plt.axis('equal')
if save_flag:
    plt.savefig(save_dir + '/door xpos.png', dpi=600, bbox_inches='tight')
if view_flag:
    plt.title('door xpos')

init_radius = []
init_angle = []
adjusted_radius = []
adjusted_angle = []
real_radius = []
real_angle = []
for j in range(len(pos_real)):
    init_r_bias = pos_init[j] - rbt_tool - center
    init_radius.append(np.linalg.norm(init_r_bias))
    init_angle.append(np.arctan2(-init_r_bias[0], init_r_bias[1]))

    adjusted_r_bias = pos_adjusted[j] - rbt_tool - center
    adjusted_radius.append(np.linalg.norm(adjusted_r_bias))
    adjusted_angle.append(np.arctan2(-adjusted_r_bias[0], adjusted_r_bias[1]))

    real_r_bias = pos_real[j] - rbt_tool - center
    real_radius.append(np.linalg.norm(real_r_bias))
    real_angle.append(np.arctan2(-real_r_bias[0], real_r_bias[1]))
init_radius = np.array(init_radius)
init_angle = np.array(init_angle)
adjusted_radius = np.array(adjusted_radius)
adjusted_angle = np.array(adjusted_angle)
real_radius = np.array(real_radius)
real_angle = np.array(real_angle)

i += 1
plt.figure(i)
plt.plot(init_radius, label='desired radius')
# plt.plot(adjusted_radius, label='radius adjusted by action')
plt.plot(real_radius, label='actual radius')
plt.legend(loc='lower left')
plt.xlabel('steps')
plt.ylabel(r'radius $\mathrm{(m)}$')
plt.xlim([0, 2000])
plt.grid()
if save_flag:
    plt.savefig(save_dir + '/door radius.png', dpi=600, bbox_inches='tight')
if view_flag:
    plt.title('door radius')

i += 1
plt.figure(i)
plt.plot(init_radius - radius, label='desired radius')
plt.plot(adjusted_radius - radius, label='radius adjusted by action')
plt.plot(real_radius - radius, label='actual radius')
plt.legend(loc='lower left')
plt.xlabel('steps')
plt.ylabel(r'radius error $\mathrm{(m)}$')
plt.xlim([0, 2000])
plt.grid()
if save_flag:
    plt.savefig(save_dir + '/door radius error.png', dpi=600, bbox_inches='tight')
if view_flag:
    plt.title('door radius error')


# 姿态误差
# mat_init = env.init_desired_xposture_list[:duration - 1, 3:12]
# quat_init = env.init_desired_xposture_list[:duration - 1, 12:16]
# quat_adjusted = result_dict["desired_xquat"]
# quat_real = result_dict["xquat"]
# quat_optimal = quaternion_multiply(quat_table, quat_init[0])
#
# orientation_error_buffer1 = []
# orientation_error_buffer2 = []
# orientation_error_buffer3 = []
# for j in range(len(result_dict["xquat"])):
#     orientation_error_buffer1.append(
#         orientation_error_quat_with_quat(quat_init[j], quat_real[j]))
#     orientation_error_buffer2.append(
#         orientation_error_quat_with_quat(quat_adjusted[j], quat_real[j]))
#     orientation_error_buffer3.append(
#         orientation_error_quat_with_quat(quat_optimal, quat_real[j]))
# orientation_error_buffer1 = np.array(orientation_error_buffer1)
# orientation_error_buffer2 = np.array(orientation_error_buffer2)
# orientation_error_buffer3 = np.array(orientation_error_buffer3)
#
# i += 1
# plt.figure(i)
# plt.plot(orientation_error_buffer1[:, 0], label='x')
# plt.plot(orientation_error_buffer1[:, 1], label='y')
# plt.plot(orientation_error_buffer1[:, 2], label='z')
# plt.legend(loc='upper right')
# plt.xlabel('steps')
# plt.ylabel(r'orientation error')
# plt.xlim([0, 2000])
# plt.grid()
# if save_flag:
#     plt.savefig(save_dir + '/door xquat error.png', dpi=600, bbox_inches='tight')
# if view_flag:
#     plt.title('orientation error with init')
#
# i += 1
# plt.figure(i)
# plt.plot(orientation_error_buffer2[:, 0], label='x')
# plt.plot(orientation_error_buffer2[:, 1], label='y')
# plt.plot(orientation_error_buffer2[:, 2], label='z')
# plt.legend(loc='upper right')
# plt.xlabel('steps')
# plt.ylabel(r'orientation error')
# plt.xlim([0, 2000])
# plt.grid()
# if save_flag:
#     plt.savefig(save_dir + '/door xquat error.png', dpi=600, bbox_inches='tight')
# if view_flag:
#     plt.title('orientation error with adjusted')
#
# i += 1
# plt.figure(i)
# plt.plot(orientation_error_buffer3[:, 0], label='x')
# plt.plot(orientation_error_buffer3[:, 1], label='y')
# plt.plot(orientation_error_buffer3[:, 2], label='z')
# plt.legend(loc='upper right')
# plt.xlabel('steps')
# plt.ylabel(r'orientation error')
# plt.xlim([0, 2000])
# plt.grid()
# if save_flag:
#     plt.savefig(save_dir + '/door xquat error.png', dpi=600, bbox_inches='tight')
# if view_flag:
#     plt.title('orientation error with optimal')

# 接触力
contact_force_door = []
for j in range(len(result_dict["xpos"])):
    robot_r_bias = result_dict["xpos"][j] - rbt_tool - center
    robot_angle = np.arctan2(-robot_r_bias[0], robot_r_bias[1])
    c = np.cos(robot_angle - np.pi / 2)
    s = np.sin(robot_angle - np.pi / 2)
    real_rotation = np.array([[c, -s, 0],
                              [s, c, 0],
                              [0, 0, 1]])
    contact_force_door.append((real_rotation.transpose() @
                                   result_dict['contact_force'][j].reshape((3, 2),
                                                                           order="F")).reshape(-1, order="F"))
contact_force_door = np.array(contact_force_door)

i += 1
plt.figure(i)
plt.plot(contact_force_door[:, 0], label='x')
plt.plot(contact_force_door[:, 1], label='y')
plt.plot(contact_force_door[:, 2], label='z')
plt.legend(loc='upper right')
plt.xlabel('steps')
plt.ylabel(r'contact force $\mathrm{(N)}$')
plt.xlim([0, 2000])
plt.grid()
if save_flag:
    plt.savefig(save_dir + '/door contact force door.png', dpi=600, bbox_inches='tight')
if view_flag:
    plt.title('door contact force door')

# 刚度大小
i += 1
plt.figure(i)
plt.plot(result_dict["K"][:, 0], label='x')
plt.plot(result_dict["K"][:, 1], label='y')
plt.plot(result_dict["K"][:, 2], label='z')
plt.legend(loc='upper right')
plt.xlabel('steps')
plt.ylabel(r'stiffness $\mathrm{(N/m)}$')
plt.xlim([0, 2000])
plt.grid()
if save_flag:
    plt.savefig(save_dir + '/door K.png', dpi=600, bbox_inches='tight')
if view_flag:
    plt.title('door K')

# 刚度姿态
# K_quat = []
# for j in range(len(result_dict["xquat"])):
#     K_quat.append(mat33_to_quat(result_dict["direction"][k].reshape((3, 3))))
# K_quat = np.array(K_quat)
# i += 1
# plt.figure(i)
# plt.plot([0, 1999], [quat_table[0], quat_table[0]], label='dx')
# plt.plot([0, 1999], [quat_table[1], quat_table[1]], label='dy')
# plt.plot([0, 1999], [quat_table[2], quat_table[2]], label='dz')
# plt.plot([0, 1999], [quat_table[3], quat_table[3]], label='dw')
# plt.plot(K_quat[:, 0], label='x')
# plt.plot(K_quat[:, 1], label='y')
# plt.plot(K_quat[:, 2], label='z')
# plt.plot(K_quat[:, 3], label='w')
# plt.legend(loc='upper right')
# plt.xlabel('steps')
# plt.ylabel(r'stiffness quat')
# plt.xlim([0, 2000])
# plt.grid()
# if save_flag:
#     plt.savefig(save_dir + '/door K quat.png', dpi=600, bbox_inches='tight')
# if view_flag:
#     plt.title('door K quat')

# 奖励
i += 1
plt.figure(i)
plt.plot(result_dict["reward"])
plt.xlabel('steps')
plt.ylabel('average return')
plt.xlim([0, 80])
plt.grid()
if save_flag:
    plt.savefig(save_dir + '/door reward.png', dpi=600, bbox_inches='tight')
if view_flag:
    plt.title('door reward')

if view_flag:
    plt.show()
