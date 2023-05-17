#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""用于评估任务二的最优表现

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
test_name = 'cabinet drawer open with plan'

rl_name = 'PPO'
time_name = '05-07-22-47'
# time_name = '05-08-09-09'
path_name = test_name + '/' + rl_name + '/' + time_name + '/'
logger_path = "eval_results/" + path_name + "best_model"
save_dir = './figs/' + test_name

_, _, rl_kwargs = env_kwargs(test_name, save_flag=False)
env = gym.make(env_name, **rl_kwargs)
result_dict = load_episode(logger_path)

save_flag = True
view_flag = True

duration = 2000

angle = np.pi / 60
r = np.sqrt(0.2175 ** 2 + 0.475 ** 2)
theta = np.arctan(0.475 / 0.2175) - angle
start_point = np.array([0.9 - r * np.cos(theta), -0.1135, 0.2 + r * np.sin(theta)]) + np.array([-0.011, 0, 0.004])
end_point = start_point + np.array([-0.3 * np.cos(angle), 0, - 0.3 * np.sin(angle)])
pos_table = np.linspace(start_point, end_point, num=duration - 1, axis=0)

table = np.array([[0.9986295, 0, -0.0523360],
                  [0, 1, 0],
                  [0.0523360, 0, 0.9986295]])
quat_table = mat33_to_quat(table)

i = 0

# 位置误差：实际运行result_dict["xpos"]、修正前env.init_desired_xposture_list[:, :3]、修正后result_dict["desired_xpos"]
# 对比修正后的期望轨迹与门的一致性
pos_init = env.init_desired_xposture_list[:duration - 1, 0:3]
pos_adjusted = result_dict["desired_xpos"]
pos_real = result_dict["xpos"]

plt.figure(i)
plt.plot(pos_table[:, 0], pos_table[:, 2], label='cabinet drawer')
plt.plot(pos_init[:, 0], pos_init[:, 2], label='desired path')
plt.plot(pos_adjusted[:, 0], pos_adjusted[:, 2], label='path adjusted by action')
plt.plot(pos_real[:, 0], pos_real[:, 2], label='actual path')
plt.legend(loc='lower right')
plt.xlabel(r'X position $\mathrm{(m)}$')
plt.ylabel(r'Z position $\mathrm{(m)}$')
plt.grid()
if save_flag:
    plt.savefig(save_dir + '/drawer xpos.png', dpi=600, bbox_inches='tight')
if view_flag:
    plt.title('drawer xpos')

xpos_error_table_buffer1 = []
xpos_error_table_buffer2 = []
for j in range(len(result_dict["xquat"])):
    xpos_error_table_buffer1.append(table.transpose() @ (pos_real - pos_init)[j])
    xpos_error_table_buffer2.append(table.transpose() @ (pos_real - pos_adjusted)[j])
xpos_error_table_buffer1 = np.array(xpos_error_table_buffer1)
xpos_error_table_buffer2 = np.array(xpos_error_table_buffer2)

i += 1
plt.figure(i)
plt.plot(xpos_error_table_buffer1[:, 0], label='x')
plt.plot(xpos_error_table_buffer1[:, 1], label='y')
plt.plot(xpos_error_table_buffer1[:, 2], label='z')
plt.legend(loc='upper right')
plt.xlabel('steps')
plt.ylabel(r'position error $\mathrm{(m)}$')
plt.xlim([0, 2000])
plt.grid()
if save_flag:
    plt.savefig(save_dir + '/drawer xpos error.png', dpi=600, bbox_inches='tight')
if view_flag:
    plt.title('drawer xpos error with init')

i += 1
plt.figure(i)
plt.plot(xpos_error_table_buffer2[:, 0], label='x')
plt.plot(xpos_error_table_buffer2[:, 1], label='y')
plt.plot(xpos_error_table_buffer2[:, 2], label='z')
plt.legend(loc='upper right')
plt.xlabel('steps')
plt.ylabel(r'position error $\mathrm{(m)}$')
plt.xlim([0, 2000])
plt.grid()
if save_flag:
    plt.savefig(save_dir + '/drawer xpos error.png', dpi=600, bbox_inches='tight')
if view_flag:
    plt.title('drawer xpos error with adjusted')

# 姿态误差
mat_init = env.init_desired_xposture_list[:duration - 1, 3:12]
quat_init = env.init_desired_xposture_list[:duration - 1, 12:16]
quat_adjusted = result_dict["desired_xquat"]
quat_real = result_dict["xquat"]
quat_optimal = quaternion_multiply(quat_table, quat_init[0])

orientation_error_buffer1 = []
orientation_error_buffer2 = []
orientation_error_buffer3 = []
for j in range(len(result_dict["xquat"])):
    orientation_error_buffer1.append(
        orientation_error_quat_with_quat(quat_init[j], quat_real[j]))
    orientation_error_buffer2.append(
        orientation_error_quat_with_quat(quat_adjusted[j], quat_real[j]))
    orientation_error_buffer3.append(
        orientation_error_quat_with_quat(quat_optimal, quat_real[j]))
orientation_error_buffer1 = np.array(orientation_error_buffer1)
orientation_error_buffer2 = np.array(orientation_error_buffer2)
orientation_error_buffer3 = np.array(orientation_error_buffer3)

i += 1
plt.figure(i)
plt.plot(orientation_error_buffer1[:, 0], label='x')
plt.plot(orientation_error_buffer1[:, 1], label='y')
plt.plot(orientation_error_buffer1[:, 2], label='z')
plt.legend(loc='upper right')
plt.xlabel('steps')
plt.ylabel(r'orientation error')
plt.xlim([0, 2000])
plt.grid()
if save_flag:
    plt.savefig(save_dir + '/drawer xquat error.png', dpi=600, bbox_inches='tight')
if view_flag:
    plt.title('orientation error with init')

i += 1
plt.figure(i)
plt.plot(orientation_error_buffer2[:, 0], label='x')
plt.plot(orientation_error_buffer2[:, 1], label='y')
plt.plot(orientation_error_buffer2[:, 2], label='z')
plt.legend(loc='upper right')
plt.xlabel('steps')
plt.ylabel(r'orientation error')
plt.xlim([0, 2000])
plt.grid()
if save_flag:
    plt.savefig(save_dir + '/drawer xquat error.png', dpi=600, bbox_inches='tight')
if view_flag:
    plt.title('orientation error with adjusted')

i += 1
plt.figure(i)
plt.plot(orientation_error_buffer3[:, 0], label='x')
plt.plot(orientation_error_buffer3[:, 1], label='y')
plt.plot(orientation_error_buffer3[:, 2], label='z')
plt.legend(loc='upper right')
plt.xlabel('steps')
plt.ylabel(r'orientation error')
plt.xlim([0, 2000])
plt.grid()
if save_flag:
    plt.savefig(save_dir + '/drawer xquat error.png', dpi=600, bbox_inches='tight')
if view_flag:
    plt.title('orientation error with optimal')

# 接触力
contact_force_table = np.empty_like(result_dict["contact_force"])
for k in range(contact_force_table.shape[0]):
    contact_force_table[k] = (table.transpose() @ result_dict["contact_force"][k].reshape((3, 2), order="F")). \
        reshape(-1, order="F")
i += 1
plt.figure(i)
plt.plot(contact_force_table[:, 0], label='x')
plt.plot(contact_force_table[:, 1], label='y')
plt.plot(contact_force_table[:, 2], label='z')
plt.legend(loc='upper right')
plt.xlabel('steps')
plt.ylabel(r'contact force $\mathrm{(N)}$')
plt.xlim([0, 2000])
plt.grid()
if save_flag:
    plt.savefig(save_dir + '/drawer contact force table.png', dpi=600, bbox_inches='tight')
if view_flag:
    plt.title('drawer contact force table')

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
    plt.savefig(save_dir + '/drawer K.png', dpi=600, bbox_inches='tight')
if view_flag:
    plt.title('drawer K')

# 刚度姿态
K_quat = []
for j in range(len(result_dict["xquat"])):
    K_quat.append(mat33_to_quat(result_dict["direction"][k].reshape((3, 3))))
K_quat = np.array(K_quat)
i += 1
plt.figure(i)
plt.plot([0, 1999], [quat_table[0], quat_table[0]], label='dx')
plt.plot([0, 1999], [quat_table[1], quat_table[1]], label='dy')
plt.plot([0, 1999], [quat_table[2], quat_table[2]], label='dz')
plt.plot([0, 1999], [quat_table[3], quat_table[3]], label='dw')
plt.plot(K_quat[:, 0], label='x')
plt.plot(K_quat[:, 1], label='y')
plt.plot(K_quat[:, 2], label='z')
plt.plot(K_quat[:, 3], label='w')
plt.legend(loc='upper right')
plt.xlabel('steps')
plt.ylabel(r'stiffness quat')
plt.xlim([0, 2000])
plt.grid()
if save_flag:
    plt.savefig(save_dir + '/drawer K quat.png', dpi=600, bbox_inches='tight')
if view_flag:
    plt.title('drawer K quat')

# 奖励
i += 1
plt.figure(i)
plt.plot(result_dict["reward"])
plt.xlabel('steps')
plt.ylabel('average return')
plt.xlim([0, 80])
plt.grid()
if save_flag:
    plt.savefig(save_dir + '/drawer reward.png', dpi=600, bbox_inches='tight')
if view_flag:
    plt.title('drawer reward')

if view_flag:
    plt.show()
