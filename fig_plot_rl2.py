#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""用于评估任务一的最优表现

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

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10.5
plt.rcParams['lines.linewidth'] = 2.0

env_name = 'TrainEnvVariableStiffnessAndPostureAndSM_v2-v8'
test_name = 'cabinet surface with plan v7'

rl_name = 'PPO'
time_name = '04-30-17-20'
# time_name = '05-15-11-11'
path_name = test_name + '/' + rl_name + '/' + time_name + '/'
logger_path = "eval_results/" + path_name + "best_model"
save_dir = './figs/' + test_name + '/'

_, _, rl_kwargs = env_kwargs(test_name, save_flag=False)
env = gym.make(env_name, **rl_kwargs)
result_dict = load_episode(logger_path)

save_flag = False
view_flag = True

duration = 2000
table = np.array([[0.9986295, 0, -0.0523360],
                  [0, 1, 0],
                  [0.0523360, 0, 0.9986295]])

i = 0

# 位置误差：实际运行result_dict["xpos"]、修正前env.init_desired_xposture_list[:, :3]、修正后result_dict["desired_xpos"]
# 对比修正后的期望轨迹与门的一致性
angle = np.pi / 60
r = np.sqrt(0.2 ** 2 + 0.575 ** 2)
theta = np.arctan(0.575 / 0.2) - angle
start_point = np.array([0.5 - r * np.cos(theta), -0.1135, r * np.sin(theta)])
end_point = start_point + np.array([0.4 * np.cos(angle), 0, 0.4 * np.sin(angle)])
pos_table = np.linspace(start_point, end_point, num=duration - 1, axis=0)

pos_init = env.init_desired_xposture_list[:duration - 1, 0:3]
pos_adjusted = result_dict["desired_xpos"]
pos_real = result_dict["xpos"]

plt.figure(i)
# plt.plot(pos_table[:, 0]-0.01*np.sin(np.pi / 60), pos_table[:, 2]+0.01*np.cos(np.pi / 60), label='table_surface')
plt.plot(pos_table[:, 0], pos_table[:, 2], label='cabinet surface')
plt.plot(pos_init[:, 0], pos_init[:, 2], label='desired path')
plt.plot(pos_adjusted[:, 0], pos_adjusted[:, 2], label='path adjusted by action')
plt.plot(pos_real[:, 0], pos_real[:, 2], label='actual path')
# plt.plot(pos_table[:, 0], pos_table[:, 2], label='table_surface')
# plt.plot(pos_init[:, 0]+0.01*np.sin(np.pi / 60), pos_init[:, 2]-0.01*np.cos(np.pi / 60), label='desired path')
# plt.plot(pos_adjusted[:, 0]+0.01*np.sin(np.pi / 60), pos_adjusted[:, 2]-0.01*np.cos(np.pi / 60), label='adjusted path')
# plt.plot(pos_real[:, 0]+0.01*np.sin(np.pi / 60), pos_real[:, 2]-0.01*np.cos(np.pi / 60), label='actual path')
plt.legend(loc='upper right')
plt.xlabel(r'X position $\mathrm{(m)}$')
plt.ylabel(r'Z position $\mathrm{(m)}$')
plt.grid()
if save_flag:
    plt.savefig(save_dir + '/surface xpos.png', dpi=600, bbox_inches='tight')
if save_flag:
    plt.title('surface xpos')

xpos_error_table_buffer1 = []
xpos_error_table_buffer2 = []
for j in range(len(result_dict["xquat"])):
    xpos_error_table_buffer1.append(table.transpose() @ (pos_real - pos_init)[j])
    xpos_error_table_buffer2.append(table.transpose() @ (pos_real - pos_adjusted)[j])
xpos_error_table_buffer1 = np.array(xpos_error_table_buffer1)
xpos_error_table_buffer2 = np.array(xpos_error_table_buffer2)

# i += 1
# plt.figure(i)
# plt.plot(xpos_error_table_buffer1)
# plt.legend(['x', 'y', 'z', 'dx', 'dy', 'dz'])
# plt.grid()
# if view_flag:
#     plt.title('xpos error table init')
# if save_fig:
#     plt.savefig(fig_path + '/' + plt.gca().get_title())

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
    plt.savefig(save_dir + '/surface xpos error.png', dpi=600, bbox_inches='tight')
if save_flag:
    plt.title('surface xpos error')

# 姿态误差

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
    plt.savefig(save_dir + '/surface contact force table.png', dpi=600, bbox_inches='tight')
if save_flag:
    plt.title('surface contact force table')

# i += 1
# plt.figure(i)
# plt.plot(result_dict["contact_force"])
# plt.legend(['x', 'y', 'z', 'rx', 'ry', 'rz'])
# plt.title('contact force')
# plt.grid()
# if save_fig:
#     plt.savefig(fig_path + '/' + plt.gca().get_title())

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
    plt.savefig(save_dir + '/surface K.png', dpi=600, bbox_inches='tight')
if save_flag:
    plt.title('surface K')

# 刚度姿态

# 奖励
i += 1
plt.figure(i)
plt.plot(result_dict["reward"])
# plt.legend(loc='upper right')
plt.xlabel('steps')
plt.ylabel('average return')
plt.xlim([0, 80])
plt.grid()
if save_flag:
    plt.savefig(save_dir + '/surface reward.png', dpi=600, bbox_inches='tight')
if save_flag:
    plt.title('surface reward')

if view_flag:
    plt.show()
