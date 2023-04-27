#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""毕业论文中轨迹规划画图

Write detailed description here

Write typical usage example here

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
4/22/23 8:17 PM   yinzikang      1.0         None
"""
import os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from gym_custom.envs.controller import get_trajectory_para, joint_interpolation, trajectory_planning_line, \
    trajectory_planning_circle
from gym_custom.envs.jk5_env_v7 import Jk5StickRobot, env_kwargs

rbt_kwargs, _, _ = env_kwargs('fig_plot')
jk5 = Jk5StickRobot(**rbt_kwargs)
# np.array([0, -1, 0, 0, -1, 1, 0, 0, 0])
time_w = 4
time_a = 1
f = 500
xpos_init, mat_init = np.array([0.4, -0.1135, 0.4]), np.array([0, 0, 1, 1, 0, 0, 0, 1, 0])
# xpos_init, mat_init = np.array([0., -0.3765, 1.0635]), np.array([1, 0, 0, 0, 0, -1, 0, 1, 0])
xpos_mid, mat_mid = np.array([0.5, -0.1135, 0.5]), np.array([0, -1, 0, 0, -1, 1, 0, 0, 0])
# xpos_end, mat_end = np.array([0.4, 0.2, 0.5]), np.array([0, 0, 1, 0, 1, 0, -1, 0, 0])
# xpos_end, mat_end = np.array([0.6, -0., 0.4]), np.array([0, 1, 0, 1, 0, 0, 0, 0, -1])
xpos_end, mat_end = np.array([0.6, -0., 0.4]), np.array([0.0000000, 0.8660254, 0.5000000,
                                                         0.8660254, -0.2500000, 0.4330127,
                                                         0.5000000, 0.4330127, -0.7500000])
qpos_init = jk5.inverse_kinematics(np.array([0, 0, -90, 90, 90, 0]) / 180 * np.pi, xpos_init, mat_init)
qpos_end = jk5.inverse_kinematics(np.array([0, 0, 0, 0, 90, 0]) / 180 * np.pi, xpos_end, mat_end)
print(qpos_init / np.pi * 180)
print(qpos_end / np.pi * 180)
print(jk5.forward_kinematics(qpos_init))
print(jk5.forward_kinematics(qpos_end))

show_flag = True
save_flag = True
save_dir = './figs/trajectory_plan'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 所有曲线曲线
acc, vel, pos = get_trajectory_para(time_w, time_a, f * time_w)
xposture_l, xvel_l, xacc_l = trajectory_planning_line(xpos_init, mat_init, xpos_end, mat_end, time_w, time_a, f)
xposture_c, xvel_c, xacc_c = trajectory_planning_circle(xpos_init, mat_init, xpos_end, mat_end, xpos_mid, time_w,
                                                        time_a, f)
qpos_j, qvel_j, qacc_j = joint_interpolation(qpos_init, qpos_end, time_w, time_a, f)

qpos_l = np.empty_like(qpos_j)
qpos_l[0] = qpos_init.copy()
qpos_c = np.empty_like(qpos_j)
qpos_c[0] = qpos_init.copy()
xpos_j = np.empty((f * time_w, 3))
xpos_j[0] = xpos_init.copy()
for idx in range(1, len(qpos_l)):
    qpos_l[idx] = jk5.inverse_kinematics(qpos_l[idx - 1], xposture_l[idx, :3], xposture_l[idx, 3:12].reshape((3, 3)))
    qpos_c[idx] = jk5.inverse_kinematics(qpos_l[idx - 1], xposture_c[idx, :3], xposture_c[idx, 3:12].reshape((3, 3)))
    xpos_j[idx], _ = jk5.forward_kinematics(qpos_j[idx])

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10.5
plt.rcParams['lines.linewidth'] = 2.0

i = 1
plt.figure(i, figsize=(6, 4))
plt.plot(pos, label=r'$\alpha$')
plt.plot(vel, label=r'$\dot{\alpha}$')
plt.plot(acc, label=r'$\ddot{\alpha}$')
plt.legend(loc='upper right')
plt.xlabel('steps')
plt.ylabel('')
plt.xlim([0, 2000])
# plt.ylim([-0.6, 1.1])
plt.grid()
if save_flag:
    plt.savefig(save_dir + '/插值因子.png', dpi=600, bbox_inches='tight')
if show_flag:
    plt.title(r'$\alpha$')
i += 1
#
# # 直线位置速度加速度
# plt.figure(i, figsize=(6, 4))
# plt.plot(xposture_l[:, 0], label=r'$x$')
# plt.plot(xposture_l[:, 1], label=r'$y$')
# plt.plot(xposture_l[:, 2], label=r'$z$')
# plt.legend(loc='upper right')
# plt.xlabel('steps')
# plt.ylabel(r'position $\mathrm{(m)}$')
# plt.xlim([0, 2000])
# # plt.ylim([-0.1, 0.6])
# plt.grid()
# if save_flag:
#     plt.savefig(save_dir + '/xpos_l.png', dpi=600, bbox_inches='tight')
# if show_flag:
#     plt.title('xposture_l')
# i += 1
#
# plt.figure(i, figsize=(6, 4))
# plt.plot(xvel_l[:, 0], label=r'$x$')
# plt.plot(xvel_l[:, 1], label=r'$y$')
# plt.plot(xvel_l[:, 2], label=r'$z$')
# plt.legend(loc='upper right')
# plt.xlabel('steps')
# plt.ylabel(r'velocity $\mathrm{(m/s)}$')
# plt.xlim([0, 2000])
# # plt.ylim([-0.05, 0.1])
# plt.grid()
# if save_flag:
#     plt.savefig(save_dir + '/xvel_l.png', dpi=600, bbox_inches='tight')
# if show_flag:
#     plt.title('xvel_l')
# i += 1
#
# plt.figure(i, figsize=(6, 4))
# plt.plot(xacc_l[:, 0], label=r'$x$')
# plt.plot(xacc_l[:, 1], label=r'$y$')
# plt.plot(xacc_l[:, 2], label=r'$z$')
# plt.legend(loc='upper right')
# plt.xlabel('steps')
# plt.ylabel(r'acceleration $\mathrm{(m/s^2)}$')
# plt.xlim([0, 2000])
# # plt.ylim([-0.12, 0.12])
# plt.grid()
# if save_flag:
#     plt.savefig(save_dir + '/xacc_l.png', dpi=600, bbox_inches='tight')
# if show_flag:
#     plt.title('xacc_l')
# i += 1
#
# # 圆弧位置速度加速度
# plt.figure(i, figsize=(6, 4))
# plt.plot(xposture_c[:, 0], label=r'$x$')
# plt.plot(xposture_c[:, 1], label=r'$y$')
# plt.plot(xposture_c[:, 2], label=r'$z$')
# plt.legend(loc='upper right')
# plt.xlabel('steps')
# plt.ylabel(r'position $\mathrm{(m)}$')
# plt.xlim([0, 2000])
# # plt.ylim([-0.05, 0.55])
# plt.grid()
# if save_flag:
#     plt.savefig(save_dir + '/xpos_c.png', dpi=600, bbox_inches='tight')
# if show_flag:
#     plt.title('xposture_c')
# i += 1
#
# plt.figure(i, figsize=(6, 4))
# plt.plot(xvel_c[:, 0], label=r'$x$')
# plt.plot(xvel_c[:, 1], label=r'$y$')
# plt.plot(xvel_c[:, 2], label=r'$z$')
# plt.legend(loc='upper right')
# plt.xlabel('steps')
# plt.ylabel('velocity $\mathrm{(m/s)}$')
# plt.xlim([0, 2000])
# # plt.ylim([-0.175, 0.15])
# plt.grid()
# if save_flag:
#     plt.savefig(save_dir + '/xvel_c.png', dpi=600, bbox_inches='tight')
# if show_flag:
#     plt.title('xvel_c')
# i += 1
#
# plt.figure(i, figsize=(6, 4))
# plt.plot(xacc_c[:, 0], label=r'$x$')
# plt.plot(xacc_c[:, 1], label=r'$y$')
# plt.plot(xacc_c[:, 2], label=r'$z$')
# plt.legend(loc='upper right')
# plt.xlabel('steps')
# plt.ylabel(r'acceleration $\mathrm{(m/s^2)}$')
# plt.xlim([0, 2000])
# # plt.ylim([-0.25, 0.2])
# plt.grid()
# if save_flag:
#     plt.savefig(save_dir + '/xacc_c.png', dpi=600, bbox_inches='tight')
# if show_flag:
#     plt.title('xacc_c')
# i += 1
#
# # 姿态速度加速度
# plt.figure(i, figsize=(6, 4))
# plt.plot(xposture_c[:, 12], label=r'$x$')
# plt.plot(xposture_c[:, 13], label=r'$y$')
# plt.plot(xposture_c[:, 14], label=r'$z$')
# plt.plot(xposture_c[:, 15], label=r'$w$')
# plt.legend(loc='upper right')
# plt.xlabel('steps')
# plt.ylabel('')
# plt.xlim([0, 2000])
# # plt.ylim([-0.05, 0.55])
# plt.grid()
# if save_flag:
#     plt.savefig(save_dir + '/quat.png', dpi=600, bbox_inches='tight')
# if show_flag:
#     plt.title('quat')
# i += 1
#
# plt.figure(i, figsize=(6, 4))
# plt.plot(xvel_c[:, 3], label=r'$x$')
# plt.plot(xvel_c[:, 4], label=r'$y$')
# plt.plot(xvel_c[:, 5], label=r'$z$')
# plt.legend(loc='upper right')
# plt.xlabel('steps')
# plt.ylabel('velocity $\mathrm{(rad/s)}$')
# plt.xlim([0, 2000])
# # plt.ylim([-0.175, 0.15])
# plt.grid()
# if save_flag:
#     plt.savefig(save_dir + '/w.png', dpi=600, bbox_inches='tight')
# if show_flag:
#     plt.title('xvel_r')
# i += 1
#
# plt.figure(i, figsize=(6, 4))
# plt.plot(xacc_c[:, 3], label=r'$x$')
# plt.plot(xacc_c[:, 4], label=r'$y$')
# plt.plot(xacc_c[:, 5], label=r'$z$')
# plt.legend(loc='upper right')
# plt.xlabel('steps')
# plt.ylabel(r'acceleration $\mathrm{(rad/s^2)}$')
# plt.xlim([0, 2000])
# # plt.ylim([-0.25, 0.2])
# plt.grid()
# if save_flag:
#     plt.savefig(save_dir + '/wdot.png', dpi=600, bbox_inches='tight')
# if show_flag:
#     plt.title('xacc_r')
# i += 1
#
# # 关节速度加速度
# plt.figure(i, figsize=(6, 4))
# plt.plot(qpos_j[:, 0], label=r'$q1$')
# plt.plot(qpos_j[:, 1], label=r'$q2$')
# plt.plot(qpos_j[:, 2], label=r'$q3$')
# plt.plot(qpos_j[:, 3], label=r'$q4$')
# plt.plot(qpos_j[:, 4], label=r'$q5$')
# plt.plot(qpos_j[:, 5], label=r'$q6$')
# plt.legend(loc='upper right')
# plt.xlabel('steps')
# plt.ylabel(r'pos $\mathrm{(rad)}$')
# plt.xlim([0, 2000])
# # plt.ylim([-0.05, 0.55])
# plt.grid()
# if save_flag:
#     plt.savefig(save_dir + '/qpos.png', dpi=600, bbox_inches='tight')
# if show_flag:
#     plt.title('qpos')
# i += 1
#
# plt.figure(i, figsize=(6, 4))
# plt.plot(qvel_j[:, 0], label=r'$q1$')
# plt.plot(qvel_j[:, 1], label=r'$q2$')
# plt.plot(qvel_j[:, 2], label=r'$q3$')
# plt.plot(qvel_j[:, 3], label=r'$q4$')
# plt.plot(qvel_j[:, 4], label=r'$q5$')
# plt.plot(qvel_j[:, 5], label=r'$q6$')
# plt.legend(loc='upper right')
# plt.xlabel('steps')
# plt.ylabel('velocity $\mathrm{(rad/s)}$')
# plt.xlim([0, 2000])
# # plt.ylim([-0.175, 0.15])
# plt.grid()
# if save_flag:
#     plt.savefig(save_dir + '/qvel.png', dpi=600, bbox_inches='tight')
# if show_flag:
#     plt.title('qvel')
# i += 1
#
# plt.figure(i, figsize=(6, 4))
# plt.plot(qacc_j[:, 0], label=r'$q1$')
# plt.plot(qacc_j[:, 1], label=r'$q2$')
# plt.plot(qacc_j[:, 2], label=r'$q3$')
# plt.plot(qacc_j[:, 3], label=r'$q4$')
# plt.plot(qacc_j[:, 4], label=r'$q5$')
# plt.plot(qacc_j[:, 5], label=r'$q6$')
# plt.legend(loc='upper right')
# plt.xlabel('steps')
# plt.ylabel(r'acceleration $\mathrm{(rad/s^2)}$')
# plt.xlim([0, 2000])
# # plt.ylim([-0.25, 0.2])
# plt.grid()
# if save_flag:
#     plt.savefig(save_dir + '/qacc.png', dpi=600, bbox_inches='tight')
# if show_flag:
#     plt.title('qacc')
# i += 1

# 笛卡尔空间与关节空间路径对比
fig = plt.figure(i, figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(xposture_l[:, 0], xposture_l[:, 1], xposture_l[:, 2], label='line-shape path in cartisian space')
ax.plot(xposture_c[:, 0], xposture_c[:, 1], xposture_c[:, 2], label='circle-shape path in cartisian space')
ax.plot(xpos_j[:, 0], xpos_j[:, 1], xpos_j[:, 2], label='path in joint space')
plt.legend(loc='upper right')
ax.set_xlabel(r'x $\mathrm{(m)}$')
ax.set_ylabel(r'y $\mathrm{(m)}$')
ax.set_zlabel(r'z $\mathrm{(m)}$')
ax.view_init(elev=40, azim=40)
if save_flag:
    plt.savefig(save_dir + '/pathes.png', dpi=600, bbox_inches='tight')
if show_flag:
    plt.title('xpos')
# ax.auto_scale_xyz([0.35, 0.65], [-0.2, 0.1], [0.3, 0.6])
i += 1

plt.show()
