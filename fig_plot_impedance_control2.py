#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""毕业论文中各种姿态误差的图对比

self.data.xfrc_applied[mp.mj_name2id(self.mjc_model, mp.mjtObj.mjOBJ_BODY, 'dummy_body')][3] = 3
self.data.xfrc_applied[mp.mj_name2id(self.mjc_model, mp.mjtObj.mjOBJ_BODY, 'dummy_body')][4] = 2
self.data.xfrc_applied[mp.mj_name2id(self.mjc_model, mp.mjtObj.mjOBJ_BODY, 'dummy_body')][5] = 3

Write typical usage example here

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
4/24/23 10:31 AM   yinzikang      1.0         None
"""

import os
import matplotlib.pyplot as plt
import numpy as np

from gym_custom.envs.jk5_env_v7 import Jk5StickRobotWithController, env_kwargs
from gym_custom.envs.controller import AdmittanceController_v3, orientation_error_quat_with_mat, \
    orientation_error_axis_angle_with_mat

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10.5
plt.rcParams['lines.linewidth'] = 2.0

_, rbt_controller_kwargs, _ = env_kwargs('fig_plot')
# 导纳控制+轴角误差数据
rbt_controller_kwargs['controller'] = AdmittanceController_v3
admittance_buffer1 = dict()
jk5_with_controller = Jk5StickRobotWithController(**rbt_controller_kwargs)
jk5_with_controller.controller.orientation_error = orientation_error_axis_angle_with_mat
jk5_with_controller.reset()
for status_name in jk5_with_controller.status_list:
    admittance_buffer1[status_name] = [jk5_with_controller.status[status_name]]
for _ in range(rbt_controller_kwargs['step_num']):
    jk5_with_controller.step()
    # jk5_with_controller.render()
    for status_name in jk5_with_controller.status_list:
        admittance_buffer1[status_name].append(jk5_with_controller.status[status_name])
for status_name in jk5_with_controller.status_list:
    admittance_buffer1[status_name] = np.array(admittance_buffer1[status_name])

# 导纳控制+四元数误差数据
rbt_controller_kwargs['controller'] = AdmittanceController_v3
admittance_buffer2 = dict()
jk5_with_controller = Jk5StickRobotWithController(**rbt_controller_kwargs)
jk5_with_controller.controller.orientation_error = orientation_error_quat_with_mat
jk5_with_controller.reset()
for status_name in jk5_with_controller.status_list:
    admittance_buffer2[status_name] = [jk5_with_controller.status[status_name]]
for _ in range(rbt_controller_kwargs['step_num']):
    jk5_with_controller.step()
    # jk5_with_controller.render()
    for status_name in jk5_with_controller.status_list:
        admittance_buffer2[status_name].append(jk5_with_controller.status[status_name])
for status_name in jk5_with_controller.status_list:
    admittance_buffer2[status_name] = np.array(admittance_buffer2[status_name])

color1 = (190 / 255, 226 / 255, 255 / 255)
color2 = (251 / 255, 154 / 255, 255 / 255)
show_flag = True
save_flag = True
save_dir = './figs/different_error'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

i = 0

i += 1
plt.figure(i)
plt.plot(admittance_buffer1["xquat"][:, 0], label='ax')
plt.plot(admittance_buffer1["xquat"][:, 1], label='ay')
plt.plot(admittance_buffer1["xquat"][:, 2], label='az')
plt.plot(admittance_buffer1["xquat"][:, 3], label='aw')
plt.plot(admittance_buffer2["xquat"][:, 0], label='qx')
plt.plot(admittance_buffer2["xquat"][:, 1], label='qy')
plt.plot(admittance_buffer2["xquat"][:, 2], label='qz')
plt.plot(admittance_buffer2["xquat"][:, 3], label='qw')
plt.plot(admittance_buffer1["desired_xquat"][:, 0], label='dx')
plt.plot(admittance_buffer1["desired_xquat"][:, 1], label='dy')
plt.plot(admittance_buffer1["desired_xquat"][:, 2], label='dz', color=color1)
plt.plot(admittance_buffer1["desired_xquat"][:, 3], label='dw', color=color2)
plt.legend(loc='upper right')
plt.xlabel('steps')
plt.ylabel('quaternion')
plt.xlim([0, 2000])
plt.grid()
if save_flag:
    plt.savefig(save_dir + '/轴角与四元数对比xquat大.png', dpi=600, bbox_inches='tight')
if show_flag:
    plt.title('quaternion')

i += 1
plt.figure(i)
plt.plot(admittance_buffer1["xquat"][:, 0], label='ax')
plt.plot(admittance_buffer1["xquat"][:, 1], label='ay')
plt.plot(admittance_buffer1["xquat"][:, 2], label='az')
plt.plot(admittance_buffer1["xquat"][:, 3], label='aw')
plt.plot(admittance_buffer2["xquat"][:, 0], label='qx')
plt.plot(admittance_buffer2["xquat"][:, 1], label='qy')
plt.plot(admittance_buffer2["xquat"][:, 2], label='qz')
plt.plot(admittance_buffer2["xquat"][:, 3], label='qw')
plt.plot(admittance_buffer1["desired_xquat"][:, 0], label='dx')
plt.plot(admittance_buffer1["desired_xquat"][:, 1], label='dy')
plt.plot(admittance_buffer1["desired_xquat"][:, 2], label='dz', color=color1)
plt.plot(admittance_buffer1["desired_xquat"][:, 3], label='dw', color=color2)
# plt.legend(loc='upper right')
# plt.xlabel('steps')
# plt.ylabel('quaternion')
plt.xlim([0, 250])
plt.grid()
if save_flag:
    plt.savefig(save_dir + '/轴角与四元数对比xquat小.png', dpi=600, bbox_inches='tight')
if show_flag:
    plt.title('quaternion')

i += 1
plt.figure(i)
oe1 = []
oe2 = []
for j in range(rbt_controller_kwargs['step_num']):
    oe1.append(
        orientation_error_axis_angle_with_mat(admittance_buffer1["desired_xmat"][j], admittance_buffer1["xmat"][j]))
    oe2.append(orientation_error_quat_with_mat(admittance_buffer2["desired_xmat"][j], admittance_buffer2["xmat"][j]))
oe1 = np.array(oe1)
oe2 = np.array(oe2)
plt.plot(oe1[:, 0], label='ax')
plt.plot(oe1[:, 1], label='ay')
plt.plot(oe1[:, 2], label='az')
plt.plot(oe2[:, 0], label='qx')
plt.plot(oe2[:, 1], label='qy')
plt.plot(oe2[:, 2], label='qz')
plt.legend(loc='upper right')
plt.xlabel('steps')
plt.ylabel(r'orientation error')
plt.xlim([0, 2000])
plt.grid()
if save_flag:
    plt.savefig(save_dir + '/轴角与四元数对比orientation error大.png', dpi=600, bbox_inches='tight')
if show_flag:
    plt.title('orientation error')

i += 1
plt.figure(i)
oe1 = []
oe2 = []
for j in range(rbt_controller_kwargs['step_num']):
    oe1.append(
        orientation_error_axis_angle_with_mat(admittance_buffer1["desired_xmat"][j], admittance_buffer1["xmat"][j]))
    oe2.append(orientation_error_quat_with_mat(admittance_buffer2["desired_xmat"][j], admittance_buffer2["xmat"][j]))
oe1 = np.array(oe1)
oe2 = np.array(oe2)
plt.plot(oe1[:, 0], label='ax')
plt.plot(oe1[:, 1], label='ay')
plt.plot(oe1[:, 2], label='az')
plt.plot(oe2[:, 0], label='qx')
plt.plot(oe2[:, 1], label='qy')
plt.plot(oe2[:, 2], label='qz')
# plt.legend(loc='upper right')
# plt.xlabel('steps')
# plt.ylabel(r'orientation error')
plt.xlim([0, 250])
plt.grid()
if save_flag:
    plt.savefig(save_dir + '/轴角与四元数对比orientation error小.png', dpi=600, bbox_inches='tight')
if show_flag:
    plt.title('orientation error')

i += 1
plt.figure(i)
plt.plot(admittance_buffer1["xvel"][:, 3], label='ax')
plt.plot(admittance_buffer1["xvel"][:, 4], label='ay')
plt.plot(admittance_buffer1["xvel"][:, 5], label='az')
plt.plot(admittance_buffer2["xvel"][:, 3], label='qx')
plt.plot(admittance_buffer2["xvel"][:, 4], label='qy')
plt.plot(admittance_buffer2["xvel"][:, 5], label='qz')
plt.plot(admittance_buffer1["desired_xvel"][:, 3], label='dx')
plt.plot(admittance_buffer1["desired_xvel"][:, 4], label='dy')
plt.plot(admittance_buffer1["desired_xvel"][:, 5], label='dz')
plt.legend(loc='upper right')
plt.xlabel('steps')
plt.ylabel(r'velocity $\mathrm{(rad/s)}$')
plt.xlim([0, 2000])
plt.grid()
if save_flag:
    plt.savefig(save_dir + '/阻抗与导纳对比角velocity大.png', dpi=600, bbox_inches='tight')
if show_flag:
    plt.title('vel')

i += 1
plt.figure(i)
plt.plot(admittance_buffer1["xvel"][:, 3], label='ax')
plt.plot(admittance_buffer1["xvel"][:, 4], label='ay')
plt.plot(admittance_buffer1["xvel"][:, 5], label='az')
plt.plot(admittance_buffer2["xvel"][:, 3], label='qx')
plt.plot(admittance_buffer2["xvel"][:, 4], label='qy')
plt.plot(admittance_buffer2["xvel"][:, 5], label='qz')
plt.plot(admittance_buffer1["desired_xvel"][:, 3], label='dx')
plt.plot(admittance_buffer1["desired_xvel"][:, 4], label='dy')
plt.plot(admittance_buffer1["desired_xvel"][:, 5], label='dz')
# plt.legend(loc='upper right')
# plt.xlabel('steps')
# plt.ylabel(r'velocity $\mathrm{(rad/s)}$')
plt.xlim([0, 250])
plt.grid()
if save_flag:
    plt.savefig(save_dir + '/阻抗与导纳对比角velocity小.png', dpi=600, bbox_inches='tight')
if show_flag:
    plt.title('vel')

i += 1
plt.figure(i)
plt.plot(jk5_with_controller.control_frequency * np.diff(admittance_buffer1["xvel"][:, 3], axis=0), label='ax')
plt.plot(jk5_with_controller.control_frequency * np.diff(admittance_buffer1["xvel"][:, 4], axis=0), label='ay')
plt.plot(jk5_with_controller.control_frequency * np.diff(admittance_buffer1["xvel"][:, 5], axis=0), label='az')
plt.plot(jk5_with_controller.control_frequency * np.diff(admittance_buffer2["xvel"][:, 3], axis=0), label='qx')
plt.plot(jk5_with_controller.control_frequency * np.diff(admittance_buffer2["xvel"][:, 4], axis=0), label='qy')
plt.plot(jk5_with_controller.control_frequency * np.diff(admittance_buffer2["xvel"][:, 5], axis=0), label='qz')
plt.plot(jk5_with_controller.control_frequency * np.diff(admittance_buffer1["desired_xvel"][:, 3], axis=0), label='dx')
plt.plot(jk5_with_controller.control_frequency * np.diff(admittance_buffer1["desired_xvel"][:, 4], axis=0), label='dy')
plt.plot(jk5_with_controller.control_frequency * np.diff(admittance_buffer1["desired_xvel"][:, 5], axis=0), label='dz')
plt.legend(loc='upper right')
plt.xlabel('steps')
plt.ylabel(r'acceleration $\mathrm{(rad/s^2)}$')
plt.xlim([0, 2000])
plt.grid()
if save_flag:
    plt.savefig(save_dir + '/阻抗与导纳对比角acc大.png', dpi=600, bbox_inches='tight')
if show_flag:
    plt.title('acc')

i += 1
plt.figure(i)
plt.plot(jk5_with_controller.control_frequency * np.diff(admittance_buffer1["xvel"][:, 3], axis=0), label='ax')
plt.plot(jk5_with_controller.control_frequency * np.diff(admittance_buffer1["xvel"][:, 4], axis=0), label='ay')
plt.plot(jk5_with_controller.control_frequency * np.diff(admittance_buffer1["xvel"][:, 5], axis=0), label='az')
plt.plot(jk5_with_controller.control_frequency * np.diff(admittance_buffer2["xvel"][:, 3], axis=0), label='qx')
plt.plot(jk5_with_controller.control_frequency * np.diff(admittance_buffer2["xvel"][:, 4], axis=0), label='qy')
plt.plot(jk5_with_controller.control_frequency * np.diff(admittance_buffer2["xvel"][:, 5], axis=0), label='qz')
plt.plot(jk5_with_controller.control_frequency * np.diff(admittance_buffer1["desired_xvel"][:, 3], axis=0), label='dx')
plt.plot(jk5_with_controller.control_frequency * np.diff(admittance_buffer1["desired_xvel"][:, 4], axis=0), label='dy')
plt.plot(jk5_with_controller.control_frequency * np.diff(admittance_buffer1["desired_xvel"][:, 5], axis=0), label='dz')
# plt.legend(loc='upper right')
# plt.xlabel('steps')
# plt.ylabel(r'acceleration $\mathrm{(rad/s^2)}$')
plt.xlim([0, 250])
plt.grid()
if save_flag:
    plt.savefig(save_dir + '/阻抗与导纳对比角acc小.png', dpi=600, bbox_inches='tight')
if show_flag:
    plt.title('acc')

plt.show()
