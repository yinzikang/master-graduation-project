#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""毕业论文中各种阻抗控制的位置图对比

self.data.xfrc_applied[mp.mj_name2id(self.mjc_model, mp.mjtObj.mjOBJ_BODY, 'dummy_body')][0] = 3
self.data.xfrc_applied[mp.mj_name2id(self.mjc_model, mp.mjtObj.mjOBJ_BODY, 'dummy_body')][1] = 2
self.data.xfrc_applied[mp.mj_name2id(self.mjc_model, mp.mjtObj.mjOBJ_BODY, 'dummy_body')][2] = 3


Write typical usage example here

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
4/24/23 10:31 AM   yinzikang      1.0         None
"""

import os
import matplotlib.pyplot as plt
import numpy as np

from gym_custom.envs.jk5_env_v7 import Jk5StickRobotWithController, env_kwargs
from gym_custom.envs.controller import AdmittanceController_v2, ImpedanceController_v2

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['lines.linewidth'] = 2.0

_, rbt_controller_kwargs, _ = env_kwargs('fig_plot')
# 阻抗控制数据
rbt_controller_kwargs['controller'] = ImpedanceController_v2
impedance_buffer = dict()
jk5_with_controller = Jk5StickRobotWithController(**rbt_controller_kwargs)
jk5_with_controller.reset()
for status_name in jk5_with_controller.status_list:
    impedance_buffer[status_name] = [jk5_with_controller.status[status_name]]
for _ in range(rbt_controller_kwargs['step_num']):
    jk5_with_controller.step()
    # jk5_with_controller.render()
    for status_name in jk5_with_controller.status_list:
        impedance_buffer[status_name].append(jk5_with_controller.status[status_name])
for status_name in jk5_with_controller.status_list:
    impedance_buffer[status_name] = np.array(impedance_buffer[status_name])
# 导纳控制数据
rbt_controller_kwargs['controller'] = AdmittanceController_v2
admittance_buffer = dict()
jk5_with_controller = Jk5StickRobotWithController(**rbt_controller_kwargs)
jk5_with_controller.reset()
for status_name in jk5_with_controller.status_list:
    admittance_buffer[status_name] = [jk5_with_controller.status[status_name]]
for _ in range(rbt_controller_kwargs['step_num']):
    jk5_with_controller.step()
    # jk5_with_controller.render()
    for status_name in jk5_with_controller.status_list:
        admittance_buffer[status_name].append(jk5_with_controller.status[status_name])
for status_name in jk5_with_controller.status_list:
    admittance_buffer[status_name] = np.array(admittance_buffer[status_name])

show_flag = True
save_flag = True
save_dir = './figs/different_control'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

i = 0

i += 1
plt.figure(i, figsize=(6, 4))
plt.plot(impedance_buffer["xpos"][:, 0], label='ix')
plt.plot(impedance_buffer["xpos"][:, 1], label='iy')
plt.plot(impedance_buffer["xpos"][:, 2], label='iz')
plt.plot(admittance_buffer["xpos"][:, 0], label='ax')
plt.plot(admittance_buffer["xpos"][:, 1], label='ay')
plt.plot(admittance_buffer["xpos"][:, 2], label='az')
plt.plot(impedance_buffer["desired_xpos"][:, 0], label='dx')
plt.plot(impedance_buffer["desired_xpos"][:, 1], label='dy')
plt.plot(impedance_buffer["desired_xpos"][:, 2], label='dz')
plt.legend(loc='upper right')
plt.xlabel('steps')
plt.ylabel(r'position $\mathrm{(m)}$')
plt.xlim([0, 2000])
plt.grid()
if save_flag:
    plt.savefig(save_dir + '/阻抗与导纳对比xpos大.png', dpi=600, bbox_inches='tight')
if show_flag:
    plt.title('pos')

i += 1
plt.figure(i, figsize=(6, 4))
plt.plot(impedance_buffer["xpos"][:, 0], label='ix')
plt.plot(impedance_buffer["xpos"][:, 1], label='iy')
plt.plot(impedance_buffer["xpos"][:, 2], label='iz')
plt.plot(admittance_buffer["xpos"][:, 0], label='ax')
plt.plot(admittance_buffer["xpos"][:, 1], label='ay')
plt.plot(admittance_buffer["xpos"][:, 2], label='az')
plt.plot(impedance_buffer["desired_xpos"][:, 0], label='dx')
plt.plot(impedance_buffer["desired_xpos"][:, 1], label='dy')
plt.plot(impedance_buffer["desired_xpos"][:, 2], label='dz')
# plt.legend(loc='upper right')
# plt.xlabel('steps')
# plt.ylabel('')
plt.xlim([0, 250])
plt.grid()
if save_flag:
    plt.savefig(save_dir + '/阻抗与导纳对比xpos小.png', dpi=600, bbox_inches='tight')
if show_flag:
    plt.title('pos')

i += 1
plt.figure(i, figsize=(6, 4))
plt.plot((impedance_buffer["xpos"] - impedance_buffer["desired_xpos"])[:, 0], label='ix')
plt.plot((impedance_buffer["xpos"] - impedance_buffer["desired_xpos"])[:, 1], label='iy')
plt.plot((impedance_buffer["xpos"] - impedance_buffer["desired_xpos"])[:, 2], label='iz')
plt.plot((admittance_buffer["xpos"] - impedance_buffer["desired_xpos"])[:, 0], label='ax')
plt.plot((admittance_buffer["xpos"] - impedance_buffer["desired_xpos"])[:, 1], label='ay')
plt.plot((admittance_buffer["xpos"] - impedance_buffer["desired_xpos"])[:, 2], label='az')
plt.legend(loc='upper right')
plt.xlabel('steps')
plt.ylabel(r'position error $\mathrm{(m)}$')
plt.xlim([0, 2000])
plt.grid()
if save_flag:
    plt.savefig(save_dir + '/阻抗与导纳对比xpos error大.png', dpi=600, bbox_inches='tight')
if show_flag:
    plt.title('pos error')

i += 1
plt.figure(i, figsize=(6, 4))
plt.plot((impedance_buffer["xpos"] - impedance_buffer["desired_xpos"])[:, 0], label='ix-dx')
plt.plot((impedance_buffer["xpos"] - impedance_buffer["desired_xpos"])[:, 1], label='iy-dy')
plt.plot((impedance_buffer["xpos"] - impedance_buffer["desired_xpos"])[:, 2], label='iz-dz')
plt.plot((admittance_buffer["xpos"] - impedance_buffer["desired_xpos"])[:, 0], label='ax-dx')
plt.plot((admittance_buffer["xpos"] - impedance_buffer["desired_xpos"])[:, 1], label='ay-dy')
plt.plot((admittance_buffer["xpos"] - impedance_buffer["desired_xpos"])[:, 2], label='az-dz')
# plt.legend(loc='upper right')
# plt.xlabel('steps')
# plt.ylabel(r'position error $\mathrm{(m)}$')
plt.xlim([0, 250])
plt.grid()
if save_flag:
    plt.savefig(save_dir + '/阻抗与导纳对比xpos error小.png', dpi=600, bbox_inches='tight')
if show_flag:
    plt.title('pos error')

i += 1
plt.figure(i, figsize=(6, 4))
plt.plot(impedance_buffer["xvel"][:, 0], label='ix')
plt.plot(impedance_buffer["xvel"][:, 1], label='iy')
plt.plot(impedance_buffer["xvel"][:, 2], label='iz')
plt.plot(admittance_buffer["xvel"][:, 0], label='ax')
plt.plot(admittance_buffer["xvel"][:, 1], label='ay')
plt.plot(admittance_buffer["xvel"][:, 2], label='az')
plt.plot(impedance_buffer["desired_xvel"][:, 0], label='dx')
plt.plot(impedance_buffer["desired_xvel"][:, 1], label='dy')
plt.plot(impedance_buffer["desired_xvel"][:, 2], label='dz')
plt.legend(loc='upper right')
plt.xlabel('steps')
plt.ylabel(r'velocity $\mathrm{(m/s)}$')
plt.xlim([0, 2000])
plt.grid()
if save_flag:
    plt.savefig(save_dir + '/阻抗与导纳对比velocity大.png', dpi=600, bbox_inches='tight')
if show_flag:
    plt.title('vel')

i += 1
plt.figure(i, figsize=(6, 4))
plt.plot(impedance_buffer["xvel"][:, 0], label='ix')
plt.plot(impedance_buffer["xvel"][:, 1], label='iy')
plt.plot(impedance_buffer["xvel"][:, 2], label='iz')
plt.plot(admittance_buffer["xvel"][:, 0], label='ax')
plt.plot(admittance_buffer["xvel"][:, 1], label='ay')
plt.plot(admittance_buffer["xvel"][:, 2], label='az')
plt.plot(impedance_buffer["desired_xvel"][:, 0], label='dx')
plt.plot(impedance_buffer["desired_xvel"][:, 1], label='dy')
plt.plot(impedance_buffer["desired_xvel"][:, 2], label='dz')
# plt.legend(loc='upper right')
# plt.xlabel('steps')
# plt.ylabel(r'velocity $\mathrm{(m/s)}$')
plt.xlim([0, 250])
plt.grid()
if save_flag:
    plt.savefig(save_dir + '/阻抗与导纳对比velocity小.png', dpi=600, bbox_inches='tight')
if show_flag:
    plt.title('vel')

i += 1
plt.figure(i, figsize=(6, 4))
plt.plot(jk5_with_controller.control_frequency * np.diff(impedance_buffer["xvel"][:, 0], axis=0), label='ix')
plt.plot(jk5_with_controller.control_frequency * np.diff(impedance_buffer["xvel"][:, 1], axis=0), label='iy')
plt.plot(jk5_with_controller.control_frequency * np.diff(impedance_buffer["xvel"][:, 2], axis=0), label='iz')
plt.plot(jk5_with_controller.control_frequency * np.diff(admittance_buffer["xvel"][:, 0], axis=0), label='ax')
plt.plot(jk5_with_controller.control_frequency * np.diff(admittance_buffer["xvel"][:, 1], axis=0), label='ay')
plt.plot(jk5_with_controller.control_frequency * np.diff(admittance_buffer["xvel"][:, 2], axis=0), label='az')
plt.plot(jk5_with_controller.control_frequency * np.diff(impedance_buffer["desired_xvel"][:, 0], axis=0), label='dx')
plt.plot(jk5_with_controller.control_frequency * np.diff(impedance_buffer["desired_xvel"][:, 1], axis=0), label='dy')
plt.plot(jk5_with_controller.control_frequency * np.diff(impedance_buffer["desired_xvel"][:, 2], axis=0), label='dz')
plt.legend(loc='upper right')
plt.xlabel('steps')
plt.ylabel(r'acceleration $\mathrm{(m/s^2)}$')
plt.xlim([0, 2000])
plt.grid()
if save_flag:
    plt.savefig(save_dir + '/阻抗与导纳对比acc大.png', dpi=600, bbox_inches='tight')
if show_flag:
    plt.title('acc')

i += 1
plt.figure(i, figsize=(6, 4))
plt.plot(jk5_with_controller.control_frequency * np.diff(impedance_buffer["xvel"][:, 0], axis=0), label='ix')
plt.plot(jk5_with_controller.control_frequency * np.diff(impedance_buffer["xvel"][:, 1], axis=0), label='iy')
plt.plot(jk5_with_controller.control_frequency * np.diff(impedance_buffer["xvel"][:, 2], axis=0), label='iz')
plt.plot(jk5_with_controller.control_frequency * np.diff(admittance_buffer["xvel"][:, 0], axis=0), label='ax')
plt.plot(jk5_with_controller.control_frequency * np.diff(admittance_buffer["xvel"][:, 1], axis=0), label='ay')
plt.plot(jk5_with_controller.control_frequency * np.diff(admittance_buffer["xvel"][:, 2], axis=0), label='az')
plt.plot(jk5_with_controller.control_frequency * np.diff(impedance_buffer["desired_xvel"][:, 0], axis=0), label='dx')
plt.plot(jk5_with_controller.control_frequency * np.diff(impedance_buffer["desired_xvel"][:, 1], axis=0), label='dy')
plt.plot(jk5_with_controller.control_frequency * np.diff(impedance_buffer["desired_xvel"][:, 2], axis=0), label='dz')
# plt.legend(loc='upper right')
# plt.xlabel('steps')
# plt.ylabel(r'acceleration $\mathrm{(m/s^2)}$')
plt.xlim([0, 250])
plt.grid()
if save_flag:
    plt.savefig(save_dir + '/阻抗与导纳对比acc小.png', dpi=600, bbox_inches='tight')
if show_flag:
    plt.title('acc')

plt.show()
