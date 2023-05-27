#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""毕业论文中经典顺应性以及任意姿态顺应性对比

Write detailed description here

Write typical usage example here

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
4/25/23 8:48 PM   yinzikang      1.0         None
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import mujoco as mp

from gym_custom.envs.jk5_env_v7 import Jk5StickRobotWithController, env_kwargs
from gym_custom.envs.controller import AdmittanceController_v4


def draw_robot(xposture, a):
    for geom_idx in range(len(geom_name_list) - 3):
        id = mp.mj_name2id(jk5_with_controller.mjc_model, mp.mjtObj.mjOBJ_GEOM, geom_name_list[geom_idx])
        color = jk5_with_controller.mjc_model.geom(geom_name_list[geom_idx]).rgba
        color[-1] = a
        jk5_with_controller.viewer.add_marker(pos=xposture[geom_idx, :3],
                                              mat=xposture[geom_idx, 3:],
                                              type=mp.mjtGeom.mjGEOM_MESH,
                                              label='',
                                              rgba=color,
                                              dataid=2 * id - 2)
        for geom_idx in range(len(geom_name_list) - 3, len(geom_name_list) - 1):
            id = mp.mj_name2id(jk5_with_controller.mjc_model, mp.mjtObj.mjOBJ_GEOM, geom_name_list[geom_idx])
            color = jk5_with_controller.mjc_model.geom(geom_name_list[geom_idx]).rgba
            size = jk5_with_controller.mjc_model.geom(geom_name_list[geom_idx]).size
            color[-1] = a
            jk5_with_controller.viewer.add_marker(pos=xposture[geom_idx, :3],
                                                  mat=xposture[geom_idx, 3:],
                                                  size=[size[0], size[0], size[1]],
                                                  type=mp.mjtGeom.mjGEOM_CYLINDER,
                                                  label='',
                                                  rgba=color,
                                                  dataid=2 * id - 2)
        for geom_idx in range(len(geom_name_list) - 1, len(geom_name_list)):
            id = mp.mj_name2id(jk5_with_controller.mjc_model, mp.mjtObj.mjOBJ_GEOM, geom_name_list[geom_idx])
            color = jk5_with_controller.mjc_model.geom(geom_name_list[geom_idx]).rgba
            size = jk5_with_controller.mjc_model.geom(geom_name_list[geom_idx]).size
            color[-1] = a
            jk5_with_controller.viewer.add_marker(pos=xposture[geom_idx, :3],
                                                  mat=xposture[geom_idx, 3:],
                                                  size=size,
                                                  type=mp.mjtGeom.mjGEOM_SPHERE,
                                                  label='',
                                                  rgba=color,
                                                  dataid=2 * id - 2)


def draw_ellipsoid(k_r_p):
    jk5_with_controller.viewer.add_marker(pos=k_r_p[2],  # Position
                                          mat=k_r_p[1],
                                          label=" ",  # Text beside the marker
                                          type=mp.mjtGeom.mjGEOM_ELLIPSOID,  # Geomety type
                                          size=k_r_p[0],  # Size of the marker
                                          rgba=(84 / 255, 179 / 255, 69 / 255, 0.75),
                                          emission=1)  # RGBA of the marker

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['lines.linewidth'] = 2.0

geom_name_list = ['base_link_geom', 'link1_geom', 'link2_geom', 'link3_geom', 'link4_geom', 'link5_geom', 'link6_geom',
                  'ft_sensor_geom', 'stick_geom', 'dummy_body_geom']
posture_buffer = np.empty((2, len(geom_name_list), 12))
record_step_list = [0, 50, 100, 200, 300, 400, 500, 700, 1000, 1400]
k_r_p_list = []

_, rbt_controller_kwargs, _ = env_kwargs('desk with plan')
table_position = np.array([ - 0.2 * np.sqrt(2), -0.3, 0])
table_rotation = np.array([[np.sqrt(2) / 2, 0, -np.sqrt(2) / 2],
                           [0, 1, 0],
                           [np.sqrt(2) / 2, 0, np.sqrt(2) / 2]])
# 导纳控制+四元数误差+相对世界顺应性
rbt_controller_kwargs['controller'] = AdmittanceController_v4
rbt_controller_kwargs['controller_parameter']['SM'] = np.eye(3, dtype=np.float64)
admittance_buffer1 = dict()
jk5_with_controller = Jk5StickRobotWithController(**rbt_controller_kwargs)
jk5_with_controller.reset()
for status_name in jk5_with_controller.status_list:
    admittance_buffer1[status_name] = [jk5_with_controller.status[status_name]]
for step in range(rbt_controller_kwargs['step_num']):
    jk5_with_controller.step()
    # if step == 0:
    #     for geom_idx in range(len(geom_name_list)):
    #         posture_buffer[0][geom_idx][:3] = jk5_with_controller.data.geom(geom_name_list[geom_idx]).xpos
    #         posture_buffer[0][geom_idx][3:] = jk5_with_controller.data.geom(geom_name_list[geom_idx]).xmat
    # if step == 1000:
    #     for geom_idx in range(len(geom_name_list)):
    #         posture_buffer[1][geom_idx][:3] = jk5_with_controller.data.geom(geom_name_list[geom_idx]).xpos
    #         posture_buffer[1][geom_idx][3:] = jk5_with_controller.data.geom(geom_name_list[geom_idx]).xmat
    # if step in record_step_list:
    #     k_r_p_list.append([jk5_with_controller.status["controller_parameter"]["K"][:3] / 50000,
    #                        jk5_with_controller.status["controller_parameter"]["SM"],
    #                        jk5_with_controller.status["xpos"]])

    # jk5_with_controller.render()
    # if step > 0:
    #     draw_robot(posture_buffer[0], 1)
    # if step > 1000:
    #     draw_robot(posture_buffer[1], 0.66)
    # for idx in range(len(record_step_list)):
    #     if step > record_step_list[idx]:
    #         draw_ellipsoid(k_r_p_list[idx])
    for status_name in jk5_with_controller.status_list:
        admittance_buffer1[status_name].append(jk5_with_controller.status[status_name])
for status_name in jk5_with_controller.status_list:
    admittance_buffer1[status_name] = np.array(admittance_buffer1[status_name])


# 导纳控制+四元数误差+相对桌子顺应性
k_r_p_list = []
rbt_controller_kwargs['controller'] = AdmittanceController_v4
rbt_controller_kwargs['controller_parameter']['SM'] = table_rotation
admittance_buffer2 = dict()
jk5_with_controller = Jk5StickRobotWithController(**rbt_controller_kwargs)
jk5_with_controller.reset()
for status_name in jk5_with_controller.status_list:
    admittance_buffer2[status_name] = [jk5_with_controller.status[status_name]]
for step in range(rbt_controller_kwargs['step_num']):
    jk5_with_controller.step()
    # if step == 0:
    #     for geom_idx in range(len(geom_name_list)):
    #         posture_buffer[0][geom_idx][:3] = jk5_with_controller.data.geom(geom_name_list[geom_idx]).xpos
    #         posture_buffer[0][geom_idx][3:] = jk5_with_controller.data.geom(geom_name_list[geom_idx]).xmat
    # if step == 1000:
    #     for geom_idx in range(len(geom_name_list)):
    #         posture_buffer[1][geom_idx][:3] = jk5_with_controller.data.geom(geom_name_list[geom_idx]).xpos
    #         posture_buffer[1][geom_idx][3:] = jk5_with_controller.data.geom(geom_name_list[geom_idx]).xmat
    # if step in record_step_list:
    #     k_r_p_list.append([jk5_with_controller.status["controller_parameter"]["K"][:3] / 50000,
    #                        jk5_with_controller.status["controller_parameter"]["SM"],
    #                        jk5_with_controller.status["xpos"]])

    # jk5_with_controller.render()
    # if step > 0:
    #     draw_robot(posture_buffer[0], 1)
    # if step > 1000:
    #     draw_robot(posture_buffer[1], 0.66)
    # for idx in range(len(record_step_list)):
    #     if step > record_step_list[idx]:
    #         draw_ellipsoid(k_r_p_list[idx])
    for status_name in jk5_with_controller.status_list:
        admittance_buffer2[status_name].append(jk5_with_controller.status[status_name])
for status_name in jk5_with_controller.status_list:
    admittance_buffer2[status_name] = np.array(admittance_buffer2[status_name])

show_flag = True
save_flag = True
save_dir = './figs/different_frame'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

i = 0

i += 1
plt.figure(i)
plt.plot(admittance_buffer1["xpos"][:, 0], label='wx')
plt.plot(admittance_buffer1["xpos"][:, 1], label='wy')
plt.plot(admittance_buffer1["xpos"][:, 2], label='wz')
plt.plot(admittance_buffer2["xpos"][:, 0], label='tx')
plt.plot(admittance_buffer2["xpos"][:, 1], label='ty')
plt.plot(admittance_buffer2["xpos"][:, 2], label='tz')
plt.plot(admittance_buffer1["desired_xpos"][:, 0], label='dx')
plt.plot(admittance_buffer1["desired_xpos"][:, 1], label='dy')
plt.plot(admittance_buffer1["desired_xpos"][:, 2], label='dz')
plt.legend(loc='upper right')
plt.xlabel('steps')
plt.ylabel(r'position in world frame $\mathrm{(m)}$')
plt.xlim([0, 2000])
plt.grid()
if save_flag:
    plt.savefig(save_dir + '/恒定与任意参考系对比xpos world大.png', dpi=600, bbox_inches='tight')
if show_flag:
    plt.title('position in world frame')

i += 1
plt.figure(i)
plt.plot(admittance_buffer1["xpos"][:, 0], label='wx')
plt.plot(admittance_buffer1["xpos"][:, 1], label='wy')
plt.plot(admittance_buffer1["xpos"][:, 2], label='wz')
plt.plot(admittance_buffer2["xpos"][:, 0], label='tx')
plt.plot(admittance_buffer2["xpos"][:, 1], label='ty')
plt.plot(admittance_buffer2["xpos"][:, 2], label='tz')
plt.plot(admittance_buffer1["desired_xpos"][:, 0], label='dx')
plt.plot(admittance_buffer1["desired_xpos"][:, 1], label='dy')
plt.plot(admittance_buffer1["desired_xpos"][:, 2], label='dz')
# plt.legend(loc='upper right')
# plt.xlabel('steps')
# plt.ylabel(r'position in world frame $\mathrm{(m)}$')
plt.xlim([0, 500])
plt.grid()
if save_flag:
    plt.savefig(save_dir + '/恒定与任意参考系对比xpos world小.png', dpi=600, bbox_inches='tight')
if show_flag:
    plt.title('position in world frame')

i += 1
xpos1_table = (np.linalg.inv(table_rotation) @ (admittance_buffer1["xpos"] - table_position).transpose()).transpose()
xpos2_table = (np.linalg.inv(table_rotation) @ (admittance_buffer2["xpos"] - table_position).transpose()).transpose()
dpos_table = (np.linalg.inv(table_rotation) @ (
        admittance_buffer2["desired_xpos"] - table_position).transpose()).transpose()
plt.figure(i)
plt.plot(xpos1_table[:, 0], label='wx')
plt.plot(xpos1_table[:, 1], label='wy')
plt.plot(xpos1_table[:, 2], label='wz')
plt.plot(xpos2_table[:, 0], label='tx')
plt.plot(xpos2_table[:, 1], label='ty')
plt.plot(xpos2_table[:, 2], label='tz')
plt.plot(dpos_table[:, 0], label='dx')
plt.plot(dpos_table[:, 1], label='dy')
plt.plot(dpos_table[:, 2], label='dz')
plt.legend(loc='upper right')
plt.xlabel('steps')
plt.ylabel(r'position in table frame $\mathrm{(m)}$')
plt.xlim([0, 2000])
plt.grid()
if save_flag:
    plt.savefig(save_dir + '/恒定与任意参考系对比xpos table大.png', dpi=600, bbox_inches='tight')
if show_flag:
    plt.title('position in table frame')

i += 1
xpos1_table = (np.linalg.inv(table_rotation) @ (admittance_buffer1["xpos"] - table_position).transpose()).transpose()
xpos2_table = (np.linalg.inv(table_rotation) @ (admittance_buffer2["xpos"] - table_position).transpose()).transpose()
dpos_table = (np.linalg.inv(table_rotation) @ (
        admittance_buffer2["desired_xpos"] - table_position).transpose()).transpose()
plt.figure(i)
plt.plot(xpos1_table[:, 0], label='wx')
plt.plot(xpos1_table[:, 1], label='wy')
plt.plot(xpos1_table[:, 2], label='wz')
plt.plot(xpos2_table[:, 0], label='tx')
plt.plot(xpos2_table[:, 1], label='ty')
plt.plot(xpos2_table[:, 2], label='tz')
plt.plot(dpos_table[:, 0], label='dx')
plt.plot(dpos_table[:, 1], label='dy')
plt.plot(dpos_table[:, 2], label='dz')
# plt.legend(loc='upper right')
# plt.xlabel('steps')
# plt.ylabel(r'position in table frame $\mathrm{(m)}$')
plt.xlim([0, 500])
plt.grid()
if save_flag:
    plt.savefig(save_dir + '/恒定与任意参考系对比xpos table小.png', dpi=600, bbox_inches='tight')
if show_flag:
    plt.title('position in table frame')

i += 1
pos_error1 = admittance_buffer1["xpos"] - admittance_buffer1["desired_xpos"]
pos_error2 = admittance_buffer2["xpos"] - admittance_buffer2["desired_xpos"]
plt.figure(i)
plt.plot(pos_error1[:, 0], label='wx-dx')
plt.plot(pos_error1[:, 1], label='wy-dy')
plt.plot(pos_error1[:, 2], label='wz-dz')
plt.plot(pos_error2[:, 0], label='tx-dx')
plt.plot(pos_error2[:, 1], label='ty-dy')
plt.plot(pos_error2[:, 2], label='tz-dz')
plt.legend(loc='upper right')
plt.xlabel('steps')
plt.ylabel(r'position error in world frame $\mathrm{(m)}$')
plt.xlim([0, 2000])
plt.grid()
if save_flag:
    plt.savefig(save_dir + '/恒定与任意参考系对比xpos error world大.png', dpi=600, bbox_inches='tight')
if show_flag:
    plt.title('position error in world frame')

i += 1
pos_error1 = admittance_buffer1["xpos"] - admittance_buffer1["desired_xpos"]
pos_error2 = admittance_buffer2["xpos"] - admittance_buffer2["desired_xpos"]
plt.figure(i)
plt.plot(pos_error1[:, 0], label='wx-dx')
plt.plot(pos_error1[:, 1], label='wy-dy')
plt.plot(pos_error1[:, 2], label='wz-dz')
plt.plot(pos_error2[:, 0], label='tx-dx')
plt.plot(pos_error2[:, 1], label='ty-dy')
plt.plot(pos_error2[:, 2], label='tz-dz')
# plt.legend(loc='upper right')
# plt.xlabel('steps')
# plt.ylabel(r'position error in world frame $\mathrm{(m)}$')
plt.xlim([0, 500])
plt.grid()
if save_flag:
    plt.savefig(save_dir + '/恒定与任意参考系对比xpos error world小.png', dpi=600, bbox_inches='tight')
if show_flag:
    plt.title('position error in world frame')

i += 1
pos_error1_table = xpos1_table - dpos_table
pos_error2_table = xpos2_table - dpos_table
plt.figure(i)
plt.plot(pos_error1_table[:, 0], label='wx-dx')
plt.plot(pos_error1_table[:, 1], label='wy-dy')
plt.plot(pos_error1_table[:, 2], label='wz-dz')
plt.plot(pos_error2_table[:, 0], label='tx-dx')
plt.plot(pos_error2_table[:, 1], label='ty-dy')
plt.plot(pos_error2_table[:, 2], label='tz-dz')
plt.legend(loc='upper right')
plt.xlabel('steps')
plt.ylabel(r'position error in table frame $\mathrm{(m)}$')
plt.xlim([0, 2000])
plt.grid()
if save_flag:
    plt.savefig(save_dir + '/恒定与任意参考系对比xpos error table大.png', dpi=600, bbox_inches='tight')
if show_flag:
    plt.title('position error in table frame')

i += 1
pos_error1_table = xpos1_table - dpos_table
pos_error2_table = xpos2_table - dpos_table
plt.figure(i)
plt.plot(pos_error1_table[:, 0], label='wx-dx')
plt.plot(pos_error1_table[:, 1], label='wy-dy')
plt.plot(pos_error1_table[:, 2], label='wz-dz')
plt.plot(pos_error2_table[:, 0], label='tx-dx')
plt.plot(pos_error2_table[:, 1], label='ty-dy')
plt.plot(pos_error2_table[:, 2], label='tz-dz')
# plt.legend(loc='upper right')
# plt.xlabel('steps')
# plt.ylabel(r'position error in table frame $\mathrm{(m)}$')
plt.xlim([0, 500])
plt.grid()
if save_flag:
    plt.savefig(save_dir + '/恒定与任意参考系对比xpos error table小.png', dpi=600, bbox_inches='tight')
if show_flag:
    plt.title('position error in table frame')

i += 1
force1 = admittance_buffer1["contact_force"][:, :3]
force2 = admittance_buffer2["contact_force"][:, :3]
forced = admittance_buffer1["desired_force"][:, :3]
plt.figure(i)
plt.plot(force1[:, 0], label='wx')
plt.plot(force1[:, 1], label='wy')
plt.plot(force1[:, 2], label='wz')
plt.plot(force2[:, 0], label='tx')
plt.plot(force2[:, 1], label='ty')
plt.plot(force2[:, 2], label='tz')
plt.plot(forced[:, 0], label='dx')
plt.plot(forced[:, 1], label='dy')
plt.plot(forced[:, 2], label='dz')
plt.legend(loc='upper right')
plt.xlabel('steps')
plt.ylabel(r'force in world frame $\mathrm{(N)}$')
plt.xlim([0, 2000])
plt.grid()
if save_flag:
    plt.savefig(save_dir + '/恒定与任意参考系对比force world大.png', dpi=600, bbox_inches='tight')
if show_flag:
    plt.title('force in world frame')

i += 1
force1 = admittance_buffer1["contact_force"][:, :3]
force2 = admittance_buffer2["contact_force"][:, :3]
forced = admittance_buffer1["desired_force"][:, :3]
plt.figure(i)
plt.plot(force1[:, 0], label='wx')
plt.plot(force1[:, 1], label='wy')
plt.plot(force1[:, 2], label='wz')
plt.plot(force2[:, 0], label='tx')
plt.plot(force2[:, 1], label='ty')
plt.plot(force2[:, 2], label='tz')
plt.plot(forced[:, 0], label='dx')
plt.plot(forced[:, 1], label='dy')
plt.plot(forced[:, 2], label='dz')
# plt.legend(loc='upper right')
# plt.xlabel('steps')
# plt.ylabel(r'force in world frame $\mathrm{(N)}$')
plt.xlim([0, 500])
plt.grid()
if save_flag:
    plt.savefig(save_dir + '/恒定与任意参考系对比force world小.png', dpi=600, bbox_inches='tight')
if show_flag:
    plt.title('force in world frame')

i += 1
force1_table = (np.linalg.inv(table_rotation) @ (admittance_buffer1["contact_force"][:, :3]).transpose()).transpose()
force2_table = (np.linalg.inv(table_rotation) @ (admittance_buffer2["contact_force"][:, :3]).transpose()).transpose()
forced_table = (np.linalg.inv(table_rotation) @ (admittance_buffer1["desired_force"][:, :3]).transpose()).transpose()
plt.figure(i)
plt.plot(force1_table[:, 0], label='wx')
plt.plot(force1_table[:, 1], label='wy')
plt.plot(force1_table[:, 2], label='wz')
plt.plot(force2_table[:, 0], label='tx')
plt.plot(force2_table[:, 1], label='ty')
plt.plot(force2_table[:, 2], label='tz')
plt.plot(forced_table[:, 0], label='dx')
plt.plot(forced_table[:, 1], label='dy')
plt.plot(forced_table[:, 2], label='dz')
plt.legend(loc='upper right')
plt.xlabel('steps')
plt.ylabel(r'force in table frame $\mathrm{(N)}$')
plt.xlim([0, 2000])
plt.grid()
if save_flag:
    plt.savefig(save_dir + '/恒定与任意参考系对比force table大.png', dpi=600, bbox_inches='tight')
if show_flag:
    plt.title('force in table frame')

i += 1
force1_table = (np.linalg.inv(table_rotation) @ (admittance_buffer1["contact_force"][:, :3]).transpose()).transpose()
force2_table = (np.linalg.inv(table_rotation) @ (admittance_buffer2["contact_force"][:, :3]).transpose()).transpose()
forced_table = (np.linalg.inv(table_rotation) @ (admittance_buffer1["desired_force"][:, :3]).transpose()).transpose()
plt.figure(i)
plt.plot(force1_table[:, 0], label='wx')
plt.plot(force1_table[:, 1], label='wy')
plt.plot(force1_table[:, 2], label='wz')
plt.plot(force2_table[:, 0], label='tx')
plt.plot(force2_table[:, 1], label='ty')
plt.plot(force2_table[:, 2], label='tz')
plt.plot(forced_table[:, 0], label='dx')
plt.plot(forced_table[:, 1], label='dy')
plt.plot(forced_table[:, 2], label='dz')
# plt.legend(loc='upper right')
# plt.xlabel('steps')
# plt.ylabel(r'force in table frame $\mathrm{(N)}$')
plt.xlim([0, 500])
plt.grid()
if save_flag:
    plt.savefig(save_dir + '/恒定与任意参考系对比force table小.png', dpi=600, bbox_inches='tight')
if show_flag:
    plt.title('force in table frame')

plt.show()
