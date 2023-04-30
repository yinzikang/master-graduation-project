#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Write target here

Write detailed description here

Write typical usage example here

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
4/28/23 11:00 AM   yinzikang      1.0         None
"""
import numpy as np
import matplotlib.pyplot as plt
from gym_custom.envs.controller import orientation_error_quat_with_quat


def eval_robot(buffer, view_flag=True, save_fig=False, logger_path=None):
    i = 0
    plt.figure(i)
    plt.plot(buffer["xpos"])
    plt.plot(buffer["desired_xpos"])
    plt.legend(['x', 'y', 'z', 'dx', 'dy', 'dz'])
    plt.title('xpos')
    plt.grid()
    if save_fig:
        plt.savefig(logger_path + '/' + plt.gca().get_title())

    i += 1
    plt.figure(i)
    plt.plot((buffer["xpos"] - buffer["desired_xpos"]))
    plt.legend(['x', 'y', 'z'])
    plt.title('xpos error')
    plt.grid()
    if save_fig:
        plt.savefig(logger_path + '/' + plt.gca().get_title())

    r = np.sqrt(0.2 ** 2 + 0.565 ** 2)
    theta = np.arctan(0.565 / 0.2)
    table_position = np.array([0.5 - r * np.cos(theta - np.pi / 60), -0.2, r * np.sin(theta - np.pi / 60)])
    table_rotation = np.array([[0.9986295, 0, -0.0523360],
                               [0, 1, 0],
                               [0.0523360, 0, 0.9986295]])
    xpos_error_table_buffer = []
    for j in range(len(buffer["xquat"])):
        xpos_table = np.linalg.inv(table_rotation) @ (buffer["xpos"][j] - table_position)
        dpos_table = np.linalg.inv(table_rotation) @ (buffer["desired_xpos"][j] - table_position)
        xpos_error_table_buffer.append(xpos_table-dpos_table)
    i += 1
    plt.figure(i)
    plt.plot(xpos_error_table_buffer)
    plt.legend(['x', 'y', 'z', 'dx', 'dy', 'dz'])
    plt.title('xpos error table')
    plt.grid()
    if save_fig:
        plt.savefig(logger_path + '/' + plt.gca().get_title())

    # i += 1
    # plt.figure(i)
    # plt.plot(buffer["xvel"][:, :3])
    # plt.plot(buffer["desired_xvel"][:, :3])
    # plt.legend(['x', 'y', 'z', 'dx', 'dy', 'dz'])
    # plt.title('xpos_vel')
    # plt.grid()
    # if save_fig:
    #     plt.savefig(logger_path + '/' + plt.gca().get_title())
    #
    # i += 1
    # plt.figure(i)
    # plt.plot(np.diff(buffer["xvel"][:, :3], axis=0))
    # plt.plot(np.diff(buffer["desired_xvel"][:, :3], axis=0))
    # # plt.legend(['x', 'y', 'z', 'dx', 'dy', 'dz'])
    # plt.legend(['x', 'y', 'dx', 'dy'])
    # plt.title('xpos_acc')
    # plt.grid()
    # if save_fig:
    #     plt.savefig(logger_path + '/' + plt.gca().get_title())
    #
    # i += 1
    # plt.figure(i)
    # plt.plot(buffer["xquat"])
    # plt.plot(buffer["desired_xquat"])
    # plt.legend(['x', 'y', 'z', 'w', 'dx', 'dy', 'dz', 'dw'])
    # plt.title('xquat')
    # plt.grid()
    # if save_fig:
    #     plt.savefig(logger_path + '/' + plt.gca().get_title())
    #
    # i += 1
    # plt.figure(i)
    # plt.plot(buffer["xvel"][:, 3:])
    # plt.plot(buffer["desired_xvel"][:, 3:])
    # plt.legend(['x', 'y', 'z', 'dx', 'dy', 'dz'])
    # plt.title('xmat_vel')
    # plt.grid()
    # if save_fig:
    #     plt.savefig(logger_path + '/' + plt.gca().get_title())
    #
    # i += 1
    # plt.figure(i)
    # orientation_error_buffer = []
    # for j in range(len(buffer["xquat"])):
    #     orientation_error_buffer.append(
    #         orientation_error_quat_with_quat(buffer["desired_xquat"][j], buffer["xquat"][j]))
    # plt.plot(orientation_error_buffer)
    # plt.legend(['x', 'y', 'z'])
    # plt.title('orientation_error')
    # plt.grid()
    # if save_fig:
    #     plt.savefig(logger_path + '/' + plt.gca().get_title())
    #
    # i += 1
    # plt.figure(i)
    # plt.plot(buffer["qpos"])
    # plt.legend(['j1', 'j2', 'j3', 'j4', 'j5', 'j6'])
    # plt.title('qpos')
    # plt.grid()
    # if save_fig:
    #     plt.savefig(logger_path + '/' + plt.gca().get_title())
    #
    # i += 1
    # plt.figure(i)
    # plt.plot(buffer["qvel"])
    # plt.legend(['j1', 'j2', 'j3', 'j4', 'j5', 'j6'])
    # plt.title('qvel')
    # plt.grid()
    # if save_fig:
    #     plt.savefig(logger_path + '/' + plt.gca().get_title())

    delta_pos = np.array(buffer["xpos"]) - np.array(buffer["desired_xpos"])
    f = np.array(buffer["contact_force"])[:, :3]
    stiffness = np.zeros_like(delta_pos)
    for j in range(len(stiffness)):
        if not np.any(f[j, :] == 0):
            stiffness[j] = delta_pos[j, :] / f[j, :]
    # i += 1
    # plt.figure(i)
    # plt.plot(stiffness[3:, :])
    # plt.legend(['x', 'y', 'z'])
    # plt.title('1/stiffness')
    # plt.grid()
    # if save_fig:
    #     plt.savefig(logger_path + '/' + plt.gca().get_title())

    i += 1
    plt.figure(i)
    plt.plot(buffer["contact_force"])
    plt.legend(['x', 'y', 'z', 'rx', 'ry', 'rz'])
    plt.title('contact_force')
    plt.grid()
    if save_fig:
        plt.savefig(logger_path + '/' + plt.gca().get_title())

    # i += 1
    # plt.figure(i)
    # plt.plot(buffer["contact_force_l"])
    # plt.legend(['x', 'y', 'z', 'rx', 'ry', 'rz'])
    # plt.title('contact_force_l')
    # plt.grid()
    # if save_fig:
    #     plt.savefig(logger_path + '/' + plt.gca().get_title())

    r = np.sqrt(0.2 ** 2 + 0.565 ** 2)
    theta = np.arctan(0.565 / 0.2)
    table_position = np.array([0.5 - r * np.cos(theta - np.pi / 60), -0.2, r * np.sin(theta - np.pi / 60)])
    table_rotation = np.array([[0.9986295, 0, -0.0523360],
                               [0, 1, 0],
                               [0.0523360, 0, 0.9986295]])
    contact_force_table = np.empty_like(buffer["contact_force"])
    for k in range(contact_force_table.shape[0]):
        contact_force_table[k] = (np.linalg.inv(table_rotation) @ buffer["contact_force"][k].reshape((3, 2), order="F")).\
            reshape(-1, order="F")
    i += 1
    plt.figure(i)
    plt.plot(contact_force_table)
    plt.legend(['x', 'y', 'z', 'rx', 'ry', 'rz'])
    plt.title('contact force table')
    plt.grid()
    if save_fig:
        plt.savefig(logger_path + '/' + plt.gca().get_title())

    # i += 1
    # plt.figure(i)
    # plt.plot(buffer["touch_force"])
    # plt.legend(['j1', 'j2', 'j3', 'j4', 'j5', 'j6'])
    # plt.title('touch_force')
    # plt.grid()
    # if save_fig:
    #     plt.savefig(logger_path + '/' + plt.gca().get_title())
    #
    # i += 1
    # plt.figure(i)
    # plt.plot(buffer["tau"])
    # plt.legend(['j1', 'j2', 'j3', 'j4', 'j5', 'j6'])
    # plt.title('tau')
    # plt.grid()
    # if save_fig:
    #     plt.savefig(logger_path + '/' + plt.gca().get_title())

    if view_flag:
        plt.show()


def eval_everything(env, result_dict, view_flag=True, save_fig=False, logger_path=None):
    for _, (name, value) in enumerate(result_dict.items()):
        result_dict[name] = np.array(value)
    i = 0
    # plt.figure(i)
    # plt.plot(result_dict["xpos"])
    # plt.plot(result_dict["desired_xpos"])
    # plt.plot(env.init_desired_xposture_list[:, :3])
    # plt.legend(['x', 'y', 'z', 'dx', 'dy', 'dz', 'idx', 'idy', 'idz'])
    # plt.title('compare_xpos')
    # plt.grid()
    # if save_fig:
    #     plt.savefig(logger_path + '/' + plt.gca().get_title())
    #
    # i += 1
    # plt.figure(i)
    # plt.plot(result_dict["xpos"] - result_dict["desired_xpos"])
    # plt.legend(['x', 'y', 'z'])
    # plt.title('compare_xpos_error')
    # plt.grid()
    # if save_fig:
    #     plt.savefig(logger_path + '/' + plt.gca().get_title())

    table = np.array([[0.9986295, 0, -0.0523360],
                      [0, 1, 0],
                      [0.0523360, 0, 0.9986295]])
    xpos_error_table_buffer = []
    for j in range(len(result_dict["xquat"])):
        xpos_error_table_buffer.append(table.transpose() @ (result_dict["xpos"] - result_dict["desired_xpos"])[j])
    i += 1
    plt.figure(i)
    plt.plot(xpos_error_table_buffer)
    plt.legend(['x', 'y', 'z', 'dx', 'dy', 'dz'])
    plt.title('xpos error table')
    plt.grid()
    if save_fig:
        plt.savefig(logger_path + '/' + plt.gca().get_title())

    # i += 1
    # plt.figure(i)
    # plt.plot(result_dict["xvel"][:, :3])
    # plt.plot(result_dict["desired_xvel"][:, :3])
    # plt.legend(['x', 'y', 'z', 'dx', 'dy', 'dz'])
    # plt.title('compare_xpos_vel')
    # plt.grid()
    # if save_fig:
    #     plt.savefig(logger_path + '/' + plt.gca().get_title())
    #
    # i += 1
    # plt.figure(i)
    # plt.plot(result_dict["desired_xacc"][:, :3])
    # plt.legend(['dx', 'dy', 'dz'])
    # plt.title('compare_xpos_acc')
    # plt.grid()
    # if save_fig:
    #     plt.savefig(logger_path + '/' + plt.gca().get_title())
    #
    # i += 1
    # plt.figure(i)
    # plt.plot(result_dict["xquat"])
    # plt.plot(result_dict["desired_xquat"])
    # plt.plot(env.init_desired_xposture_list[:, 12:16])
    # plt.legend(['x', 'y', 'z', 'w', 'dx', 'dy', 'dz', 'idw', 'idx', 'idy', 'idz', 'idw'])
    # plt.title('compare_xquat')
    # plt.grid()
    # if save_fig:
    #     plt.savefig(logger_path + '/' + plt.gca().get_title())
    #
    # i += 1
    # plt.figure(i)
    # plt.plot(result_dict["xvel"][:, 3:])
    # plt.plot(result_dict["desired_xvel"][:, 3:])
    # plt.legend(['x', 'y', 'z', 'dx', 'dy', 'dz'])
    # plt.title('compare_xmat_vel')
    # plt.grid()
    # if save_fig:
    #     plt.savefig(logger_path + '/' + plt.gca().get_title())

    i += 1
    plt.figure(i)
    orientation_error_buffer = []
    for j in range(len(result_dict["xquat"])):
        orientation_error_buffer.append(
            orientation_error_quat_with_quat(result_dict["desired_xquat"][j], result_dict["xquat"][j]))
    plt.plot(orientation_error_buffer)
    plt.legend(['x', 'y', 'z'])
    plt.title('compare_orientation_error')
    plt.grid()
    if save_fig:
        plt.savefig(logger_path + '/' + plt.gca().get_title())

    # i += 1
    # plt.figure(i)
    # plt.plot(result_dict["K"])
    # plt.legend(['x', 'y', 'z', 'rx', 'ry', 'rz'])
    # plt.title('K')
    # plt.grid()
    # if save_fig:
    #     plt.savefig(logger_path + '/' + plt.gca().get_title())

    if 'delta_pos' in result_dict:
        i += 1
        plt.figure(i)
        plt.plot(result_dict["delta_pos"])
        plt.legend(['x', 'y', 'z'])
        plt.title('delta_pos')
        plt.grid()
        if save_fig:
            plt.savefig(logger_path + '/' + plt.gca().get_title())

        i += 1
        plt.figure(i)
        plt.plot(result_dict["delta_mat"].reshape((result_dict["delta_mat"].shape[0], -1)))
        plt.legend(['1', '2', '3', '4', '5', '6', '7', '8', '9'])
        plt.title('delta_mat')
        plt.grid()
        if save_fig:
            plt.savefig(logger_path + '/' + plt.gca().get_title())

        i += 1
        plt.figure(i)
        plt.plot(result_dict["delta_quat"])
        plt.legend(['x', 'y', 'z', 'w'])
        plt.title('delta_quat')
        plt.grid()
        if save_fig:
            plt.savefig(logger_path + '/' + plt.gca().get_title())

    i += 1
    plt.figure(i)
    plt.plot(result_dict["contact_force"])
    plt.legend(['x', 'y', 'z', 'rx', 'ry', 'rz'])
    plt.title('contact force')
    plt.grid()
    if save_fig:
        plt.savefig(logger_path + '/' + plt.gca().get_title())
    #
    # i += 1
    # plt.figure(i)
    # plt.plot(result_dict["contact_force_l"])
    # plt.legend(['x', 'y', 'z', 'rx', 'ry', 'rz'])
    # plt.title('contact force local')
    # plt.grid()
    # if save_fig:
    #     plt.savefig(logger_path + '/' + plt.gca().get_title())

    table = np.array([[0.9986295, 0, -0.0523360],
                      [0, 1, 0],
                      [0.0523360, 0, 0.9986295]])
    contact_force_table = np.empty_like(result_dict["contact_force"])
    for k in range(contact_force_table.shape[0]):
        contact_force_table[k] = (table.transpose() @ result_dict["contact_force"][k].reshape((3, 2), order="F")).\
            reshape(-1, order="F")
    i += 1
    plt.figure(i)
    plt.plot(contact_force_table)
    plt.legend(['x', 'y', 'z', 'rx', 'ry', 'rz'])
    plt.title('contact force table')
    plt.grid()
    if save_fig:
        plt.savefig(logger_path + '/' + plt.gca().get_title())

    # i += 1
    # plt.figure(i)
    # plt.plot(result_dict["desired_force"])
    # plt.legend(['x', 'y', 'z', 'rx', 'ry', 'rz'])
    # plt.title('desired force')
    # plt.grid()
    # if save_fig:
    #     plt.savefig(logger_path + '/' + plt.gca().get_title())

    # delta_pos = result_dict["xpos"] - result_dict["desired_xpos"]
    # f = result_dict["contact_force"][:, :3]
    # stiffness = np.zeros_like(delta_pos)
    # for j in range(len(stiffness)):
    #     if not np.any(delta_pos[j, :] == 0):
    #         stiffness[j] = f[j, :] / delta_pos[j, :]
    # i += 1
    # plt.figure(i)
    # plt.plot(stiffness[3:, :])
    # plt.legend(['x', 'y', 'z'])
    # plt.title('stiffness')
    # plt.grid()
    # if save_fig:
    #     plt.savefig(logger_path + '/' + plt.gca().get_title())

    # i += 1
    # plt.figure(i)
    # plt.plot(result_dict["qpos"])
    # plt.legend(['1', '2', '3', '4', '5', '6'])
    # plt.title('qpos')
    # plt.grid()
    # if save_fig:
    #     plt.savefig(logger_path + '/' + plt.gca().get_title())
    #
    # i += 1
    # plt.figure(i)
    # plt.plot(result_dict["qvel"])
    # plt.legend(['1', '2', '3', '4', '5', '6'])
    # plt.title('qvel')
    # plt.grid()
    # if save_fig:
    #     plt.savefig(logger_path + '/' + plt.gca().get_title())
    #
    # i += 1
    # plt.figure(i)
    # plt.plot(result_dict["tau"])
    # plt.legend(['1', '2', '3', '4', '5', '6'])
    # plt.title('tau')
    # plt.grid()
    # if save_fig:
    #     plt.savefig(logger_path + '/' + plt.gca().get_title())

    # i += 1
    # plt.figure(i)
    # plt.plot(result_dict["observation"][:, -1, :3])
    # plt.legend(['x', 'y', 'z'])
    # plt.title('observation xpos')
    # plt.grid()
    # if save_fig:
    #     plt.savefig(logger_path + '/' + plt.gca().get_title())

    # i += 1
    # plt.figure(i)
    # plt.plot(result_dict["observation"][:, -1, 3:7])
    # plt.legend(['x', 'y', 'z', 'w'])
    # plt.title('observation xquat')
    # plt.grid()
    # if save_fig:
    #     plt.savefig(logger_path + '/' + plt.gca().get_title())
    #
    # i += 1
    # plt.figure(i)
    # plt.plot(result_dict["observation"][:, -1, 7:10])
    # plt.legend(['x', 'y', 'z'])
    # plt.title('observation pvel')
    # plt.grid()
    # if save_fig:
    #     plt.savefig(logger_path + '/' + plt.gca().get_title())
    #
    # i += 1
    # plt.figure(i)
    # plt.plot(result_dict["observation"][:, -1, 10:13])
    # plt.legend(['x', 'y', 'z'])
    # plt.title('observation rvel')
    # plt.grid()
    # if save_fig:
    #     plt.savefig(logger_path + '/' + plt.gca().get_title())

    i += 1
    plt.figure(i)
    plt.plot(result_dict["action"][:, :6])
    plt.legend(['x', 'y', 'z', 'rx', 'ry', 'rz'])
    plt.title('action k')
    plt.grid()
    if save_fig:
        plt.savefig(logger_path + '/' + plt.gca().get_title())

    i += 1
    plt.figure(i)
    plt.plot(result_dict["action"][:, 6:9])
    plt.legend(['x', 'y', 'z'])
    plt.title('action xpos')
    plt.grid()
    if save_fig:
        plt.savefig(logger_path + '/' + plt.gca().get_title())

    i += 1
    plt.figure(i)
    plt.plot(result_dict["action"][:, 9:13])
    plt.legend(['x', 'y', 'z', 'w'])
    plt.title('action xquat')
    plt.grid()
    if save_fig:
        plt.savefig(logger_path + '/' + plt.gca().get_title())

    i += 1
    plt.figure(i)
    plt.plot(result_dict["action"][:, 13:17])
    plt.legend(['x', 'y', 'z', 'w'])
    plt.title('k quat')
    plt.grid()
    if save_fig:
        plt.savefig(logger_path + '/' + plt.gca().get_title())

    i += 1
    plt.figure(i)
    plt.plot(result_dict["reward"])
    plt.title('reward')
    plt.grid()
    if save_fig:
        plt.savefig(logger_path + '/' + plt.gca().get_title())

    if view_flag:
        plt.show()
