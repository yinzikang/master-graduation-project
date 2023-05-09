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


def eval_robot(task, result_dict, view_flag=True, save_fig=False, logger_path=None):
    i = 0
    plt.figure(i)
    plt.plot(result_dict["xpos"])
    plt.plot(result_dict["desired_xpos"])
    plt.legend(['x', 'y', 'z', 'dx', 'dy', 'dz'])
    plt.title('compare_xpos')
    plt.grid()
    if save_fig:
        plt.savefig(logger_path + '/' + plt.gca().get_title())

    i += 1
    plt.figure(i)
    plt.plot(result_dict["xpos"] - result_dict["desired_xpos"])
    plt.legend(['x', 'y', 'z'])
    plt.title('compare_xpos_error')
    plt.grid()
    if save_fig:
        plt.savefig(logger_path + '/' + plt.gca().get_title())

    if 'cabinet surface' in task or 'cabinet drawer' in task:
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

    if 'cabinet door' in task:
        cabinet_pos = np.array([0.8, -0.2, 0.3])
        r_bias = np.array([-0.025, 0.34, 0])
        angle_init = np.arctan(np.abs(r_bias[0] / r_bias[1]))
        radius = np.linalg.norm(r_bias)
        center = cabinet_pos + np.array([-0.2 + 0.0075, -0.19, 0.22])
        rbt_tool = np.array([-0.011, -0.004, 0])
        real_r_bias_buffer = []
        real_radius_buffer = []
        real_angle_buffer = []
        desired_r_bias_buffer = []
        desired_radius_buffer = []
        desired_angle_buffer = []
        for j in range(len(result_dict["xpos"])):
            real_r_bias = result_dict["xpos"][j] - rbt_tool - center
            real_radius_buffer.append(np.linalg.norm(real_r_bias))
            real_angle_buffer.append(np.arctan2(-real_r_bias[0],real_r_bias[1]))

            desired_r_bias = result_dict["desired_xpos"][j] - rbt_tool - center
            desired_radius_buffer.append(np.linalg.norm(desired_r_bias))
            desired_angle_buffer.append(np.arctan2(-desired_r_bias[0],desired_r_bias[1]))

        real_r_bias_buffer = np.array(real_r_bias_buffer)
        real_radius_buffer = np.array(real_radius_buffer)
        real_angle_buffer = np.array(real_angle_buffer)
        desired_r_bias_buffer = np.array(desired_r_bias_buffer)
        desired_radius_buffer = np.array(desired_radius_buffer)
        desired_angle_buffer = np.array(desired_angle_buffer)

        i += 1
        plt.figure(i)
        plt.plot(result_dict["xpos"][:, 0], result_dict["xpos"][:, 1])
        plt.plot(result_dict["desired_xpos"][:, 0], result_dict["desired_xpos"][:, 1])
        plt.legend(['real traj', 'desired traj'])
        plt.title('traj')
        plt.grid()
        plt.axis('equal')
        if save_fig:
            plt.savefig(logger_path + '/' + plt.gca().get_title())

        i += 1
        plt.figure(i)
        plt.plot(real_radius_buffer)
        plt.plot(desired_radius_buffer)
        plt.axhline(radius)
        plt.legend(['real r', 'desired r', 'door r'])
        plt.title('radius')
        plt.grid()
        if save_fig:
            plt.savefig(logger_path + '/' + plt.gca().get_title())

        i += 1
        plt.figure(i)
        plt.plot(real_radius_buffer - radius)
        plt.plot(desired_radius_buffer - radius)
        plt.legend(['real r error', 'desired r error'])
        plt.title('radius error')
        plt.grid()
        if save_fig:
            plt.savefig(logger_path + '/' + plt.gca().get_title())

        i += 1
        plt.figure(i)
        plt.plot(real_angle_buffer)
        plt.plot(desired_angle_buffer)
        plt.legend(['real angle', 'desired angle'])
        plt.title('robot angle')
        plt.grid()
        if save_fig:
            plt.savefig(logger_path + '/' + plt.gca().get_title())

        i += 1
        plt.figure(i)
        plt.plot(real_angle_buffer - angle_init)
        plt.plot(desired_angle_buffer - angle_init)
        plt.legend(['real angle', 'desired angle'])
        plt.title('door angle')
        plt.grid()
        if save_fig:
            plt.savefig(logger_path + '/' + plt.gca().get_title())

        i += 1
        plt.figure(i)
        plt.plot(real_angle_buffer - desired_angle_buffer)
        plt.legend(['angle error'])
        plt.title('angle error')
        plt.grid()
        if save_fig:
            plt.savefig(logger_path + '/' + plt.gca().get_title())

    i += 1
    plt.figure(i)
    plt.plot(result_dict["xvel"][:, :3])
    plt.plot(result_dict["desired_xvel"][:, :3])
    plt.legend(['x', 'y', 'z', 'dx', 'dy', 'dz'])
    plt.title('compare_xpos_vel')
    plt.grid()
    if save_fig:
        plt.savefig(logger_path + '/' + plt.gca().get_title())

    i += 1
    plt.figure(i)
    plt.plot(result_dict["desired_xacc"][:, :3])
    plt.legend(['dx', 'dy', 'dz'])
    plt.title('compare_xpos_acc')
    plt.grid()
    if save_fig:
        plt.savefig(logger_path + '/' + plt.gca().get_title())

    i += 1
    plt.figure(i)
    plt.plot(result_dict["xquat"])
    plt.plot(result_dict["desired_xquat"])
    plt.legend(['x', 'y', 'z', 'w', 'dx', 'dy', 'dz', 'dw'])
    plt.title('compare_xquat')
    plt.grid()
    if save_fig:
        plt.savefig(logger_path + '/' + plt.gca().get_title())

    i += 1
    plt.figure(i)
    plt.plot(result_dict["xvel"][:, 3:])
    plt.plot(result_dict["desired_xvel"][:, 3:])
    plt.legend(['x', 'y', 'z', 'dx', 'dy', 'dz'])
    plt.title('compare_xmat_vel')
    plt.grid()
    if save_fig:
        plt.savefig(logger_path + '/' + plt.gca().get_title())

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

    i += 1
    plt.figure(i)
    plt.plot(result_dict["contact_force"])
    plt.legend(['x', 'y', 'z', 'rx', 'ry', 'rz'])
    plt.title('contact force')
    plt.grid()
    if save_fig:
        plt.savefig(logger_path + '/' + plt.gca().get_title())

    i += 1
    plt.figure(i)
    plt.plot(result_dict["desired_force"])
    plt.legend(['x', 'y', 'z', 'rx', 'ry', 'rz'])
    plt.title('desired force')
    plt.grid()
    if save_fig:
        plt.savefig(logger_path + '/' + plt.gca().get_title())

    i += 1
    plt.figure(i)
    plt.plot(result_dict["contact_force"] - result_dict["desired_force"])
    plt.legend(['x', 'y', 'z', 'rx', 'ry', 'rz'])
    plt.title('force error')
    plt.grid()
    if save_fig:
        plt.savefig(logger_path + '/' + plt.gca().get_title())

    i += 1
    plt.figure(i)
    plt.plot(result_dict["contact_force_l"])
    plt.legend(['x', 'y', 'z', 'rx', 'ry', 'rz'])
    plt.title('contact force local')
    plt.grid()
    if save_fig:
        plt.savefig(logger_path + '/' + plt.gca().get_title())

    if 'cabinet surface' in task or 'cabinet drawer' in task:
        table = np.array([[0.9986295, 0, -0.0523360],
                          [0, 1, 0],
                          [0.0523360, 0, 0.9986295]])
        contact_force_table = np.empty_like(result_dict["contact_force"])
        desired_force_table = np.empty_like(result_dict["desired_force"])
        force_error_table = np.empty_like(result_dict["desired_force"])
        for k in range(contact_force_table.shape[0]):
            contact_force_table[k] = (table.transpose() @ result_dict["contact_force"][k].reshape((3, 2), order="F")). \
                reshape(-1, order="F")
            desired_force_table[k] = (table.transpose() @ result_dict["desired_force"][k].reshape((3, 2), order="F")). \
                reshape(-1, order="F")
            force_error_table[k] = contact_force_table[k] - desired_force_table[k]

        i += 1
        plt.figure(i)
        plt.plot(contact_force_table)
        plt.legend(['x', 'y', 'z', 'rx', 'ry', 'rz'])
        plt.title('contact force table')
        plt.grid()
        if save_fig:
            plt.savefig(logger_path + '/' + plt.gca().get_title())

        i += 1
        plt.figure(i)
        plt.plot(desired_force_table)
        plt.legend(['x', 'y', 'z', 'rx', 'ry', 'rz'])
        plt.title('desired force table')
        plt.grid()
        if save_fig:
            plt.savefig(logger_path + '/' + plt.gca().get_title())

        i += 1
        plt.figure(i)
        plt.plot(force_error_table)
        plt.legend(['x', 'y', 'z', 'rx', 'ry', 'rz'])
        plt.title('force error table')
        plt.grid()
        if save_fig:
            plt.savefig(logger_path + '/' + plt.gca().get_title())

    if 'cabinet door' in task:
        cabinet_pos = np.array([0.8, -0.2, 0.3])
        r_bias = np.array([-0.025, 0.34, 0])
        angle_init = np.arctan(np.abs(r_bias[0] / r_bias[1]))
        radius = np.linalg.norm(r_bias)
        center = cabinet_pos + np.array([-0.2 + 0.0075, -0.19, 0.22])
        rbt_tool = np.array([-0.011, -0.004, 0])
        real_force_door_buffer = []
        for j in range(len(result_dict["xpos"])):
            real_r_bias = result_dict["xpos"][j] - rbt_tool - center
            real_angle = np.arctan2(-real_r_bias[0], real_r_bias[1])
            c = np.cos(np.pi / 2 - real_angle)
            s = np.sin(np.pi / 2 - real_angle)
            real_rotation = np.array([[c, s, 0],
                                      [-s, c, 0],
                                      [0, 0, 1]])
            real_force_door_buffer.append((real_rotation.transpose() @
                                           result_dict['contact_force'][j].reshape((3, 2),
                                                                                   order="F")).reshape(-1, order="F"))
        real_force_door_buffer = np.array(real_force_door_buffer)

        i += 1
        plt.figure(i)
        plt.plot(real_force_door_buffer)
        plt.legend(['x', 'y', 'z', 'rx', 'ry', 'rz'])
        plt.title('force door')
        plt.grid()
        if save_fig:
            plt.savefig(logger_path + '/' + plt.gca().get_title())

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

    plt.show()


def eval_everything(env, result_dict, view_flag=True, save_fig=False, logger_path=None):
    for _, (name, value) in enumerate(result_dict.items()):
        result_dict[name] = np.array(value)
    i = 0
    plt.figure(i)
    plt.plot(result_dict["xpos"])
    plt.plot(result_dict["desired_xpos"])
    plt.plot(env.init_desired_xposture_list[:, :3])
    plt.legend(['x', 'y', 'z', 'dx', 'dy', 'dz', 'idx', 'idy', 'idz'])
    plt.title('compare_xpos')
    plt.grid()
    if save_fig:
        plt.savefig(logger_path + '/' + plt.gca().get_title())

    i += 1
    plt.figure(i)
    plt.plot(result_dict["xpos"] - result_dict["desired_xpos"])
    plt.legend(['x', 'y', 'z'])
    plt.title('compare_xpos_error')
    plt.grid()
    if save_fig:
        plt.savefig(logger_path + '/' + plt.gca().get_title())

    if 'cabinet surface' in env.task or 'cabinet drawer' in env.task:
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

    if 'cabinet door' in env.task:
        cabinet_pos = np.array([0.8, -0.2, 0.3])
        r_bias = np.array([-0.025, 0.34, 0])
        angle_init = np.arctan(np.abs(r_bias[0] / r_bias[1]))
        radius = np.linalg.norm(r_bias)
        center = cabinet_pos + np.array([-0.2 + 0.0075, -0.19, 0.22])
        rbt_tool = np.array([-0.011, -0.004, 0])
        real_radius_buffer = []
        real_angle_buffer = []
        desired_radius_buffer = []
        desired_angle_buffer = []
        for j in range(len(result_dict["xpos"])):
            real_r_bias = result_dict["xpos"][j] - rbt_tool - center
            real_radius_buffer.append(np.linalg.norm(real_r_bias))
            real_angle_buffer.append(np.arctan2(-real_r_bias[0],real_r_bias[1]))

            desired_r_bias = result_dict["desired_xpos"][j] - rbt_tool - center
            desired_radius_buffer.append(np.linalg.norm(desired_r_bias))
            desired_angle_buffer.append(np.arctan2(-desired_r_bias[0],desired_r_bias[1]))

        real_radius_buffer = np.array(real_radius_buffer)
        real_angle_buffer = np.array(real_angle_buffer)
        desired_radius_buffer = np.array(desired_radius_buffer)
        desired_angle_buffer = np.array(desired_angle_buffer)

        perfect_door_angle = np.linspace(0, np.pi / 2, 1999) + angle_init
        door_pos = np.concatenate((-radius * np.sin(perfect_door_angle).reshape(-1, 1),
                                   radius * np.cos(perfect_door_angle).reshape(-1, 1),
                                   np.zeros_like(perfect_door_angle).reshape(-1, 1)), axis=1) + center.reshape(1, 3)

        i += 1
        plt.figure(i)
        plt.plot(result_dict["xpos"][:, 0] - rbt_tool[0], result_dict["xpos"][:, 1] - rbt_tool[1])
        plt.plot(result_dict["desired_xpos"][:, 0] - rbt_tool[0], result_dict["desired_xpos"][:, 1] - rbt_tool[1])
        plt.plot(door_pos[:, 0], door_pos[:, 1])
        plt.plot([result_dict["xpos"][0, 0] - rbt_tool[0], center[0]],
                 [result_dict["xpos"][0, 1] - rbt_tool[1], center[1]])
        plt.plot([result_dict["xpos"][-1, 0] - rbt_tool[0], center[0]],
                 [result_dict["xpos"][-1, 1] - rbt_tool[1], center[1]])
        plt.plot([result_dict["desired_xpos"][0, 0] - rbt_tool[0], center[0]],
                 [result_dict["desired_xpos"][0, 1] - rbt_tool[1], center[1]])
        plt.plot([result_dict["desired_xpos"][-1, 0] - rbt_tool[0], center[0]],
                 [result_dict["desired_xpos"][-1, 1] - rbt_tool[1], center[1]])
        plt.plot([door_pos[0, 0], center[0]], [door_pos[0, 1], center[1]])
        plt.plot([door_pos[-1, 0], center[0]], [door_pos[-1, 1], center[1]])

        plt.legend(['real traj', 'desired traj', 'door traj'])
        plt.title('traj')  # 机器人夹子中心
        plt.grid()
        plt.axis('equal')
        if save_fig:
            plt.savefig(logger_path + '/' + plt.gca().get_title())

        i += 1
        plt.figure(i)
        plt.plot(real_radius_buffer)
        plt.plot(desired_radius_buffer)
        plt.axhline(radius)
        plt.legend(['real r', 'desired r', 'door r'])
        plt.title('radius')
        plt.grid()
        if save_fig:
            plt.savefig(logger_path + '/' + plt.gca().get_title())

        i += 1
        plt.figure(i)
        plt.plot(real_radius_buffer - radius)
        plt.plot(desired_radius_buffer - radius)
        plt.legend(['real r error', 'desired r error'])
        plt.title('radius error')
        plt.grid()
        if save_fig:
            plt.savefig(logger_path + '/' + plt.gca().get_title())

        i += 1
        plt.figure(i)
        plt.plot(real_angle_buffer)
        plt.plot(desired_angle_buffer)
        plt.legend(['real angle', 'desired angle'])
        plt.title('robot angle')
        plt.grid()
        if save_fig:
            plt.savefig(logger_path + '/' + plt.gca().get_title())

        i += 1
        plt.figure(i)
        plt.plot(real_angle_buffer - angle_init)
        plt.plot(desired_angle_buffer - angle_init)
        plt.legend(['real angle', 'desired angle'])
        plt.title('door angle')
        plt.grid()
        if save_fig:
            plt.savefig(logger_path + '/' + plt.gca().get_title())

        i += 1
        plt.figure(i)
        plt.plot(real_angle_buffer - desired_angle_buffer)
        plt.legend(['angle error'])
        plt.title('angle error')
        plt.grid()
        if save_fig:
            plt.savefig(logger_path + '/' + plt.gca().get_title())

    i += 1
    plt.figure(i)
    plt.plot(result_dict["xvel"][:, :3])
    plt.plot(result_dict["desired_xvel"][:, :3])
    plt.legend(['x', 'y', 'z', 'dx', 'dy', 'dz'])
    plt.title('compare_xpos_vel')
    plt.grid()
    if save_fig:
        plt.savefig(logger_path + '/' + plt.gca().get_title())

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
    #
    # i += 1
    # plt.figure(i)
    # orientation_error_buffer = []
    # for j in range(len(result_dict["xquat"])):
    #     orientation_error_buffer.append(
    #         orientation_error_quat_with_quat(result_dict["desired_xquat"][j], result_dict["xquat"][j]))
    # plt.plot(orientation_error_buffer)
    # plt.legend(['x', 'y', 'z'])
    # plt.title('compare_orientation_error')
    # plt.grid()
    # if save_fig:
    #     plt.savefig(logger_path + '/' + plt.gca().get_title())

    i += 1
    plt.figure(i)
    plt.plot(result_dict["K"])
    plt.legend(['x', 'y', 'z', 'rx', 'ry', 'rz'])
    plt.title('K')
    plt.grid()
    if save_fig:
        plt.savefig(logger_path + '/' + plt.gca().get_title())

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

    # i += 1
    # plt.figure(i)
    # plt.plot(result_dict["desired_force"])
    # plt.legend(['x', 'y', 'z', 'rx', 'ry', 'rz'])
    # plt.title('desired force')
    # plt.grid()
    # if save_fig:
    #     plt.savefig(logger_path + '/' + plt.gca().get_title())
    #
    # i += 1
    # plt.figure(i)
    # plt.plot(result_dict["contact_force"] - result_dict["desired_force"])
    # plt.legend(['x', 'y', 'z', 'rx', 'ry', 'rz'])
    # plt.title('force error')
    # plt.grid()
    # if save_fig:
    #     plt.savefig(logger_path + '/' + plt.gca().get_title())
    #
    # i += 1
    # plt.figure(i)
    # plt.plot(result_dict["contact_force_l"])
    # plt.legend(['x', 'y', 'z', 'rx', 'ry', 'rz'])
    # plt.title('contact force local')
    # plt.grid()
    # if save_fig:
    #     plt.savefig(logger_path + '/' + plt.gca().get_title())

    if 'cabinet surface' in env.task or 'cabinet drawer' in env.task:
        table = np.array([[0.9986295, 0, -0.0523360],
                          [0, 1, 0],
                          [0.0523360, 0, 0.9986295]])
        contact_force_table = np.empty_like(result_dict["contact_force"])
        desired_force_table = np.empty_like(result_dict["desired_force"])
        force_error_table = np.empty_like(result_dict["desired_force"])
        for k in range(contact_force_table.shape[0]):
            contact_force_table[k] = (table.transpose() @ result_dict["contact_force"][k].reshape((3, 2), order="F")). \
                reshape(-1, order="F")
            desired_force_table[k] = (table.transpose() @ result_dict["desired_force"][k].reshape((3, 2), order="F")). \
                reshape(-1, order="F")
            force_error_table[k] = contact_force_table[k] - desired_force_table[k]

        i += 1
        plt.figure(i)
        plt.plot(contact_force_table)
        plt.legend(['x', 'y', 'z', 'rx', 'ry', 'rz'])
        plt.title('contact force table')
        plt.grid()
        if save_fig:
            plt.savefig(logger_path + '/' + plt.gca().get_title())

        i += 1
        plt.figure(i)
        plt.plot(desired_force_table)
        plt.legend(['x', 'y', 'z', 'rx', 'ry', 'rz'])
        plt.title('desired force table')
        plt.grid()
        if save_fig:
            plt.savefig(logger_path + '/' + plt.gca().get_title())

        i += 1
        plt.figure(i)
        plt.plot(force_error_table)
        plt.legend(['x', 'y', 'z', 'rx', 'ry', 'rz'])
        plt.title('force error table')
        plt.grid()
        if save_fig:
            plt.savefig(logger_path + '/' + plt.gca().get_title())

    if 'cabinet door' in env.task:
        cabinet_pos = np.array([0.8, -0.2, 0.3])
        r_bias = np.array([-0.025, 0.34, 0])
        angle_init = np.arctan(np.abs(r_bias[0] / r_bias[1]))
        radius = np.linalg.norm(r_bias)
        center = cabinet_pos + np.array([-0.2 + 0.0075, -0.19, 0.22])
        rbt_tool = np.array([-0.011, -0.004, 0])
        real_force_door_buffer = []
        for j in range(len(result_dict["xpos"])):
            robot_r_bias = result_dict["xpos"][j] - rbt_tool - center
            robot_angle = np.arctan2(-robot_r_bias[0], robot_r_bias[1])
            c = np.cos(robot_angle - np.pi / 2)
            s = np.sin(robot_angle - np.pi / 2)
            real_rotation = np.array([[c, -s, 0],
                                      [s, c, 0],
                                      [0, 0, 1]])
            real_force_door_buffer.append((real_rotation.transpose() @
                                           result_dict['contact_force'][j].reshape((3, 2),
                                                                                   order="F")).reshape(-1, order="F"))
        real_force_door_buffer = np.array(real_force_door_buffer)

        i += 1
        plt.figure(i)
        plt.plot(real_force_door_buffer)
        plt.legend(['x', 'y', 'z', 'rx', 'ry', 'rz'])
        plt.title('force door')
        plt.grid()
        if save_fig:
            plt.savefig(logger_path + '/' + plt.gca().get_title())

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

    i += 1
    plt.figure(i)
    plt.plot(result_dict["observation"][:, -1, :3])
    plt.legend(['x', 'y', 'z'])
    plt.title('observation xpos')
    plt.grid()
    if save_fig:
        plt.savefig(logger_path + '/' + plt.gca().get_title())

    i += 1
    plt.figure(i)
    plt.plot(result_dict["observation"][:, -1, 3:7])
    plt.legend(['x', 'y', 'z', 'w'])
    plt.title('observation xquat')
    plt.grid()
    if save_fig:
        plt.savefig(logger_path + '/' + plt.gca().get_title())

    i += 1
    plt.figure(i)
    plt.plot(result_dict["observation"][:, -1, 7:10])
    plt.legend(['x', 'y', 'z'])
    plt.title('observation pvel')
    plt.grid()
    if save_fig:
        plt.savefig(logger_path + '/' + plt.gca().get_title())

    i += 1
    plt.figure(i)
    plt.plot(result_dict["observation"][:, -1, 10:13])
    plt.legend(['x', 'y', 'z'])
    plt.title('observation rvel')
    plt.grid()
    if save_fig:
        plt.savefig(logger_path + '/' + plt.gca().get_title())

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
