#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""所有任务的参数，返回，保存，加载

Write detailed description here

Write typical usage example here

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
3/23/23 2:52 PM   yinzikang      1.0         None
"""
import numpy as np
import copy
import json
import os
import os.path as osp
from gym_custom.envs.controller import trajectory_planning_line, trajectory_planning_circle, \
    ComputedTorqueController, ImpedanceController, AdmittanceController, ImpedanceController_v2, \
    AdmittanceController_v2, AdmittanceController_v4
from gym_custom.envs.transformations import quaternion_matrix, quaternion_slerp, quaternion_from_matrix


def env_kwargs(task=None, save_flag=False, save_path=None):
    """
    不同的任务给定不同参数
    期望轨迹是用末端连杆（力传感器安装位置结合逆运动学算出来的）
    当前可用的只有带plan的
    cabinet surface1
    :param task:
    :param save_flag:
    :param save_path:
    :return:
    """
    rbt_kwargs = rbt_controller_kwargs = rl_env_kwargs = None
    if task == 'fig_plot':
        # 机器人参数
        mjc_model_path = 'robot/jk5_fig_plot.xml'
        qpos_init_list = np.array([0, 0, -90, 0, 90, 0]) / 180 * np.pi
        p_bias = np.array([0, 0, 0])
        r_bias = np.eye(3)
        rbt_kwargs = dict(mjc_model_path=mjc_model_path, task=task, qpos_init_list=qpos_init_list,
                          p_bias=p_bias, r_bias=r_bias)

        # 用于Jk5StickStiffnessEnv的超参数
        robot_frequency = 500
        time_whole = 4
        time_acceleration = 0
        step_num = robot_frequency * time_whole
        # 期望轨迹
        xpos_init, xmat_init = np.array([0.4965, -0.1135, 0.3040]), np.array([0, 1, 0, 1, 0, 0, 0, 0, -1])
        xpos_end, xmat_end = np.array([0.4965, -0.1135, 0.3040]), np.array([0, 1, 0, 1, 0, 0, 0, 0, -1])
        desired_xposture_list, desired_xvel_list, desired_xacc_list = trajectory_planning_line(xpos_init, xmat_init,
                                                                                               xpos_end, xmat_end,
                                                                                               time_whole,
                                                                                               time_acceleration,
                                                                                               robot_frequency)
        desired_force_list = np.array([[0, 0, 30, 0, 0, 0]], dtype=np.float64).repeat(step_num, axis=0)
        # 阻抗参数
        wn = 20
        damping_ratio = np.sqrt(2)
        K = np.array([300, 100, 100, 300, 100, 100], dtype=np.float64)
        M = K / (wn * wn)
        B = 2 * damping_ratio * np.sqrt(M * K)
        SM = np.eye(3, dtype=np.float64)
        # SM = np.array([[0.9986295, 0, -0.0523360],
        #                   [0, 1, 0],
        #                   [0.0523360, 0, 0.9986295]])
        controller_parameter = {'M': M, 'B': B, 'K': K, 'SM': SM}
        controller = AdmittanceController_v2
        # wn = 20
        # damping_ratio = np.sqrt(2)
        # kp = wn * wn * np.ones(6, dtype=np.float64)
        # kd = 2 * damping_ratio * np.sqrt(kp)
        # controller_parameter = {'kp': kp, 'kd': kd}
        # controller = ComputedTorqueController

        rbt_controller_kwargs = copy.deepcopy(rbt_kwargs)
        rbt_controller_kwargs.update(controller_parameter=controller_parameter, controller=controller,
                                     step_num=step_num,
                                     desired_xposture_list=desired_xposture_list, desired_xvel_list=desired_xvel_list,
                                     desired_xacc_list=desired_xacc_list, desired_force_list=desired_force_list)

        # 用于TrainEnv的超参数
        min_K = np.array([50, 50, 50, 50, 50, 50], dtype=np.float64)
        max_K = np.array([5000, 5000, 5000, 5000, 5000, 5000], dtype=np.float64)
        max_force = np.array([50, 50, 50, 50, 50, 50], dtype=np.float64)
        # [0.35, -0.1135, 0.565, 0., 1., 0., 1., 0., 0., 0., 0., -1., 0.70710678, 0.70710678, 0., 0.]
        min_desired_xposture = np.array([0.35, -0.1235, 0.56, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1])
        # [0.65, -0.1135, 0.565, 0., 1., 0., 1., 0., 0., 0., 0., -1., 0.70710678, 0.70710678, 0., 0.]
        max_desired_xposture = np.array([0.65, -0.1035, 0.6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        min_desired_xvel = -0.15 * np.ones(6)
        max_desired_xvel = 0.15 * np.ones(6)
        min_desired_xacc = desired_xacc_list.min() * np.ones(6)
        max_desired_xacc = desired_xacc_list.max() * np.ones(6)
        rl_frequency = 20  # 500可以整除，越大越多
        observation_range = 3

        rl_env_kwargs = copy.deepcopy(rbt_controller_kwargs)
        rl_env_kwargs.update(min_K=min_K, max_K=max_K, max_force=max_force,
                             min_desired_xposture=min_desired_xposture, max_desired_xposture=max_desired_xposture,
                             min_desired_xvel=min_desired_xvel, max_desired_xvel=max_desired_xvel,
                             min_desired_xacc=min_desired_xacc, max_desired_xacc=max_desired_xacc,
                             rl_frequency=rl_frequency, observation_range=observation_range)
    elif task == 'desk':
        # 实验内容
        mjc_model_path = 'robot/jk5_table_v2.xml'
        qpos_init_list = np.array([0, -30, 120, 0, -90, 0]) / 180 * np.pi
        p_bias = np.zeros(3)
        r_bias = np.eye(3)
        rl_frequency = 20  # 500可以整除，越大越多
        observation_range = 1
        step_num = 2000
        # 期望轨迹
        desired_xpos_list = np.concatenate((np.linspace(-0.45, -0.75, step_num).reshape(step_num, 1),
                                            -0.1135 * np.ones((step_num, 1), dtype=float),
                                            0.05 * np.ones((step_num, 1), dtype=float)), axis=1)
        desired_mat_list = np.array([[0, -1, 0, -1, 0, 0, 0, 0, -1]], dtype=np.float64).repeat(step_num, axis=0)
        desired_xposture_list = np.concatenate((desired_xpos_list, desired_mat_list), axis=1)
        desired_xvel_list = np.array([[-0.3 / step_num, 0, 0, 0, 0, 0]], dtype=np.float64).repeat(step_num, axis=0)
        desired_xacc_list = np.array([[0, 0, 0, 0, 0, 0]], dtype=np.float64).repeat(step_num, axis=0)
        desired_force_list = np.array([[0, 0, 30, 0, 0, 0]], dtype=np.float64).repeat(step_num, axis=0)
        # desired_xpos_list = np.concatenate((np.linspace(-0.5, -0.5, step_num).reshape(step_num, 1),
        #                                     -0.1135 * np.ones((step_num, 1), dtype=float),
        #                                     0.05 * np.ones((step_num, 1), dtype=float)), axis=1)
        # desired_mat_list = np.array([[0, -1, 0, -1, 0, 0, 0, 0, -1]], dtype=np.float64).repeat(step_num, axis=0)
        # desired_xposture_list = np.concatenate((desired_xpos_list, desired_mat_list), axis=1)
        # desired_xvel_list = np.array([[-0. / step_num, 0, 0, 0, 0, 0]], dtype=np.float64).repeat(step_num, axis=0)
        # desired_xacc_list = np.array([[0, 0, 0, 0, 0, 0]], dtype=np.float64).repeat(step_num, axis=0)
        # desired_force_list = np.array([[0, 0, 30, 0, 0, 0]], dtype=np.float64).repeat(step_num, axis=0)
        # 阻抗参数
        wn = 20
        damping_ratio = np.sqrt(2)
        K = np.array([1000, 1000, 1000, 1000, 1000, 1000], dtype=np.float64)
        M = K / (wn * wn)
        M_matrix = np.diag(M)
        B = 2 * damping_ratio * np.sqrt(M * K)
        controller_parameter = {'M': M_matrix, 'B': B, 'K': K}
        controller = ImpedanceController
        min_K = np.array([100, 100, 100, 100, 100, 100], dtype=np.float64)
        max_K = np.array([3000, 3000, 3000, 3000, 3000, 3000], dtype=np.float64)
        rbt_kwargs = dict(mjc_model_path=mjc_model_path, task=task, qpos_init_list=qpos_init_list,
                          p_bias=p_bias, r_bias=r_bias)
        # 用于Jk5StickStiffnessEnv的超参数
        rbt_controller_kwargs = copy.deepcopy(rbt_kwargs)
        rbt_controller_kwargs.update(controller_parameter=controller_parameter, controller=controller,
                                     step_num=step_num,
                                     desired_xposture_list=desired_xposture_list, desired_xvel_list=desired_xvel_list,
                                     desired_xacc_list=desired_xacc_list, desired_force_list=desired_force_list)

        # 用于TrainEnv的超参数
        rl_env_kwargs = copy.deepcopy(rbt_controller_kwargs)
        rl_env_kwargs.update(min_K=min_K, max_K=max_K,
                             rl_frequency=rl_frequency, observation_range=observation_range)

        return rbt_kwargs, rbt_controller_kwargs, rl_env_kwargs
    elif task == 'desk with plan':
        # 实验内容
        mjc_model_path = 'robot/jk5_table_v2.xml'
        qpos_init_list = np.array([0, 30, -120, 0, 90, 0]) / 180 * np.pi
        p_bias = np.zeros(3)
        r_bias = np.eye(3)
        rbt_kwargs = dict(mjc_model_path=mjc_model_path, task=task, qpos_init_list=qpos_init_list,
                          p_bias=p_bias, r_bias=r_bias)

        # 用于Jk5StickStiffnessEnv的超参数
        robot_frequency = 500
        time_whole = 4
        time_acceleration = 0.5
        step_num = robot_frequency * time_whole
        # 期望轨迹
        # xpos_init, xmat_init = np.array([0.45, -0.1135, 0.05]), np.array([0, 1, 0, 1, 0, 0, 0, 0, -1])
        xpos_init, xmat_init = np.array([0.6 - 0.2 * np.sqrt(2) + 0.1, -0.1135, 0.1]), \
            np.array([0, np.sqrt(2) / 2, np.sqrt(2) / 2,
                      1, 0, 0,
                      0, np.sqrt(2) / 2, -np.sqrt(2) / 2])
        # xpos_end, xmat_end = np.array([ 0.75, -0.1135, 0.05]), np.array([0, 1, 0, 1, 0, 0, 0, 0, -1])
        xpos_end, xmat_end = np.array([0.6 + 0.2 * np.sqrt(2) - 0.1, -0.1135, 0.4 * np.sqrt(2) - 0.1]), \
            np.array([0, np.sqrt(2) / 2, np.sqrt(2) / 2,
                      1, 0, 0,
                      0, np.sqrt(2) / 2, -np.sqrt(2) / 2])
        desired_xposture_list, desired_xvel_list, desired_xacc_list = trajectory_planning_line(xpos_init, xmat_init,
                                                                                               xpos_end, xmat_end,
                                                                                               time_whole,
                                                                                               time_acceleration,
                                                                                               robot_frequency)
        desired_force_list = np.array([[-5 * np.sqrt(2), 0, 5 * np.sqrt(2), 0, 0, 0]],
                                      dtype=np.float64).repeat(step_num, axis=0)
        # 阻抗参数
        wn = 20
        damping_ratio = np.sqrt(2)
        K = np.array([1000, 1000, 1000, 1000, 1000, 1000], dtype=np.float64)
        M = K / (wn * wn)
        B = 2 * damping_ratio * np.sqrt(M * K)
        SM = np.eye(3, dtype=np.float64)
        # SM = np.array([[np.sqrt(2) / 2, 0, -np.sqrt(2) / 2],
        #                [0, 1, 0],
        #                [np.sqrt(2) / 2, 0, np.sqrt(2) / 2]])
        controller_parameter = {'M': M, 'B': B, 'K': K, 'SM': SM}
        controller = AdmittanceController_v4

        rbt_controller_kwargs = copy.deepcopy(rbt_kwargs)
        rbt_controller_kwargs.update(controller_parameter=controller_parameter, controller=controller,
                                     step_num=step_num,
                                     desired_xposture_list=desired_xposture_list, desired_xvel_list=desired_xvel_list,
                                     desired_xacc_list=desired_xacc_list, desired_force_list=desired_force_list)

        # 用于TrainEnv的超参数
        min_K = np.array([100, 100, 100, 100, 100, 100], dtype=np.float64)
        max_K = np.array([3000, 3000, 3000, 3000, 3000, 3000], dtype=np.float64)
        max_force = np.array([50, 50, 50, 50, 50, 50], dtype=np.float64)
        # [0.35, -0.1135, 0.565, 0., 1., 0., 1., 0., 0., 0., 0., -1., 0.70710678, 0.70710678, 0., 0.]
        min_desired_xposture = np.array([0.35, -0.1235, 0.56, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1])
        # [0.65, -0.1135, 0.565, 0., 1., 0., 1., 0., 0., 0., 0., -1., 0.70710678, 0.70710678, 0., 0.]
        max_desired_xposture = np.array([0.65, -0.1035, 0.6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        min_desired_xvel = -0.15 * np.ones(6)
        max_desired_xvel = 0.15 * np.ones(6)
        min_desired_xacc = desired_xacc_list.min() * np.ones(6)
        max_desired_xacc = desired_xacc_list.max() * np.ones(6)
        rl_frequency = 20  # 500可以整除，越大越多
        observation_range = 3

        rl_env_kwargs = copy.deepcopy(rbt_controller_kwargs)
        rl_env_kwargs.update(min_K=min_K, max_K=max_K, max_force=max_force,
                             min_desired_xposture=min_desired_xposture, max_desired_xposture=max_desired_xposture,
                             min_desired_xvel=min_desired_xvel, max_desired_xvel=max_desired_xvel,
                             min_desired_xacc=min_desired_xacc, max_desired_xacc=max_desired_xacc,
                             rl_frequency=rl_frequency, observation_range=observation_range)

    elif task == 'open door':
        # 实验内容
        mjc_model_path = 'robot/jk5_opendoor.xml'
        xpos_init_list = np.array([0.003, -0.81164083, 0.275])
        qpos_init_list = np.array([0, 60, 60, -30, -90, 0]) / 180 * np.pi
        p_bias = np.array([0, 0, 0.3885])
        r_bias = quaternion_matrix([0.5, 0.5, 0.5, 0.5])[:3, :3]
        rl_frequency = 250
        observation_range = 1
        step_num = 2000
        # 期望轨迹
        center_pred = np.array([-0.03, -0.4, 0.275])
        handle_pred = np.abs(xpos_init_list[0] - center_pred[0])  # 把手长度预测
        door_pred = np.abs(xpos_init_list[1] - center_pred[1])  # 门板长度预测
        radius_pred = np.linalg.norm(xpos_init_list[:2] - center_pred[:2])  # 半径预测
        angle_pred = np.arctan(handle_pred / door_pred)  # 角度预测
        total_angle = np.pi / 2  # 开门角度
        rot_matrix_init = np.eye(4)
        rot_matrix_init[:3, :3] = np.array([[0, 0, -1], [0, -1, 0], [-1, 0, 0]])  # 初始旋转矩阵
        quat_init = quaternion_from_matrix(rot_matrix_init)  # 初始四元数
        rot_matrix_last = np.eye(4)
        rot_matrix_last[:3, :3] = np.array([[0, 1, 0], [0, 0, -1], [-1, 0, -0]])  # 最终旋转矩阵
        quat_last = quaternion_from_matrix(rot_matrix_last)  # 最终四元数
        # 轨迹规划
        desired_xpos_list = np.empty((step_num, 3))  # xpos traj
        desired_mat_list = np.empty((step_num, 9))  # xmat traj
        for i in range(step_num):
            desired_xpos_list[i][0] = center_pred[0] + \
                                      radius_pred * np.sin(angle_pred + total_angle * i / (step_num - 1))
            desired_xpos_list[i][1] = center_pred[1] \
                                      - radius_pred * np.cos(angle_pred + total_angle * i / (step_num - 1))
            desired_xpos_list[i][2] = 0.275
            desired_mat_list[i] = quaternion_matrix(quaternion_slerp(quat_init, quat_last, i / (step_num - 1)))[:3,
                                  :3].reshape(
                -1)

        desired_xposture_list = np.concatenate((desired_xpos_list, desired_mat_list), axis=1)
        desired_xvel_list = np.array([[0, 0, 0, 0, 0, 0]], dtype=np.float64).repeat(step_num, axis=0)
        desired_xacc_list = np.array([[0, 0, 0, 0, 0, 0]], dtype=np.float64).repeat(step_num, axis=0)
        desired_force_list = np.array([[0, 0, 0, 0, 0, 0]], dtype=np.float64).repeat(step_num, axis=0)
        # 阻抗参数
        wn = 20
        damping_ratio = np.sqrt(2)
        K = np.array([1000, 1000, 1000, 1000, 1000, 1000], dtype=np.float64)
        M = K / (wn * wn)
        M_matrix = np.diag(M)
        B = 2 * damping_ratio * np.sqrt(M * K)
        controller_parameter = {'M': M_matrix, 'B': B, 'K': K}
        controller = ImpedanceController
        min_K = np.array([100, 100, 100, 100, 100, 100], dtype=np.float64)
        max_K = np.array([3000, 3000, 3000, 3000, 3000, 3000], dtype=np.float64)
        rbt_kwargs = dict(mjc_model_path=mjc_model_path, task=task, qpos_init_list=qpos_init_list,
                          p_bias=p_bias, r_bias=r_bias)
        # 用于Jk5StickStiffnessEnv的超参数
        rbt_controller_kwargs = copy.deepcopy(rbt_kwargs)
        rbt_controller_kwargs.update(controller_parameter=controller_parameter, controller=controller,
                                     step_num=step_num,
                                     desired_xposture_list=desired_xposture_list, desired_xvel_list=desired_xvel_list,
                                     desired_xacc_list=desired_xacc_list, desired_force_list=desired_force_list)

        # 用于TrainEnv的超参数
        rl_env_kwargs = copy.deepcopy(rbt_controller_kwargs)
        rl_env_kwargs.update(min_K=min_K, max_K=max_K,
                             rl_frequency=rl_frequency, observation_range=observation_range)
    elif task == 'close door':
        # 实验内容
        mjc_model_path = 'robot/jk5_door.xml'
        qpos_init_list = np.array([0, -10, 145, -135, -90, 0]) / 180 * np.pi
        xpos_init_list = np.array([0.38273612, -0.3625067, 0.275])
        p_bias = np.array([0, 0, 0.3885])
        r_bias = quaternion_matrix([0.5, 0.5, 0.5, 0.5])[:3, :3]
        rl_frequency = 250
        observation_range = 1
        step_num = 2000
        # 期望轨迹
        center_pred = np.array([-0.03, -0.4, 0.275])
        handle_pred = np.abs(xpos_init_list[0] - center_pred[0])  # 把手长度预测
        door_pred = np.abs(xpos_init_list[1] - center_pred[1])  # 门板长度预测
        radius_pred = np.linalg.norm(xpos_init_list[:2] - center_pred[:2])  # 半径预测
        angle_pred = np.arctan(door_pred / handle_pred) + np.pi / 2  # 角度预测
        total_angle = np.pi / 2  # 开门角度

        rot_matrix_init = np.eye(4)
        rot_matrix_init[:3, :3] = np.array([[0, 1, 0], [0, 0, -1], [-1, 0, -0]])  # 初始旋转矩阵
        quat_init = quaternion_from_matrix(rot_matrix_init)  # 初始四元数
        rot_matrix_last = np.eye(4)
        rot_matrix_last[:3, :3] = np.array([[0, 0, -1], [0, -1, 0], [-1, 0, 0]])  # 最终旋转矩阵
        quat_last = quaternion_from_matrix(rot_matrix_last)  # 最终四元数
        # 轨迹规划
        desired_xpos_list = np.empty((step_num, 3))  # xpos traj
        desired_mat_list = np.empty((step_num, 9))  # xmat traj
        for i in range(step_num):
            desired_xpos_list[i][0] = center_pred[0] \
                                      + radius_pred * np.sin(angle_pred - total_angle * i / (step_num - 1))
            desired_xpos_list[i][1] = center_pred[1] \
                                      - radius_pred * np.cos(angle_pred - total_angle * i / (step_num - 1))
            desired_xpos_list[i][2] = 0.275
            desired_mat_list[i] = quaternion_matrix(quaternion_slerp(quat_init, quat_last, i / (step_num - 1)))[:3,
                                  :3].reshape(-1)

        desired_xposture_list = np.concatenate((desired_xpos_list, desired_mat_list), axis=1)
        desired_xvel_list = np.array([[0, 0, 0, 0, 0, 0]], dtype=np.float64).repeat(step_num, axis=0)
        desired_xacc_list = np.array([[0, 0, 0, 0, 0, 0]], dtype=np.float64).repeat(step_num, axis=0)
        desired_force_list = np.array([[0, 0, 0, 0, 0, 0]], dtype=np.float64).repeat(step_num, axis=0)
        # 阻抗参数
        wn = 20
        damping_ratio = np.sqrt(2)
        K = np.array([1000, 1000, 1000, 1000, 1000, 1000], dtype=np.float64)
        M = K / (wn * wn)
        M_matrix = np.diag(M)
        B = 2 * damping_ratio * np.sqrt(M * K)
        controller_parameter = {'M': M_matrix, 'B': B, 'K': K}
        controller = ImpedanceController
        min_K = np.array([100, 100, 100, 100, 100, 100], dtype=np.float64)
        max_K = np.array([3000, 3000, 3000, 3000, 3000, 3000], dtype=np.float64)
        rbt_kwargs = dict(mjc_model_path=mjc_model_path, task=task, qpos_init_list=qpos_init_list,
                          p_bias=p_bias, r_bias=r_bias)
        # 用于Jk5StickStiffnessEnv的超参数
        rbt_controller_kwargs = copy.deepcopy(rbt_kwargs)
        rbt_controller_kwargs.update(controller_parameter=controller_parameter, controller=controller,
                                     step_num=step_num,
                                     desired_xposture_list=desired_xposture_list, desired_xvel_list=desired_xvel_list,
                                     desired_xacc_list=desired_xacc_list, desired_force_list=desired_force_list)

        # 用于TrainEnv的超参数
        rl_env_kwargs = copy.deepcopy(rbt_controller_kwargs)
        rl_env_kwargs.update(min_K=min_K, max_K=max_K,
                             rl_frequency=rl_frequency, observation_range=observation_range)

    elif task == 'cabinet surface':
        # 实验内容
        mjc_model_path = 'robot/jk5_cabinet_v1.xml'
        qpos_init_list = np.array([0, -30, 60, 0, -90, 0]) / 180 * np.pi
        p_bias = np.zeros(3)
        r_bias = np.eye(3)
        rl_frequency = 20  # 500可以整除，越大越多
        observation_range = 1
        step_num = 2000
        # 期望轨迹
        desired_xpos_list = np.concatenate((np.linspace(-0.4, -0.4, step_num).reshape(step_num, 1),
                                            np.linspace(-0.15, 0.15, step_num).reshape(step_num, 1),
                                            0.565 * np.ones((step_num, 1), dtype=float)), axis=1)
        desired_mat_list = np.array([[0, -1, 0, -1, 0, 0, 0, 0, -1]], dtype=np.float64).repeat(step_num, axis=0)
        desired_xposture_list = np.concatenate((desired_xpos_list, desired_mat_list), axis=1)
        desired_xvel_list = np.array([[0, 0.3 / 4, 0, 0, 0, 0]], dtype=np.float64).repeat(step_num, axis=0)
        desired_xacc_list = np.array([[0, 0, 0, 0, 0, 0]], dtype=np.float64).repeat(step_num, axis=0)
        desired_force_list = np.array([[0, 0, 30, 0, 0, 0]], dtype=np.float64).repeat(step_num, axis=0)
        # 阻抗参数
        wn = 20
        damping_ratio = np.sqrt(2)
        K = np.array([3000, 3000, 3000, 3000, 3000, 3000], dtype=np.float64)
        M = K / (wn * wn)
        M_matrix = np.diag(M)
        B = 2 * damping_ratio * np.sqrt(M * K)
        controller_parameter = {'M': M_matrix, 'B': B, 'K': K}
        controller = AdmittanceController
        min_K = np.array([100, 100, 100, 100, 100, 100], dtype=np.float64)
        max_K = np.array([3000, 3000, 3000, 3000, 3000, 3000], dtype=np.float64)
        rbt_kwargs = dict(mjc_model_path=mjc_model_path, task=task, qpos_init_list=qpos_init_list,
                          p_bias=p_bias, r_bias=r_bias)
        # 用于Jk5StickStiffnessEnv的超参数
        rbt_controller_kwargs = copy.deepcopy(rbt_kwargs)
        rbt_controller_kwargs.update(controller_parameter=controller_parameter, controller=controller,
                                     step_num=step_num,
                                     desired_xposture_list=desired_xposture_list, desired_xvel_list=desired_xvel_list,
                                     desired_xacc_list=desired_xacc_list, desired_force_list=desired_force_list)

        # 用于TrainEnv的超参数
        rl_env_kwargs = copy.deepcopy(rbt_controller_kwargs)
        rl_env_kwargs.update(min_K=min_K, max_K=max_K,
                             rl_frequency=rl_frequency, observation_range=observation_range)
    elif task == 'cabinet surface with plan':
        # 机器人参数
        mjc_model_path = 'robot/jk5_cabinet_surface.xml'
        qpos_init_list = np.array([0, 3.08546578e-01, -1.58109212e+00, -2.98250782e-01, 1.57079633e+00, 0])
        p_bias = np.array([0, 0, 0.2])
        r_bias = np.eye(3)
        rbt_kwargs = dict(mjc_model_path=mjc_model_path, task=task, qpos_init_list=qpos_init_list,
                          p_bias=p_bias, r_bias=r_bias)

        # 用于Jk5StickStiffnessEnv的超参数
        robot_frequency = 500
        time_whole = 4
        time_acceleration = 0.5
        step_num = robot_frequency * time_whole
        # 期望轨迹
        xpos_init, xmat_init = np.array([0.35, -0.1135, 0.565]), np.array([0, 1, 0, 1, 0, 0, 0, 0, -1])
        xpos_end, xmat_end = np.array([0.65, -0.1135, 0.565]), np.array([0, 1, 0, 1, 0, 0, 0, 0, -1])
        desired_xposture_list, desired_xvel_list, desired_xacc_list = trajectory_planning_line(xpos_init, xmat_init,
                                                                                               xpos_end, xmat_end,
                                                                                               time_whole,
                                                                                               time_acceleration,
                                                                                               robot_frequency)
        desired_force_list = np.array([[0, 0, 30, 0, 0, 0]], dtype=np.float64).repeat(step_num, axis=0)
        # 阻抗参数
        wn = 20
        damping_ratio = np.sqrt(2)
        K = np.array([1000, 1000, 1000, 1000, 1000, 1000], dtype=np.float64)
        M = K / (wn * wn)
        M_matrix = np.diag(M)
        B = 2 * damping_ratio * np.sqrt(M * K)
        controller_parameter = {'M': M_matrix, 'B': B, 'K': K}
        controller = AdmittanceController
        # wn = 20
        # damping_ratio = np.sqrt(2)
        # kp = wn * wn * np.ones(6, dtype=np.float64)
        # kd = 2 * damping_ratio * np.sqrt(kp)
        # controller_parameter = {'kp': kp, 'kd': kd}
        # controller = ComputedTorqueController

        rbt_controller_kwargs = copy.deepcopy(rbt_kwargs)
        rbt_controller_kwargs.update(controller_parameter=controller_parameter, controller=controller,
                                     step_num=step_num,
                                     desired_xposture_list=desired_xposture_list, desired_xvel_list=desired_xvel_list,
                                     desired_xacc_list=desired_xacc_list, desired_force_list=desired_force_list)

        # 用于TrainEnv的超参数
        min_K = np.array([50, 50, 50, 50, 50, 50], dtype=np.float64)
        max_K = np.array([5000, 5000, 5000, 5000, 5000, 5000], dtype=np.float64)
        max_force = np.array([50, 50, 50, 50, 50, 50], dtype=np.float64)
        # [0.35, -0.1135, 0.565, 0., 1., 0., 1., 0., 0., 0., 0., -1., 0.70710678, 0.70710678, 0., 0.]
        min_desired_xposture = np.array([0.35, -0.1235, 0.56, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1])
        # [0.65, -0.1135, 0.565, 0., 1., 0., 1., 0., 0., 0., 0., -1., 0.70710678, 0.70710678, 0., 0.]
        max_desired_xposture = np.array([0.65, -0.1035, 0.6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        min_desired_xvel = -0.15 * np.ones(6)
        max_desired_xvel = 0.15 * np.ones(6)
        min_desired_xacc = desired_xacc_list.min() * np.ones(6)
        max_desired_xacc = desired_xacc_list.max() * np.ones(6)
        rl_frequency = 20  # 500可以整除，越大越多
        observation_range = 1

        rl_env_kwargs = copy.deepcopy(rbt_controller_kwargs)
        rl_env_kwargs.update(min_K=min_K, max_K=max_K, max_force=max_force,
                             min_desired_xposture=min_desired_xposture, max_desired_xposture=max_desired_xposture,
                             min_desired_xvel=min_desired_xvel, max_desired_xvel=max_desired_xvel,
                             min_desired_xacc=min_desired_xacc, max_desired_xacc=max_desired_xacc,
                             rl_frequency=rl_frequency, observation_range=observation_range)
    elif task == 'cabinet surface with plan test':
        # 机器人参数
        mjc_model_path = 'robot/jk5_cabinet_surface_v1.xml'
        qpos_init_list = np.array([0, 3.08546578e-01, -1.58109212e+00, -2.98250782e-01, 1.57079633e+00, 0])
        p_bias = np.array([0, 0, 0.2])
        r_bias = np.eye(3)
        rbt_kwargs = dict(mjc_model_path=mjc_model_path, task=task, qpos_init_list=qpos_init_list,
                          p_bias=p_bias, r_bias=r_bias)

        # 用于Jk5StickStiffnessEnv的超参数
        robot_frequency = 500
        time_whole = 4
        time_acceleration = 0.5
        step_num = robot_frequency * time_whole
        # 期望轨迹
        table = np.array([[0.9986295, 0, -0.0523360],
                          [0, 1, 0],
                          [0.0523360, 0, 0.9986295]])
        init = np.array([[0, 1, 0],
                         [1, 0, 0],
                         [0, 0, -1]])
        # xpos_init, xmat_init = np.array([0.35, -0.1135, 0.565]), (table @ init).reshape(-1)
        # xpos_end, xmat_end = np.array([0.65, -0.1135, 0.565 + 0.3 * np.tan(3 / 180 * np.pi)]), (table @ init).reshape(-1)
        xpos_init, xmat_init = np.array([0.45, -0.1135, 0.565]), (table @ init).reshape(-1)
        xpos_end, xmat_end = np.array([0.45, -0.1135, 0.565]), (table @ init).reshape(-1)
        desired_xposture_list, desired_xvel_list, desired_xacc_list = trajectory_planning_line(xpos_init, xmat_init,
                                                                                               xpos_end, xmat_end,
                                                                                               time_whole,
                                                                                               time_acceleration,
                                                                                               robot_frequency)
        desired_force_list = np.array([[-30 * np.sin(np.pi / 60), 0, 30 * np.cos(np.pi / 60), 0, 0, 0]],
                                      dtype=np.float64).repeat(step_num, axis=0)
        # 阻抗参数
        wn = 20
        damping_ratio = np.sqrt(2)
        K = np.array([1000, 1000, 1000, 1000, 1000, 1000], dtype=np.float64)
        M = K / (wn * wn)
        B = 2 * damping_ratio * np.sqrt(M * K)
        # SM = np.eye(3)
        SM = np.array([[0.9986295, 0, -0.0523360],
                       [0, 1, 0],
                       [0.0523360, 0, 0.9986295]])
        controller_parameter = {'M': M, 'B': B, 'K': K, 'SM': SM}
        controller = AdmittanceController_v2

        rbt_controller_kwargs = copy.deepcopy(rbt_kwargs)
        rbt_controller_kwargs.update(controller_parameter=controller_parameter, controller=controller,
                                     step_num=step_num,
                                     desired_xposture_list=desired_xposture_list, desired_xvel_list=desired_xvel_list,
                                     desired_xacc_list=desired_xacc_list, desired_force_list=desired_force_list)

        # 用于TrainEnv的超参数
        min_K = np.array([50, 50, 50, 50, 50, 50], dtype=np.float64)
        max_K = np.array([5000, 5000, 5000, 5000, 5000, 5000], dtype=np.float64)
        max_force = np.array([50, 50, 50, 50, 50, 50], dtype=np.float64)
        # [0.35, -0.1135, 0.565, 0., 1., 0., 1., 0., 0., 0., 0., -1., 0.70710678, 0.70710678, 0., 0.]
        min_desired_xposture = np.array([0.35, -0.1335, 0.56, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1])
        # [0.65, -0.1135, 0.565, 0., 1., 0., 1., 0., 0., 0., 0., -1., 0.70710678, 0.70710678, 0., 0.]
        max_desired_xposture = np.array([0.65, -0.0935, 0.6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        min_desired_xvel = -0.15 * np.ones(6)
        max_desired_xvel = 0.15 * np.ones(6)
        min_desired_xacc = desired_xacc_list.min() * np.ones(6)
        max_desired_xacc = desired_xacc_list.max() * np.ones(6)
        rl_frequency = 20  # 500可以整除，越大越多
        observation_range = 1

        rl_env_kwargs = copy.deepcopy(rbt_controller_kwargs)
        rl_env_kwargs.update(min_K=min_K, max_K=max_K, max_force=max_force,
                             min_desired_xposture=min_desired_xposture, max_desired_xposture=max_desired_xposture,
                             min_desired_xvel=min_desired_xvel, max_desired_xvel=max_desired_xvel,
                             min_desired_xacc=min_desired_xacc, max_desired_xacc=max_desired_xacc,
                             rl_frequency=rl_frequency, observation_range=observation_range)
    elif task == 'cabinet surface with plan v7':
        # 机器人参数
        mjc_model_path = 'robot/jk5_cabinet_surface_v1.xml'
        qpos_init_list = np.array([0, 3.08546578e-01, -1.58109212e+00, -2.98250782e-01, 1.57079633e+00, 0])
        p_bias = np.array([0, 0, 0.2])
        r_bias = np.eye(3)
        rbt_kwargs = dict(mjc_model_path=mjc_model_path, task=task, qpos_init_list=qpos_init_list,
                          p_bias=p_bias, r_bias=r_bias)

        # 用于Jk5StickStiffnessEnv的超参数
        robot_frequency = 500
        time_whole = 4
        time_acceleration = 0.5
        step_num = robot_frequency * time_whole
        # 期望轨迹
        xpos_init, xmat_init = np.array([0.35, -0.1135, 0.565]), np.array([0, 1, 0, 1, 0, 0, 0, 0, -1])
        xpos_end, xmat_end = np.array([0.65, -0.1135, 0.565]), np.array([0, 1, 0, 1, 0, 0, 0, 0, -1])
        desired_xposture_list, desired_xvel_list, desired_xacc_list = trajectory_planning_line(xpos_init, xmat_init,
                                                                                               xpos_end, xmat_end,
                                                                                               time_whole,
                                                                                               time_acceleration,
                                                                                               robot_frequency)
        desired_force_list = np.array([[-30 * np.sin(np.pi / 60), 0, 30 * np.cos(np.pi / 60), 0, 0, 0]],
                                      dtype=np.float64).repeat(step_num, axis=0)
        # 阻抗参数
        wn = 20
        damping_ratio = np.sqrt(2)
        K = np.array([1000, 1000, 1000, 1000, 1000, 1000], dtype=np.float64)
        M = K / (wn * wn)
        B = 2 * damping_ratio * np.sqrt(M * K)
        SM = np.eye(3, dtype=np.float64)
        # SM = np.array([[0.9986295, 0, -0.0523360],
        #                   [0, 1, 0],
        #                   [0.0523360, 0, 0.9986295]])
        controller_parameter = {'M': M, 'B': B, 'K': K, 'SM': SM}
        controller = AdmittanceController_v2
        # wn = 20
        # damping_ratio = np.sqrt(2)
        # kp = wn * wn * np.ones(6, dtype=np.float64)
        # kd = 2 * damping_ratio * np.sqrt(kp)
        # controller_parameter = {'kp': kp, 'kd': kd}
        # controller = ComputedTorqueController

        rbt_controller_kwargs = copy.deepcopy(rbt_kwargs)
        rbt_controller_kwargs.update(controller_parameter=controller_parameter, controller=controller,
                                     step_num=step_num,
                                     desired_xposture_list=desired_xposture_list, desired_xvel_list=desired_xvel_list,
                                     desired_xacc_list=desired_xacc_list, desired_force_list=desired_force_list)

        # 用于TrainEnv的超参数
        min_K = np.array([50, 50, 50, 50, 50, 50], dtype=np.float64)
        max_K = np.array([5000, 5000, 5000, 5000, 5000, 5000], dtype=np.float64)
        max_force = np.array([50, 50, 50, 50, 50, 50], dtype=np.float64)
        # [0.35, -0.1135, 0.565, 0., 1., 0., 1., 0., 0., 0., 0., -1., 0.70710678, 0.70710678, 0., 0.]
        min_desired_xposture = np.array([0.35, -0.1335, 0.56, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1])
        # [0.65, -0.1135, 0.565, 0., 1., 0., 1., 0., 0., 0., 0., -1., 0.70710678, 0.70710678, 0., 0.]
        max_desired_xposture = np.array([0.65, -0.0935, 0.6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        min_desired_xvel = -0.15 * np.ones(6)
        max_desired_xvel = 0.15 * np.ones(6)
        min_desired_xacc = desired_xacc_list.min() * np.ones(6)
        max_desired_xacc = desired_xacc_list.max() * np.ones(6)
        rl_frequency = 20  # 500可以整除，越大越多
        observation_range = 10

        rl_env_kwargs = copy.deepcopy(rbt_controller_kwargs)
        rl_env_kwargs.update(min_K=min_K, max_K=max_K, max_force=max_force,
                             min_desired_xposture=min_desired_xposture, max_desired_xposture=max_desired_xposture,
                             min_desired_xvel=min_desired_xvel, max_desired_xvel=max_desired_xvel,
                             min_desired_xacc=min_desired_xacc, max_desired_xacc=max_desired_xacc,
                             rl_frequency=rl_frequency, observation_range=observation_range)

    elif task == 'cabinet drawer open':
        # 实验内容
        mjc_model_path = 'robot/jk5_cabinet_drawer.xml'
        qpos_init_list = np.array([0, 1.16644734e+01, -9.85767024e+01, 8.69122291e+01, 90, 0]) / 180 * np.pi
        p_bias = np.zeros(3)
        r_bias = np.eye(3)
        rl_frequency = 20  # 500可以整除，越大越多
        observation_range = 1
        step_num = 2000
        # 期望轨迹
        desired_xpos_list = np.concatenate((np.linspace(0.5715, 0.2715, step_num).reshape(step_num, 1),
                                            np.linspace(-0.1135, -0.1135, step_num).reshape(step_num, 1),
                                            np.linspace(0.681, 0.681, step_num).reshape(step_num, 1)), axis=1)
        desired_mat_list = np.array([[0, 0, 1, 1, 0, 0, 0, 1, 0]], dtype=np.float64).repeat(step_num, axis=0)
        desired_xposture_list = np.concatenate((desired_xpos_list, desired_mat_list), axis=1)
        desired_xvel_list = np.array([[-0.3 / 4, 0, 0, 0, 0, 0]], dtype=np.float64).repeat(step_num, axis=0)
        desired_xacc_list = np.array([[0, 0, 0, 0, 0, 0]], dtype=np.float64).repeat(step_num, axis=0)
        desired_force_list = np.array([[0, 0, 30, 0, 0, 0]], dtype=np.float64).repeat(step_num, axis=0)
        # 阻抗参数
        wn = 20
        damping_ratio = np.sqrt(2)
        K = np.array([1000, 1000, 1000, 1000, 1000, 1000], dtype=np.float64)
        M = K / (wn * wn)
        M_matrix = np.diag(M)
        B = 2 * damping_ratio * np.sqrt(M * K)
        controller_parameter = {'M': M_matrix, 'B': B, 'K': K}
        controller = AdmittanceController
        min_K = np.array([100, 100, 100, 100, 100, 100], dtype=np.float64)
        max_K = np.array([3000, 3000, 3000, 3000, 3000, 3000], dtype=np.float64)
        rbt_kwargs = dict(mjc_model_path=mjc_model_path, task=task, qpos_init_list=qpos_init_list,
                          p_bias=p_bias, r_bias=r_bias)
        # 用于Jk5StickStiffnessEnv的超参数
        rbt_controller_kwargs = copy.deepcopy(rbt_kwargs)
        rbt_controller_kwargs.update(controller_parameter=controller_parameter, controller=controller,
                                     step_num=step_num,
                                     desired_xposture_list=desired_xposture_list, desired_xvel_list=desired_xvel_list,
                                     desired_xacc_list=desired_xacc_list, desired_force_list=desired_force_list)

        # 用于TrainEnv的超参数
        rl_env_kwargs = copy.deepcopy(rbt_controller_kwargs)
        rl_env_kwargs.update(min_K=min_K, max_K=max_K,
                             rl_frequency=rl_frequency, observation_range=observation_range)
    elif task == 'cabinet drawer open with plan test':
        # 机器人参数
        mjc_model_path = 'robot/jk5_cabinet_drawer.xml'
        # qpos_init_list = np.array([0, 1.16644734e+01, -9.85767024e+01, 8.69122291e+01, 90, 0]) / 180 * np.pi
        qpos_init_list = np.array([0,  2.60251853e-02, -1.60034083e+00,  1.57431564e+00,  1.57079633e+00, 0])
        p_bias = np.zeros(3)
        r_bias = np.eye(3)
        rbt_kwargs = dict(mjc_model_path=mjc_model_path, task=task, qpos_init_list=qpos_init_list,
                          p_bias=p_bias, r_bias=r_bias)

        # 用于Jk5StickStiffnessEnv的超参数
        robot_frequency = 500
        time_whole = 4
        time_acceleration = 0.5
        step_num = robot_frequency * time_whole
        # 期望轨迹
        # xpos_init, xmat_init = np.array([0.5825-0.011, -0.1135, 0.675 + 0.004]), np.array([0, 0, 1, 1, 0, 0, 0, 1, 0])
        xpos_init, xmat_init = np.array([0.6579384969754818 - 0.011, -0.1135, 0.6629659585255823 + 0.004]), np.array(
            [0, 0, 1, 1, 0, 0, 0, 1, 0])
        # xpos_end, xmat_end = np.array([0.3715, -0.1135, 0.681]), np.array([0, 0, 1, 1, 0, 0, 0, 1, 0])
        xpos_end, xmat_end = xpos_init - np.array([0.3 * np.cos(np.pi / 60), 0, 0.3 * np.sin(np.pi / 60)]), np.array(
            [0, 0, 1, 1, 0, 0, 0, 1, 0])
        # xpos_end, xmat_end = xpos_init.copy(), np.array([0, 0, 1, 1, 0, 0, 0, 1, 0])
        desired_xposture_list, desired_xvel_list, desired_xacc_list = trajectory_planning_line(xpos_init, xmat_init,
                                                                                               xpos_end, xmat_end,
                                                                                               time_whole,
                                                                                               time_acceleration,
                                                                                               robot_frequency)
        desired_force_list = np.array([[0, 0, 0, 0, 0, 0]], dtype=np.float64).repeat(step_num, axis=0)
        # 阻抗参数
        wn = 20
        damping_ratio = np.sqrt(2)
        K = np.array([1000, 1000, 1000, 1000, 1000, 1000], dtype=np.float64)
        M = K / (wn * wn)
        # SM = np.eye(3, dtype=np.float64)
        SM = np.array([[0.9986295, 0, -0.0523360],
                       [0, 1, 0],
                       [0.0523360, 0, 0.9986295]])
        B = 2 * damping_ratio * np.sqrt(M * K)
        controller_parameter = {'M': M, 'B': B, 'K': K, 'SM': SM}
        controller = AdmittanceController_v2
        rbt_controller_kwargs = copy.deepcopy(rbt_kwargs)
        rbt_controller_kwargs.update(controller_parameter=controller_parameter, controller=controller,
                                     step_num=step_num,
                                     desired_xposture_list=desired_xposture_list, desired_xvel_list=desired_xvel_list,
                                     desired_xacc_list=desired_xacc_list, desired_force_list=desired_force_list)

        # 用于TrainEnv的超参数
        min_K = np.array([100, 100, 100, 100, 100, 100], dtype=np.float64)
        max_K = np.array([5000, 5000, 5000, 5000, 5000, 5000], dtype=np.float64)
        max_force = np.array([50, 50, 50, 50, 50, 50], dtype=np.float64)
        min_desired_xposture = np.array([0.3715, -0.1335, 0.641, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1])
        # [0.65, -0.1135, 0.565, 0., 1., 0., 1., 0., 0., 0., 0., -1., 0.70710678, 0.70710678, 0., 0.]
        max_desired_xposture = np.array([0.6715, -0.0935, 0.721, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        min_desired_xvel = -0.1 * np.ones(6)
        max_desired_xvel = 0.1 * np.ones(6)
        min_desired_xacc = desired_xacc_list.min() * np.ones(6)
        max_desired_xacc = desired_xacc_list.max() * np.ones(6)
        rl_frequency = 20  # 500可以整除，越大越多
        observation_range = 10

        rl_env_kwargs = copy.deepcopy(rbt_controller_kwargs)
        rl_env_kwargs.update(min_K=min_K, max_K=max_K, max_force=max_force,
                             min_desired_xposture=min_desired_xposture, max_desired_xposture=max_desired_xposture,
                             min_desired_xvel=min_desired_xvel, max_desired_xvel=max_desired_xvel,
                             min_desired_xacc=min_desired_xacc, max_desired_xacc=max_desired_xacc,
                             rl_frequency=rl_frequency, observation_range=observation_range)
    elif task == 'cabinet drawer open with plan':
        # 机器人参数
        mjc_model_path = 'robot/jk5_cabinet_drawer.xml'
        qpos_init_list = np.array([0,  2.60251853e-02, -1.60034083e+00,  1.57431564e+00,  1.57079633e+00, 0])
        p_bias = np.zeros(3)
        r_bias = np.eye(3)
        rbt_kwargs = dict(mjc_model_path=mjc_model_path, task=task, qpos_init_list=qpos_init_list,
                          p_bias=p_bias, r_bias=r_bias)

        # 用于Jk5StickStiffnessEnv的超参数
        robot_frequency = 500
        time_whole = 4
        time_acceleration = 0.5
        step_num = robot_frequency * time_whole
        # 期望轨迹
        # xpos_init, xmat_init = np.array([0.6825-0.011, -0.1135, 0.675 + 0.004]), np.array([0, 0, 1, 1, 0, 0, 0, 1, 0])
        xpos_init, xmat_init = np.array([0.6579384969754818 - 0.011, -0.1135, 0.6629659585255823 + 0.004]), np.array(
            [0, 0, 1, 1, 0, 0, 0, 1, 0])
        # xpos_end, xmat_end = np.array([0.3715, -0.1135, 0.681]), np.array([0, 0, 1, 1, 0, 0, 0, 1, 0])
        xpos_end, xmat_end = xpos_init - np.array([0.3, 0, 0]), np.array([0, 0, 1, 1, 0, 0, 0, 1, 0])
        # xpos_end, xmat_end = xpos_init.copy(), np.array([0, 0, 1, 1, 0, 0, 0, 1, 0])
        desired_xposture_list, desired_xvel_list, desired_xacc_list = trajectory_planning_line(xpos_init, xmat_init,
                                                                                               xpos_end, xmat_end,
                                                                                               time_whole,
                                                                                               time_acceleration,
                                                                                               robot_frequency)
        desired_force_list = np.array([[0, 0, 0, 0, 0, 0]], dtype=np.float64).repeat(step_num, axis=0)
        # 阻抗参数
        wn = 20
        damping_ratio = np.sqrt(2)
        K = np.array([1000, 1000, 1000, 1000, 1000, 1000], dtype=np.float64)
        M = K / (wn * wn)
        SM = np.eye(3, dtype=np.float64)
        B = 2 * damping_ratio * np.sqrt(M * K)
        controller_parameter = {'M': M, 'B': B, 'K': K, 'SM': SM}
        controller = AdmittanceController_v2
        rbt_controller_kwargs = copy.deepcopy(rbt_kwargs)
        rbt_controller_kwargs.update(controller_parameter=controller_parameter, controller=controller,
                                     step_num=step_num,
                                     desired_xposture_list=desired_xposture_list, desired_xvel_list=desired_xvel_list,
                                     desired_xacc_list=desired_xacc_list, desired_force_list=desired_force_list)

        # 用于TrainEnv的超参数
        min_K = np.array([100, 100, 100, 100, 100, 100], dtype=np.float64)
        max_K = np.array([5000, 5000, 5000, 5000, 5000, 5000], dtype=np.float64)
        max_force = np.array([100, 100, 100, 100, 100, 100], dtype=np.float64)
        min_desired_xposture = np.array([0.3715, -0.1335, 0.641, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1])
        # [0.65, -0.1135, 0.565, 0., 1., 0., 1., 0., 0., 0., 0., -1., 0.70710678, 0.70710678, 0., 0.]
        max_desired_xposture = np.array([0.6715, -0.0935, 0.721, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        min_desired_xvel = -0.1 * np.ones(6)
        max_desired_xvel = 0.1 * np.ones(6)
        min_desired_xacc = desired_xacc_list.min() * np.ones(6)
        max_desired_xacc = desired_xacc_list.max() * np.ones(6)
        rl_frequency = 20  # 500可以整除，越大越多
        observation_range = 10

        rl_env_kwargs = copy.deepcopy(rbt_controller_kwargs)
        rl_env_kwargs.update(min_K=min_K, max_K=max_K, max_force=max_force,
                             min_desired_xposture=min_desired_xposture, max_desired_xposture=max_desired_xposture,
                             min_desired_xvel=min_desired_xvel, max_desired_xvel=max_desired_xvel,
                             min_desired_xacc=min_desired_xacc, max_desired_xacc=max_desired_xacc,
                             rl_frequency=rl_frequency, observation_range=observation_range)

    elif task == 'cabinet drawer close':
        # 实验内容
        mjc_model_path = 'robot/jk5_cabinet_v2.xml'
        qpos_init_list = np.array([0, 5.33701542e+01, -1.15618680e+02, 6.22485257e+01, 90, 0]) / 180 * np.pi
        p_bias = np.zeros(3)
        r_bias = np.eye(3)
        rl_frequency = 20  # 500可以整除，越大越多
        observation_range = 1
        step_num = 2000
        # 期望轨迹
        desired_xpos_list = np.concatenate((np.linspace(0.2715, 0.5715, step_num).reshape(step_num, 1),
                                            np.linspace(-0.1135, -0.1135, step_num).reshape(step_num, 1),
                                            np.linspace(0.681, 0.661, step_num).reshape(step_num, 1)), axis=1)
        desired_mat_list = np.array([[0, 0, 1, 1, 0, 0, 0, 1, 0]], dtype=np.float64).repeat(step_num, axis=0)
        desired_xposture_list = np.concatenate((desired_xpos_list, desired_mat_list), axis=1)
        desired_xvel_list = np.array([[0.3 / 4, 0, 0, 0, 0, 0]], dtype=np.float64).repeat(step_num, axis=0)
        desired_xacc_list = np.array([[0, 0, 0, 0, 0, 0]], dtype=np.float64).repeat(step_num, axis=0)
        desired_force_list = np.array([[0, 0, 30, 0, 0, 0]], dtype=np.float64).repeat(step_num, axis=0)
        # 阻抗参数
        wn = 20
        damping_ratio = np.sqrt(2)
        K = np.array([3000, 3000, 3000, 3000, 3000, 3000], dtype=np.float64)
        M = K / (wn * wn)
        M_matrix = np.diag(M)
        B = 2 * damping_ratio * np.sqrt(M * K)
        controller_parameter = {'M': M_matrix, 'B': B, 'K': K}
        controller = AdmittanceController
        min_K = np.array([100, 100, 100, 100, 100, 100], dtype=np.float64)
        max_K = np.array([3000, 3000, 3000, 3000, 3000, 3000], dtype=np.float64)
        rbt_kwargs = dict(mjc_model_path=mjc_model_path, task=task, qpos_init_list=qpos_init_list,
                          p_bias=p_bias, r_bias=r_bias)
        # 用于Jk5StickStiffnessEnv的超参数
        rbt_controller_kwargs = copy.deepcopy(rbt_kwargs)
        rbt_controller_kwargs.update(controller_parameter=controller_parameter, controller=controller,
                                     step_num=step_num,
                                     desired_xposture_list=desired_xposture_list, desired_xvel_list=desired_xvel_list,
                                     desired_xacc_list=desired_xacc_list, desired_force_list=desired_force_list)

        # 用于TrainEnv的超参数
        rl_env_kwargs = copy.deepcopy(rbt_controller_kwargs)
        rl_env_kwargs.update(min_K=min_K, max_K=max_K,
                             rl_frequency=rl_frequency, observation_range=observation_range)

    elif task == 'cabinet door open':
        # 实验内容
        mjc_model_path = 'robot/jk5_cabinet_v2.xml'
        qpos_init_list = np.array([27.47586566, 2.51816863, -131.42640482, 128.90823617, 62.52413433, 90]) / 180 * np.pi
        p_bias = np.zeros(3)
        r_bias = np.eye(3)
        rl_frequency = 20  # 500可以整除，越大越多
        observation_range = 1
        step_num = 2000
        # 期望轨迹
        desired_xpos_list = np.concatenate((np.linspace(0.5715, 0.5715, step_num).reshape(step_num, 1),
                                            np.linspace(-0.1135, -0.1135, step_num).reshape(step_num, 1),
                                            np.linspace(0.42, 0.42, step_num).reshape(step_num, 1)), axis=1)
        desired_mat_list = np.array([[0, 0, 1, 1, 0, 0, 0, 1, 0]], dtype=np.float64).repeat(step_num, axis=0)
        desired_xposture_list = np.concatenate((desired_xpos_list, desired_mat_list), axis=1)
        desired_xvel_list = np.array([[0 / 0, 0, 0, 0, 0, 0]], dtype=np.float64).repeat(step_num, axis=0)
        desired_xacc_list = np.array([[0, 0, 0, 0, 0, 0]], dtype=np.float64).repeat(step_num, axis=0)
        desired_force_list = np.array([[0, 0, 30, 0, 0, 0]], dtype=np.float64).repeat(step_num, axis=0)
        # 阻抗参数
        wn = 20
        damping_ratio = np.sqrt(2)
        K = np.array([1000, 1000, 1000, 1000, 1000, 1000], dtype=np.float64)
        M = K / (wn * wn)
        M_matrix = np.diag(M)
        B = 2 * damping_ratio * np.sqrt(M * K)
        controller_parameter = {'M': M_matrix, 'B': B, 'K': K}
        controller = AdmittanceController
        min_K = np.array([100, 100, 100, 100, 100, 100], dtype=np.float64)
        max_K = np.array([3000, 3000, 3000, 3000, 3000, 3000], dtype=np.float64)
        rbt_kwargs = dict(mjc_model_path=mjc_model_path, task=task, qpos_init_list=qpos_init_list,
                          p_bias=p_bias, r_bias=r_bias)
        # 用于Jk5StickStiffnessEnv的超参数
        rbt_controller_kwargs = copy.deepcopy(rbt_kwargs)
        rbt_controller_kwargs.update(controller_parameter=controller_parameter, controller=controller,
                                     step_num=step_num,
                                     desired_xposture_list=desired_xposture_list, desired_xvel_list=desired_xvel_list,
                                     desired_xacc_list=desired_xacc_list, desired_force_list=desired_force_list)

        # 用于TrainEnv的超参数
        rl_env_kwargs = copy.deepcopy(rbt_controller_kwargs)
        rl_env_kwargs.update(min_K=min_K, max_K=max_K,
                             rl_frequency=rl_frequency, observation_range=observation_range)
    elif task == 'cabinet door open with plan test':
        # 机器人参数
        mjc_model_path = 'robot/jk5_cabinet_door.xml'
        qpos_init_list = np.array([0.19755593, 0.18191913, -2.11904879, 1.93712982, 1.3732404, 1.57079603])
        p_bias = np.zeros(3)
        r_bias = np.eye(3)
        rbt_kwargs = dict(mjc_model_path=mjc_model_path, task=task, qpos_init_list=qpos_init_list,
                          p_bias=p_bias, r_bias=r_bias)

        # 用于Jk5StickStiffnessEnv的超参数
        robot_frequency = 500
        time_whole = 4
        time_acceleration = 0.5
        step_num = robot_frequency * time_whole
        # 期望轨迹
        cabinet_pos = np.array([0.8, -0.2, 0.3])
        r = np.sqrt(0.34 ** 2 + 0.025 ** 2)
        center = cabinet_pos + np.array([-0.2 + 0.0075, -0.19, 0.22])
        rbt_tool = np.array([-0.011, -0.004, 0])
        xpos_init, xmat_init = rbt_tool + center + np.array([-0.025, 0.34, 0]), np.array([0, 0, 1, 0, -1, 0, 1, 0, 0])
        xpos_end, xmat_end = rbt_tool + center + np.array([-0.34, -0.025, 0]), np.array([0, 0, 1, 0, -1, 0, 1, 0, 0])
        xpos_mid = rbt_tool + center + np.array([-r / np.sqrt(2), r / np.sqrt(2), 0])
        desired_xposture_list, desired_xvel_list, desired_xacc_list = trajectory_planning_circle(xpos_init, xmat_init,
                                                                                                 xpos_end, xmat_end,
                                                                                                 xpos_mid,
                                                                                                 time_whole,
                                                                                                 time_acceleration,
                                                                                                 robot_frequency)
        desired_force_list = np.array([[0, 0, 0, 0, 0, 0]], dtype=np.float64).repeat(step_num, axis=0)
        # 阻抗参数
        wn = 20
        damping_ratio = np.sqrt(2)
        K = np.array([1000, 1000, 1000, 1000, 1000, 1000], dtype=np.float64)
        M = K / (wn * wn)
        SM = np.eye(3, dtype=np.float64)
        B = 2 * damping_ratio * np.sqrt(M * K)
        controller_parameter = {'M': M, 'B': B, 'K': K, 'SM': SM}
        controller = AdmittanceController_v2
        rbt_controller_kwargs = copy.deepcopy(rbt_kwargs)
        rbt_controller_kwargs.update(controller_parameter=controller_parameter, controller=controller,
                                     step_num=step_num,
                                     desired_xposture_list=desired_xposture_list, desired_xvel_list=desired_xvel_list,
                                     desired_xacc_list=desired_xacc_list, desired_force_list=desired_force_list)

        # 用于TrainEnv的超参数
        min_K = np.array([100, 100, 100, 100, 100, 100], dtype=np.float64)
        max_K = np.array([5000, 5000, 5000, 5000, 5000, 5000], dtype=np.float64)
        max_force = np.array([80, 80, 80, 80, 80, 80], dtype=np.float64)
        min_desired_xposture = np.array([0.25, -0.42, 0.51, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1])
        # [0.65, -0.1135, 0.565, 0., 1., 0., 1., 0., 0., 0., 0., -1., 0.70710678, 0.70710678, 0., 0.]
        max_desired_xposture = np.array([0.58, -0.05, 0.53, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        min_desired_xvel = -0.2 * np.ones(6)
        max_desired_xvel = 0.2 * np.ones(6)
        min_desired_xacc = -0.6 * np.ones(6) * np.ones(6)
        max_desired_xacc = -0.6 * np.ones(6) * np.ones(6)
        rl_frequency = 20  # 500可以整除，越大越多
        observation_range = 10

        rl_env_kwargs = copy.deepcopy(rbt_controller_kwargs)
        rl_env_kwargs.update(min_K=min_K, max_K=max_K, max_force=max_force,
                             min_desired_xposture=min_desired_xposture, max_desired_xposture=max_desired_xposture,
                             min_desired_xvel=min_desired_xvel, max_desired_xvel=max_desired_xvel,
                             min_desired_xacc=min_desired_xacc, max_desired_xacc=max_desired_xacc,
                             rl_frequency=rl_frequency, observation_range=observation_range)
    elif task == 'cabinet door open with plan':
        # 机器人参数
        mjc_model_path = 'robot/jk5_cabinet_door.xml'
        qpos_init_list = np.array([0.19755593, 0.18191913, -2.11904879, 1.93712982, 1.3732404, 1.57079603])
        p_bias = np.zeros(3)
        r_bias = np.eye(3)
        rbt_kwargs = dict(mjc_model_path=mjc_model_path, task=task, qpos_init_list=qpos_init_list,
                          p_bias=p_bias, r_bias=r_bias)

        # 用于Jk5StickStiffnessEnv的超参数
        robot_frequency = 500
        time_whole = 4
        time_acceleration = 0.5
        step_num = robot_frequency * time_whole
        # 期望轨迹
        cabinet_pos = np.array([0.8, -0.2, 0.3])
        r_bias = np.array([-0.025, 0.34, 0])
        radius = np.linalg.norm(r_bias)
        center = cabinet_pos + np.array([-0.2 + 0.0075, -0.19, 0.22])
        rbt_tool = np.array([-0.011, -0.004, 0])
        xpos_init, xmat_init = rbt_tool + center + r_bias, np.array([0, 0, 1, 0, -1, 0, 1, 0, 0])

        center_pred = center + np.array([0, 0.05, 0])  # 轨迹规划所用的中心与实际值有1cm的偏移
        r_bias_pred = xpos_init - rbt_tool - center_pred
        radius_pred = np.linalg.norm(r_bias_pred)

        xpos_end = rbt_tool + center_pred + np.array([-r_bias_pred[1], r_bias_pred[0], 0])
        xmat_end = np.array([0, 0, 1, 0, -1, 0, 1, 0, 0])
        xpos_mid = rbt_tool + center_pred + np.array([-radius_pred / np.sqrt(2), radius_pred / np.sqrt(2), 0])
        desired_xposture_list, desired_xvel_list, desired_xacc_list = trajectory_planning_circle(xpos_init, xmat_init,
                                                                                                 xpos_end, xmat_end,
                                                                                                 xpos_mid,
                                                                                                 time_whole,
                                                                                                 time_acceleration,
                                                                                                 robot_frequency)
        desired_force_list = np.array([[0, 0, 0, 0, 0, 0]], dtype=np.float64).repeat(step_num, axis=0)
        # 阻抗参数
        wn = 20
        damping_ratio = np.sqrt(2)
        K = np.array([1000, 1000, 1000, 1000, 1000, 1000], dtype=np.float64)
        M = K / (wn * wn)
        SM = np.eye(3, dtype=np.float64)
        B = 2 * damping_ratio * np.sqrt(M * K)
        controller_parameter = {'M': M, 'B': B, 'K': K, 'SM': SM}
        controller = AdmittanceController_v2
        rbt_controller_kwargs = copy.deepcopy(rbt_kwargs)
        rbt_controller_kwargs.update(controller_parameter=controller_parameter, controller=controller,
                                     step_num=step_num,
                                     desired_xposture_list=desired_xposture_list, desired_xvel_list=desired_xvel_list,
                                     desired_xacc_list=desired_xacc_list, desired_force_list=desired_force_list)

        # 用于TrainEnv的超参数
        min_K = np.array([500, 500, 500, 500, 500, 500], dtype=np.float64)
        max_K = np.array([5000, 5000, 5000, 5000, 5000, 5000], dtype=np.float64)
        max_force = np.array([75, 75, 75, 75, 75, 75], dtype=np.float64)
        min_desired_xposture = np.array([0.2715, -0.1335, 0.641, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1])
        # [0.65, -0.1135, 0.565, 0., 1., 0., 1., 0., 0., 0., 0., -1., 0.70710678, 0.70710678, 0., 0.]
        max_desired_xposture = np.array([0.5715, -0.0935, 0.721, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        min_desired_xvel = -0.1 * np.ones(6)
        max_desired_xvel = 0.1 * np.ones(6)
        min_desired_xacc = desired_xacc_list.min() * np.ones(6)
        max_desired_xacc = desired_xacc_list.max() * np.ones(6)
        rl_frequency = 20  # 500可以整除，越大越多
        observation_range = 10

        rl_env_kwargs = copy.deepcopy(rbt_controller_kwargs)
        rl_env_kwargs.update(min_K=min_K, max_K=max_K, max_force=max_force,
                             min_desired_xposture=min_desired_xposture, max_desired_xposture=max_desired_xposture,
                             min_desired_xvel=min_desired_xvel, max_desired_xvel=max_desired_xvel,
                             min_desired_xacc=min_desired_xacc, max_desired_xacc=max_desired_xacc,
                             rl_frequency=rl_frequency, observation_range=observation_range)

    elif task == 'cabinet door close':
        # 实验内容
        mjc_model_path = 'robot/jk5_cabinet_v2.xml'
        qpos_init_list = np.array([27.47586566, 2.51816863, -131.42640482, 128.90823617, 62.52413433, 90]) / 180 * np.pi
        p_bias = np.zeros(3)
        r_bias = np.eye(3)
        rl_frequency = 20  # 500可以整除，越大越多
        observation_range = 1
        step_num = 2000
        # 期望轨迹
        desired_xpos_list = np.concatenate((np.linspace(0.2664, 0.2664, step_num).reshape(step_num, 1),
                                            np.linspace(-0.3289, -0.3289, step_num).reshape(step_num, 1),
                                            np.linspace(0.42, 0.42, step_num).reshape(step_num, 1)), axis=1)
        desired_mat_list = np.array([[0, 0, 1, 1, 0, 0, 0, 1, 0]], dtype=np.float64).repeat(step_num, axis=0)
        desired_xposture_list = np.concatenate((desired_xpos_list, desired_mat_list), axis=1)
        desired_xvel_list = np.array([[-0.3 / 4, 0, 0, 0, 0, 0]], dtype=np.float64).repeat(step_num, axis=0)
        desired_xacc_list = np.array([[0, 0, 0, 0, 0, 0]], dtype=np.float64).repeat(step_num, axis=0)
        desired_force_list = np.array([[0, 0, 30, 0, 0, 0]], dtype=np.float64).repeat(step_num, axis=0)
        # 阻抗参数
        wn = 20
        damping_ratio = np.sqrt(2)
        K = np.array([1000, 1000, 1000, 1000, 1000, 1000], dtype=np.float64)
        M = K / (wn * wn)
        M_matrix = np.diag(M)
        B = 2 * damping_ratio * np.sqrt(M * K)
        controller_parameter = {'M': M_matrix, 'B': B, 'K': K}
        controller = AdmittanceController

        min_K = np.array([100, 100, 100, 100, 100, 100], dtype=np.float64)
        max_K = np.array([3000, 3000, 3000, 3000, 3000, 3000], dtype=np.float64)
        rbt_kwargs = dict(mjc_model_path=mjc_model_path, task=task, qpos_init_list=qpos_init_list,
                          p_bias=p_bias, r_bias=r_bias)
        # 用于Jk5StickStiffnessEnv的超参数
        rbt_controller_kwargs = copy.deepcopy(rbt_kwargs)
        rbt_controller_kwargs.update(controller_parameter=controller_parameter, controller=controller,
                                     step_num=step_num,
                                     desired_xposture_list=desired_xposture_list, desired_xvel_list=desired_xvel_list,
                                     desired_xacc_list=desired_xacc_list, desired_force_list=desired_force_list)

        # 用于TrainEnv的超参数
        rl_env_kwargs = copy.deepcopy(rbt_controller_kwargs)
        rl_env_kwargs.update(min_K=min_K, max_K=max_K,
                             rl_frequency=rl_frequency, observation_range=observation_range)
    elif task == 'cabinet door close with plan test':
        # 机器人参数
        mjc_model_path = 'robot/jk5_cabinet_door.xml'
        qpos_init_list = np.array([-1.31203182, -0.02868445, -1.4032487, -1.70965992, -2.88282817, -1.57079674])
        p_bias = np.zeros(3)
        r_bias = np.eye(3)
        rbt_kwargs = dict(mjc_model_path=mjc_model_path, task=task, qpos_init_list=qpos_init_list,
                          p_bias=p_bias, r_bias=r_bias)

        # 用于Jk5StickStiffnessEnv的超参数
        robot_frequency = 500
        time_whole = 4
        time_acceleration = 0.5
        step_num = robot_frequency * time_whole
        # 期望轨迹
        cabinet_pos = np.array([0.8, -0.2, 0.3])
        r = np.sqrt(0.34 ** 2 + 0.025 ** 2)
        center = cabinet_pos + np.array([-0.2 + 0.0075, -0.19, 0.22])
        rbt_tool = np.array([-0.011, -0.004, 0])
        cabinet_pos = np.array([0.8, -0.2, 0.3])
        r = np.sqrt(0.34 ** 2 + 0.025 ** 2)
        center = cabinet_pos + np.array([-0.2 + 0.0075, -0.19, 0.22])
        rbt_tool = np.array([-0.011, -0.004, 0])
        xpos_init, xmat_init = rbt_tool + center + np.array([-0.34, -0.025, 0]), np.array([0, 0, 1, 0, -1, 0, 1, 0, 0])
        xpos_end, xmat_end = rbt_tool + center + np.array([-0.025, 0.34, 0]), np.array([0, 0, 1, 0, -1, 0, 1, 0, 0])
        xpos_mid = rbt_tool + center + np.array([-r / np.sqrt(2), r / np.sqrt(2), 0])
        desired_xposture_list, desired_xvel_list, desired_xacc_list = trajectory_planning_circle(xpos_init, xmat_init,
                                                                                                 xpos_end, xmat_end,
                                                                                                 xpos_mid,
                                                                                                 time_whole,
                                                                                                 time_acceleration,
                                                                                                 robot_frequency)
        desired_force_list = np.array([[0, 0, 0, 0, 0, 0]], dtype=np.float64).repeat(step_num, axis=0)
        # 阻抗参数
        wn = 20
        damping_ratio = np.sqrt(2)
        K = np.array([1000, 1000, 1000, 1000, 1000, 1000], dtype=np.float64)
        M = K / (wn * wn)
        SM = np.eye(3, dtype=np.float64)
        B = 2 * damping_ratio * np.sqrt(M * K)
        controller_parameter = {'M': M, 'B': B, 'K': K, 'SM': SM}
        controller = AdmittanceController_v2
        rbt_controller_kwargs = copy.deepcopy(rbt_kwargs)
        rbt_controller_kwargs.update(controller_parameter=controller_parameter, controller=controller,
                                     step_num=step_num,
                                     desired_xposture_list=desired_xposture_list, desired_xvel_list=desired_xvel_list,
                                     desired_xacc_list=desired_xacc_list, desired_force_list=desired_force_list)

        # 用于TrainEnv的超参数
        min_K = np.array([100, 100, 100, 100, 100, 100], dtype=np.float64)
        max_K = np.array([5000, 5000, 5000, 5000, 5000, 5000], dtype=np.float64)
        max_force = np.array([50, 50, 50, 50, 50, 50], dtype=np.float64)
        min_desired_xposture = np.array([0.2715, -0.1335, 0.641, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1])
        # [0.65, -0.1135, 0.565, 0., 1., 0., 1., 0., 0., 0., 0., -1., 0.70710678, 0.70710678, 0., 0.]
        max_desired_xposture = np.array([0.5715, -0.0935, 0.721, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        min_desired_xvel = -0.1 * np.ones(6)
        max_desired_xvel = 0.1 * np.ones(6)
        min_desired_xacc = desired_xacc_list.min() * np.ones(6)
        max_desired_xacc = desired_xacc_list.max() * np.ones(6)
        rl_frequency = 20  # 500可以整除，越大越多
        observation_range = 10

        rl_env_kwargs = copy.deepcopy(rbt_controller_kwargs)
        rl_env_kwargs.update(min_K=min_K, max_K=max_K, max_force=max_force,
                             min_desired_xposture=min_desired_xposture, max_desired_xposture=max_desired_xposture,
                             min_desired_xvel=min_desired_xvel, max_desired_xvel=max_desired_xvel,
                             min_desired_xacc=min_desired_xacc, max_desired_xacc=max_desired_xacc,
                             rl_frequency=rl_frequency, observation_range=observation_range)

    if save_flag:
        save_env_kwargs(save_path, rl_env_kwargs)
    return rbt_kwargs, rbt_controller_kwargs, rl_env_kwargs


def save_env_kwargs(save_path=None, kwargs_dict=None):
    config_json = convert_json(kwargs_dict)
    if not osp.exists(save_path):
        os.makedirs(save_path)
    output = json.dumps(config_json, separators=(',', ':\t'), indent=4, sort_keys=True)
    with open(osp.join(save_path, "env_kwargs_dict.json"), 'w') as out:
        out.write(output)


def load_env_kwargs(task=None):
    return 0


def convert_json(obj):
    """ Convert obj to a version which can be serialized with JSON. """
    """复杂的类型不会经过任何支路，直接输出str(obj)
        很多类型如类、方法都是有默认的str的，所以不用操心，只需要把自己想特别处理的加上即可
    """
    if is_json_serializable(obj):
        return obj
    else:
        if isinstance(obj, dict):
            return {convert_json(k): convert_json(v)
                    for k, v in obj.items()}

        elif isinstance(obj, tuple):
            return (convert_json(x) for x in obj)

        elif isinstance(obj, list):
            return [convert_json(x) for x in obj]

        elif isinstance(obj, np.ndarray):
            return convert_json(obj.tolist())

        elif hasattr(obj, '__name__') and not ('lambda' in obj.__name__):
            return convert_json(obj.__name__)

        elif hasattr(obj, '__dict__') and obj.__dict__:
            obj_dict = {convert_json(k): convert_json(v)
                        for k, v in obj.__dict__.items()}
            return {str(obj): obj_dict}

        return str(obj)


def is_json_serializable(v):
    try:
        json.dumps(v)
        return True
    except:
        return False
