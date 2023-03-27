#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""所有任务的参数

Write detailed description here

Write typical usage example here

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
3/23/23 2:52 PM   yinzikang      1.0         None
"""
import numpy as np
from module.controller import *
from module.transformations import quaternion_matrix, quaternion_slerp


def load_env_kwargs(task=None):
    """
    不同的任务给定不同参数
    期望轨迹是用末端连杆（力传感器安装位置结合逆运动学算出来的）
    当前可用的只有带plan的
    cabinet surface1
    :param task:
    :return:
    """
    if task == 'desk':
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

        return rbt_kwargs, rbt_controller_kwargs, rl_env_kwargs
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

        return rbt_kwargs, rbt_controller_kwargs, rl_env_kwargs

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

        return rbt_kwargs, rbt_controller_kwargs, rl_env_kwargs
    elif task == 'cabinet surface with plan':
        # 实验内容
        mjc_model_path = 'robot/jk5_cabinet_v1.xml'
        qpos_init_list = np.array([0, -30, 60, 0, -90, 0]) / 180 * np.pi
        p_bias = np.zeros(3)
        r_bias = np.eye(3)
        rl_frequency = 20  # 500可以整除，越大越多
        observation_range = 1
        step_num = 2000
        time_whole = 4
        time_acceleration = 0
        f = 500
        # 期望轨迹
        xpos_init, xmat_init = np.array([-0.4, -0.15, 0.565]), np.array([0, -1, 0, -1, 0, 0, 0, 0, -1])
        xpos_end, xmat_end = np.array([-0.4, 0.15, 0.565]), np.array([0, -1, 0, -1, 0, 0, 0, 0, -1])
        desired_xposture_list, desired_xvel_list, desired_xacc_list = trajectory_planning_line(xpos_init, xmat_init,
                                                                                               xpos_end, xmat_end,
                                                                                               time_whole,
                                                                                               time_acceleration, f)
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
        max_K = np.array([5000, 5000, 5000, 5000, 5000, 5000], dtype=np.float64)
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
    elif task == 'cabinet surface abandoned':
        # 实验内容
        mjc_model_path = 'robot/jk5_cabinet_v2.xml'
        qpos_init_list = np.array([0, 10, -120, 110, 90, 0]) / 180 * np.pi
        p_bias = np.zeros(3)
        r_bias = np.eye(3)
        rl_frequency = 20  # 500可以整除，越大越多
        observation_range = 1
        step_num = 2000
        # 期望轨迹
        desired_xpos_list = np.concatenate((np.linspace(0.6, 0.6, step_num).reshape(step_num, 1),
                                            np.linspace(-0.15, 0.15, step_num).reshape(step_num, 1),
                                            0.55 * np.ones((step_num, 1), dtype=float)), axis=1)
        desired_mat_list = np.array([[0, 0, 1, 1, 0, 0, 0, 1, 0]], dtype=np.float64).repeat(step_num, axis=0)
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

        return rbt_kwargs, rbt_controller_kwargs, rl_env_kwargs

    elif task == 'cabinet drawer open':
        # 实验内容
        mjc_model_path = 'robot/jk5_cabinet_v2.xml'
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

        return rbt_kwargs, rbt_controller_kwargs, rl_env_kwargs
    elif task == 'cabinet drawer open with plan':
        # 实验内容
        mjc_model_path = 'robot/jk5_cabinet_v2.xml'
        qpos_init_list = np.array([0, 1.16644734e+01, -9.85767024e+01, 8.69122291e+01, 90, 0]) / 180 * np.pi
        p_bias = np.zeros(3)
        r_bias = np.eye(3)
        rl_frequency = 20  # 500可以整除，越大越多
        observation_range = 1
        step_num = 2000
        time_whole = 4
        time_acceleration = 2
        f = 500
        # 期望轨迹
        xpos_init, xmat_init = np.array([0.5715, -0.1135, 0.681]), np.array([0, 0, 1, 1, 0, 0, 0, 1, 0])
        xpos_end, xmat_end = np.array([0.2715, -0.1135, 0.681]), np.array([0, 0, 1, 1, 0, 0, 0, 1, 0])
        desired_xposture_list, desired_xvel_list, desired_xacc_list = trajectory_planning_line(xpos_init, xmat_init,
                                                                                               xpos_end, xmat_end,
                                                                                               time_whole,
                                                                                               time_acceleration, f)
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

        return rbt_kwargs, rbt_controller_kwargs, rl_env_kwargs
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

        return rbt_kwargs, rbt_controller_kwargs, rl_env_kwargs

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

        return rbt_kwargs, rbt_controller_kwargs, rl_env_kwargs
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

        return rbt_kwargs, rbt_controller_kwargs, rl_env_kwargs

    return None
