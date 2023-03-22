#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""机器人控制器相关

轨迹生成
姿态误差
多种控制器

Write typical usage example here

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
3/16/23 10:22 PM   yinzikang      1.0         None
"""
import copy

import numpy as np
from module.transformations import quaternion_from_matrix


def generate_trajectory(cart_init_pos, cart_end_pos, dot_num):
    trajectory = []
    for i in range(dot_num):
        trajectory.append(cart_init_pos + (cart_end_pos - cart_init_pos) / (dot_num - 1) * i)
    return trajectory


def orientation_error_axis_angle(desired, current):
    rc1 = current[0:3, 0]
    rc2 = current[0:3, 1]
    rc3 = current[0:3, 2]
    rd1 = desired[0:3, 0]
    rd2 = desired[0:3, 1]
    rd3 = desired[0:3, 2]
    error = 0.5 * (np.cross(rc1, rd1) + np.cross(rc2, rd2) + np.cross(rc3, rd3))
    return error


def orientation_error_quaternion(desired, current):
    """
    虚部+实部，前三个当偏差
    转换到当前坐标系
    :param desired:
    :param current:
    :return:
    """
    mat44 = np.eye(4)
    mat44[:3, :3] = np.linalg.inv(current) @ desired
    quat = quaternion_from_matrix(mat44)
    error = current @ quat[:3].transpose()
    return error


# 迪卡尔空间下计算力矩控制器
class ComputedTorqueController:
    def __init__(self, orientation_error):
        self.orientation_error = orientation_error

    def step(self, status):
        # 状态
        desired_xpos = status['desired_xpos']
        desired_xmat = status['desired_xmat']
        desired_xvel = status['desired_xvel']
        desired_xacc = status['desired_xacc']
        xpos = status['xpos']
        xmat = status['xmat']
        xvel = status['xvel']

        J = status['J']
        D_x = status['D_x']
        CG_x = status['CG_x']
        contact_force = status['contact_force']

        # 控制器参数
        kp = status['controller_parameter']['kp']
        kd = status['controller_parameter']['kd']

        xposture_error = np.concatenate([desired_xpos - xpos, self.orientation_error(desired_xmat, xmat)])
        xvel_error = desired_xvel - xvel

        solved_acc = desired_xacc + np.multiply(kd, xvel_error) + np.multiply(kp, xposture_error)
        F = np.dot(D_x, solved_acc) + CG_x - contact_force
        tau = np.dot(J.T, F)

        return tau


# 迪卡尔空间下阻抗控制器
class ImpedanceController:
    def __init__(self, orientation_error):
        self.orientation_error = orientation_error

    def step(self, status):
        # 状态
        desired_xpos = status['desired_xpos']
        desired_xmat = status['desired_xmat']
        desired_xvel = status['desired_xvel']
        desired_xacc = status['desired_xacc']
        xpos = status['xpos']
        xmat = status['xmat']
        xvel = status['xvel']

        J = status['J']
        D_x = status['D_x']
        CG_x = status['CG_x']
        contact_force = status['contact_force']

        # 控制器参数
        M = status['controller_parameter']['M']
        B = status['controller_parameter']['B']
        K = status['controller_parameter']['K']

        xposture_error = np.concatenate([desired_xpos - xpos, self.orientation_error(desired_xmat, xmat)])
        xvel_error = desired_xvel - xvel

        # 阻抗控制率
        T = np.multiply(B, xvel_error) + np.multiply(K, xposture_error) + contact_force
        solved_acc = desired_xacc + np.dot(np.linalg.inv(M), T)
        F = np.dot(D_x, solved_acc) + CG_x - contact_force
        tau = np.dot(J.T, F)

        return tau


class AdmittanceController:
    def __init__(self, orientation_error):
        self.orientation_error = orientation_error
        self.computed_torque_controller = ComputedTorqueController(orientation_error)
        wn = 20
        damping_ratio = np.sqrt(2)
        kp = wn * wn * np.ones(6, dtype=np.float64)
        kd = 2 * damping_ratio * np.sqrt(kp)
        self.computed_torque_controller_para = {'kp': kp, 'kd': kd}
        self.compliant_xpos = None
        self.compliant_xmat = None
        self.compliant_xvel = None
        self.compliant_xacc = None

    def step(self, status):
        # 状态
        desired_xpos = status['desired_xpos']
        desired_xmat = status['desired_xmat']
        desired_xvel = status['desired_xvel']
        desired_xacc = status['desired_xacc']

        contact_force = status['contact_force']

        # 控制器参数
        M = status['controller_parameter']['M']
        B = status['controller_parameter']['B']
        K = status['controller_parameter']['K']
        timestep = status['timestep']

        # 初始化
        if self.compliant_xpos is None:
            self.compliant_xpos = copy.deepcopy(desired_xpos)
            self.compliant_xmat = copy.deepcopy(desired_xmat)
            self.compliant_xvel = copy.deepcopy(desired_xvel)
            self.compliant_xacc = copy.deepcopy(desired_xacc)
        # xposture_error, xvel_error, w_c dot, related to base frame
        xposture_error = np.concatenate([desired_xpos - self.compliant_xpos,
                                         self.orientation_error(desired_xmat, self.compliant_xmat)])
        xvel_error = desired_xvel - self.compliant_xvel
        T = np.multiply(B, xvel_error) + np.multiply(K, xposture_error) + contact_force
        solved_acc = desired_xacc + np.dot(np.linalg.inv(M), T)  # compliant_xacc
        self.compliant_xacc = solved_acc
        self.compliant_xvel += self.compliant_xacc * timestep
        self.compliant_xpos += self.compliant_xvel[:3] * timestep
        W = np.array([[0., -self.compliant_xacc[5], self.compliant_xacc[4]],
                      [self.compliant_xacc[5], 0., -self.compliant_xacc[3]],
                      [-self.compliant_xacc[4], self.compliant_xacc[3], 0.]])
        self.compliant_xmat += W @ self.compliant_xmat * timestep

        compliant_status = copy.deepcopy(status)
        compliant_status.update(controller_parameter=self.computed_torque_controller_para,
                                desired_xpos=self.compliant_xpos,
                                desired_xmat=self.compliant_xmat,
                                desired_xvel=self.compliant_xvel,
                                desired_xacc=self.compliant_xacc)

        tau = self.computed_torque_controller.step(compliant_status)

        return tau
