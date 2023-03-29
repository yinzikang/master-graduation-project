#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""机器人轨迹规划、控制器相关

轨迹生成
姿态误差
多种控制器
四元数格式为xyzw

Write typical usage example here

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
3/16/23 10:22 PM   yinzikang      1.0         None
"""
import copy
import PyKDL as kdl
import numpy as np
from module.transformations import quaternion_from_matrix, quaternion_matrix, quaternion_conjugate, quaternion_multiply, \
    quaternion_inverse
import matplotlib.pyplot as plt


def to_kdl_qpos(joint_num, qpos):
    kdl_qpos = kdl.JntArray(joint_num)
    for i in range(joint_num):
        kdl_qpos[i] = qpos[i]
    return kdl_qpos


def to_kdl_xpos(xpos):
    return kdl.Vector(xpos[0], xpos[1], xpos[2])


def to_kdl_xmat(xmat):
    xmat = xmat.reshape(-1)
    return kdl.Rotation(xmat[0], xmat[1], xmat[2],
                        xmat[3], xmat[4], xmat[5],
                        xmat[6], xmat[7], xmat[8])


def to_kdl_frame(xpos, xmat):
    return kdl.Frame(to_kdl_xmat(xmat), to_kdl_xpos(xpos))


def to_numpy_qpos(joint_num, kdl_qpos_list):
    qpos_list = np.empty(joint_num)
    for i in range(joint_num):
        qpos_list[i] = kdl_qpos_list[i]
    return qpos_list


def to_numpy_xpos(kdl_xpos):
    return np.array([kdl_xpos[0], kdl_xpos[1], kdl_xpos[2]])


def to_numpy_xmat(kdl_xmat):
    return np.array([kdl_xmat[0, 0], kdl_xmat[0, 1], kdl_xmat[0, 2],
                     kdl_xmat[1, 0], kdl_xmat[1, 1], kdl_xmat[1, 2],
                     kdl_xmat[2, 0], kdl_xmat[2, 1], kdl_xmat[2, 2]]).reshape(3, 3)


def to_numpy_frame(kdl_frame):
    return to_numpy_xmat(kdl_frame.p), to_numpy_xmat(kdl_frame.M)


def mat33_to_quat(xmat33):
    xmat44 = np.eye(4)
    xmat44[:3, :3] = xmat33
    xquat = quaternion_from_matrix(xmat44)
    if xquat[3] < 0:
        xquat = - xquat

    return xquat


def mat44_to_quat(xmat44):
    xquat = quaternion_from_matrix(xmat44)
    if xquat[3] < 0:
        xquat = - xquat

    return xquat


def quat_to_mat_33(xquat):
    return quaternion_matrix(xquat)[:3, :3]


def get_trajectory_para(time_whole, time_accel, step_num):
    """
    计算形式为三次多项式的速度的参数，使得速度积分值为1，返回每个时刻的a、v、p
    该a、v、p不具有物理意义，等待比例缩放
    :param time_whole: 整个运动时间
    :param time_accel: 加速区间时间
    :param step_num: 总步数
    :return:
    """
    if time_accel == 0:
        pos = np.linspace(0, 1, step_num, dtype=np.float32).reshape((-1, 1))
        vel = 1. / time_whole * np.ones((step_num, 1), dtype=np.float32)
        acc = np.zeros((step_num, 1), dtype=np.float32)
    else:
        para = 6. / (time_accel ** 3) / (time_accel - time_whole)
        time = np.linspace(0, time_whole, step_num, dtype=np.float32).reshape((-1, 1))
        acc = np.empty_like(time)
        vel = np.empty_like(time)
        pos = np.empty_like(time)

        current_step = 0
        split_step1 = int(time_accel / time_whole * step_num)  # 加速、匀速分割
        split_step2 = int((time_whole - time_accel) / time_whole * step_num)  # 匀速、减速分割
        for x in time[:split_step1]:
            acc[current_step] = para * (x ** 2 - time_accel * x)
            vel[current_step] = para * (1. / 3 * x ** 3 - 1. / 2 * time_accel * x ** 2)
            pos[current_step] = para * (1. / 12 * x ** 4 - 1. / 6 * time_accel * x ** 3)
            current_step += 1
        for x in time[split_step1:split_step2]:
            acc[current_step] = 0
            vel[current_step] = para * (-1. / 6) * time_accel ** 3
            pos[current_step] = para * (-1. / 12) * time_accel ** 4 + \
                                para * (-1. / 6) * time_accel ** 3 * (x - time_accel)
            current_step += 1
        for x in time[split_step2:]:
            acc[current_step] = - para * ((time_whole - x) ** 2 - time_accel * (time_whole - x))
            vel[current_step] = para * (1. / 3 * (time_whole - x) ** 3 - 1. / 2 * time_accel * (time_whole - x) ** 2)
            pos[current_step] = 2 * para * (-1. / 12) * time_accel ** 4 + \
                                para * (-1. / 6) * time_accel ** 3 * (time_whole - 2 * time_accel) - \
                                para * (1. / 12 * (time_whole - x) ** 4 - 1. / 6 * time_accel * (time_whole - x) ** 3)
            current_step += 1

    return acc, vel, pos


def position_interpolation_line(init_xpos, end_xpos, time_whole, time_acceleration, control_frequency):
    """
    返回step_num*3
    :param init_xpos:
    :param end_xpos:
    :param time_whole:
    :param time_acceleration:
    :param control_frequency:
    :return:
    """
    step_num = int(time_whole * control_frequency)
    error = end_xpos - init_xpos  # 直线运动，误差为位置的差距
    a, v, p = get_trajectory_para(time_whole, time_acceleration, step_num)
    acc_buffer = a * error
    vel_buffer = v * error
    pos_buffer = p * error + init_xpos

    return pos_buffer, vel_buffer, acc_buffer


def position_interpolation_circle(init_xpos, end_xpos, mid_xpos, time_whole, time_acceleration, control_frequency):
    # 求旋转平面与法向量
    P1 = mid_xpos - init_xpos  # 轨迹前半段
    P2 = end_xpos - mid_xpos  # 轨迹后半段
    direction = np.cross(P1, P2)  # 轨迹前半段插乘后半段得到旋转平面法向量，并利用右手定则确定旋转方向
    # 创建新坐标系(U,V,W,mid_frame)，即该坐标系原点位于中间点
    U = -P1 / np.linalg.norm(P1)  # U方向单位向量，方向为mid到init
    W = direction / np.linalg.norm(direction)  # W方向单位向量
    V = np.cross(W, U)  # V方向单位向量，方向为mid到end
    # 两个坐标系之间的完整关系
    pos = mid_xpos
    mat = np.concatenate((U, V, W)).reshape((3, 3), order="F")

    # 求旋转平面上三点坐标以及圆弧的圆心与半径
    P1_1 = (np.linalg.inv(mat) @ (init_xpos - pos).reshape((3, 1))).reshape(-1)  # UVW中起点坐标,一定为(P1P2,0,0)
    P2_1 = (np.linalg.inv(mat) @ (mid_xpos - pos).reshape((3, 1))).reshape(-1)  # UVW中中点坐标，一定为(0,0,0)
    P3_1 = (np.linalg.inv(mat) @ (end_xpos - pos).reshape((3, 1))).reshape(-1)  # UVW中终点坐标,W一定为0
    # 求圆心与半径,原点位于P1_1与P2_1中垂线上，因此U为定值，此外W为定值，设方程求解其位置
    circle_origin = np.array([P1_1[0] / 2, (P3_1[0] ** 2 + P3_1[1] ** 2 - P1_1[0] * P3_1[0]) / 2 / P3_1[1], 0])
    radius = np.sqrt(circle_origin[0] ** 2 + circle_origin[1] ** 2)

    # 求旋转平面上三点对应旋转角度,np.arccos返回值处于0-pi
    # 限制条件：角度单调变化，中间角度位于两者之间；总变化范围不超过2pi
    init_angle = np.arccos((P1_1[0] - circle_origin[0]) / radius) if P1_1[1] >= circle_origin[1] \
        else -np.arccos((P1_1[0] - circle_origin[0]) / radius)
    mid_angle = np.arccos((P2_1[0] - circle_origin[0]) / radius) if P2_1[1] >= circle_origin[1] \
        else -np.arccos((P2_1[0] - circle_origin[0]) / radius)
    end_angle = np.arccos((P3_1[0] - circle_origin[0]) / radius) if P3_1[1] >= circle_origin[1] \
        else -np.arccos((P3_1[0] - circle_origin[0]) / radius)
    mid_angle_possible_list = [mid_angle - 2 * np.pi, mid_angle, mid_angle + 2 * np.pi]
    end_angle_possible_list = [end_angle - 2 * np.pi, end_angle, end_angle + 2 * np.pi]
    for i in mid_angle_possible_list:
        for j in end_angle_possible_list:
            if min(init_angle, j) <= i <= max(init_angle, j):
                if abs(j - init_angle) < 2 * np.pi:
                    mid_angle = i
                    end_angle = j
                    print(init_angle, mid_angle, end_angle)

    step_num = int(time_whole * control_frequency)
    error = end_angle - init_angle  # 圆周运动，误差为角度的差异
    # 获得新坐标系上的角加速度、角速度、角度，求解线加速度（向心加速度、切向加速度），线速度，位置
    a, w, p = get_trajectory_para(time_whole, time_acceleration, step_num)
    theta = p * error + init_angle
    pos_buffer = radius * np.concatenate([np.cos(theta), np.sin(theta), np.zeros_like(theta)], axis=1) + circle_origin
    vel_buffer = w * radius * np.concatenate([-np.sin(theta), np.cos(theta), np.zeros_like(theta)], axis=1)
    acc_buffer1 = a * radius * np.concatenate([-np.sin(theta), np.cos(theta), np.zeros_like(theta)], axis=1)
    acc_buffer2 = w ** 2 * np.concatenate([-np.cos(theta), -np.sin(theta), np.zeros_like(theta)], axis=1)
    # 转换到空间中
    pos_buffer = (mat @ pos_buffer.transpose()).transpose() + pos
    vel_buffer = (mat @ vel_buffer.transpose()).transpose()
    acc_buffer = (mat @ (acc_buffer1 + acc_buffer2).transpose()).transpose()

    return pos_buffer, vel_buffer, acc_buffer


def orientation_interpolation(init_xmat, end_xmat, time_whole, time_acceleration, control_frequency):
    step_num = int(time_whole * control_frequency)
    init_xquat = mat33_to_quat(init_xmat.reshape((3, 3)))
    end_xquat = mat33_to_quat(end_xmat.reshape((3, 3)))
    cos_theta = init_xquat @ end_xquat
    if cos_theta < 0:
        end_xquat = - end_xquat
        cos_theta = - cos_theta
    sin_theta = np.sqrt(1. - cos_theta ** 2)
    theta = np.arctan2(sin_theta, cos_theta)  # 返回以弧度为单位的角度，范围为[-pi,+pi]

    if abs(theta) < 1e-5:
        mat_buffer = np.expand_dims(end_xmat, 0).repeat(step_num, axis=0)
        quat = np.expand_dims(end_xquat, 0).repeat(step_num, axis=0)
        vel_buffer = np.zeros((step_num, 3))
        acc_buffer = np.zeros((step_num, 3))
    else:
        a, v, p = get_trajectory_para(time_whole, time_acceleration, step_num)
        quat = np.sin((1 - p) * theta) / sin_theta * init_xquat + np.sin(
            p * theta) / sin_theta * end_xquat  # step_num*4
        quat_d = theta * (
                - np.cos((1 - p) * theta) / sin_theta * init_xquat + np.cos(p * theta) / sin_theta * end_xquat) * v
        quat_dd = - theta ** 2 * quat * v * v + \
                  theta * (- np.cos((1 - p) * theta) / sin_theta * init_xquat + np.cos(
            p * theta) / sin_theta * end_xquat) * a
        mat_buffer = np.empty((step_num, 9))
        vel_buffer = np.empty((step_num, 3))
        acc_buffer = np.empty((step_num, 3))
        for i in range(quat.shape[0]):
            mat_buffer[i, :] = quat_to_mat_33(quat[i, :]).reshape(-1)
            vel_buffer[i, :] = (2 * quaternion_multiply(quat_d[i, :], quaternion_conjugate(quat[i, :])))[:3]
            acc_buffer[i, :] = 2 * (quaternion_multiply(quat_dd[i, :], quaternion_conjugate(quat[i, :])) +
                                    quaternion_multiply(
                                        quaternion_multiply(quat_d[i, :], quaternion_conjugate(quat[i, :])),
                                        quaternion_multiply(quat_d[i, :],
                                                            quaternion_conjugate(quat[i, :]))))[:3]

    return mat_buffer, quat, vel_buffer, acc_buffer


def trajectory_planning_line(init_xpos, init_xmat, end_xpos, end_xmat, time_whole, time_acceleration,
                             control_frequency):
    xpos, xvel1, xacc1 = position_interpolation_line(init_xpos, end_xpos, time_whole, time_acceleration,
                                                     control_frequency)
    xmat, xquat, xvel2, xacc2 = orientation_interpolation(init_xmat, end_xmat, time_whole, time_acceleration,
                                                          control_frequency)
    xposture = np.concatenate((xpos, xmat, xquat), axis=1)
    xvel = np.concatenate((xvel1, xvel2), axis=1)
    xacc = np.concatenate((xacc1, xacc2), axis=1)
    return xposture, xvel, xacc


def trajectory_planning_circle(init_xpos, init_xmat, end_xpos, end_xmat, mid_xpos, time_whole, time_acceleration,
                               control_frequency):
    xpos, xvel1, xacc1 = position_interpolation_circle(init_xpos, end_xpos, mid_xpos, time_whole, time_acceleration,
                                                       control_frequency)
    xmat, xquat, xvel2, xacc2 = orientation_interpolation(init_xmat, end_xmat, time_whole, time_acceleration,
                                                          control_frequency)
    xposture = np.concatenate((xpos, xmat), axis=1)
    xvel = np.concatenate((xvel1, xvel2), axis=1)
    xacc = np.concatenate((xacc1, xacc2), axis=1)
    return xposture, xvel, xacc


def orientation_error_axis_angle_with_mat(desired, current):
    rc1 = current[0:3, 0]
    rc2 = current[0:3, 1]
    rc3 = current[0:3, 2]
    rd1 = desired[0:3, 0]
    rd2 = desired[0:3, 1]
    rd3 = desired[0:3, 2]
    error = 0.5 * (np.cross(rc1, rd1) + np.cross(rc2, rd2) + np.cross(rc3, rd3))
    return error


def orientation_error_quat_with_mat(desired, current):
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
    if quat[3] < 0:
        quat = - quat
    error = current @ quat[:3]
    return error


def orientation_error_quat_with_quat(desired, current):
    """
    虚部+实部，前三个当偏差
    转换到当前坐标系
    :param desired:
    :param current:
    :return:
    """
    quat = quaternion_multiply(quaternion_inverse(current), desired)
    if quat[3] < 0:
        quat = - quat
    quat[3] = 0
    error = quaternion_multiply(current, quaternion_multiply(quat, quaternion_inverse(current)))[:3]
    return error


# 迪卡尔空间下计算力矩控制器
class ComputedTorqueController:
    def step(self, status):
        # 状态
        desired_xpos = status['desired_xpos']
        desired_xmat = status['desired_xmat']
        desired_xquat = status['desired_xquat']
        desired_xvel = status['desired_xvel']
        desired_xacc = status['desired_xacc']
        xpos = status['xpos']
        xmat = status['xmat']
        xquat = status['xquat']
        xvel = status['xvel']

        J = status['J']
        D_x = status['D_x']
        CG_x = status['CG_x']
        contact_force = status['contact_force']

        # 控制器参数
        kp = status['controller_parameter']['kp']
        kd = status['controller_parameter']['kd']

        # xposture_error = np.concatenate([desired_xpos - xpos,
        #                                  orientation_error_axis_angle_with_mat(desired_xmat, xmat)])
        # xposture_error = np.concatenate([desired_xpos - xpos,
        #                                  orientation_error_quat_with_mat(desired_xmat, xmat)])
        xposture_error = np.concatenate([desired_xpos - xpos,
                                         orientation_error_quat_with_quat(desired_xquat, xquat)])
        xvel_error = desired_xvel - xvel

        solved_acc = desired_xacc + np.multiply(kd, xvel_error) + np.multiply(kp, xposture_error)
        F = np.dot(D_x, solved_acc) + CG_x - contact_force
        tau = np.dot(J.T, F)

        return tau


# 迪卡尔空间下阻抗控制器
class ImpedanceController:
    def step(self, status):
        # 状态
        desired_xpos = status['desired_xpos']
        desired_xmat = status['desired_xmat']
        desired_xquat = status['desired_xquat']
        desired_xvel = status['desired_xvel']
        desired_xacc = status['desired_xacc']
        xpos = status['xpos']
        xmat = status['xmat']
        xquat = status['xquat']
        xvel = status['xvel']

        J = status['J']
        D_x = status['D_x']
        CG_x = status['CG_x']
        contact_force = status['contact_force']

        # 控制器参数
        M = status['controller_parameter']['M']
        B = status['controller_parameter']['B']
        K = status['controller_parameter']['K']

        # xposture_error = np.concatenate([desired_xpos - xpos,
        #                                  orientation_error_axis_angle_with_mat(desired_xmat, xmat)])
        # xposture_error = np.concatenate([desired_xpos - xpos,
        #                                  orientation_error_quat_with_mat(desired_xmat, xmat)])
        xposture_error = np.concatenate([desired_xpos - xpos,
                                         orientation_error_quat_with_quat(desired_xquat, xquat)])
        xvel_error = desired_xvel - xvel

        # 阻抗控制率
        T = np.multiply(B, xvel_error) + np.multiply(K, xposture_error) + contact_force
        solved_acc = desired_xacc + np.dot(np.linalg.inv(M), T)
        F = np.dot(D_x, solved_acc) + CG_x - contact_force
        tau = np.dot(J.T, F)

        return tau


class AdmittanceController:
    def __init__(self):
        self.computed_torque_controller = ComputedTorqueController()
        # 计算力矩控制器参数
        wn = 20
        damping_ratio = np.sqrt(2)
        kp = wn * wn * np.ones(6, dtype=np.float64)
        kd = 2 * damping_ratio * np.sqrt(kp)
        self.computed_torque_controller_para = {'kp': kp, 'kd': kd}

        self.compliant_xpos = None
        self.compliant_xmat = None
        self.compliant_xquat = None
        self.compliant_xvel = None
        self.compliant_xacc = None

    def step(self, status):
        # 状态
        desired_xpos = status['desired_xpos']
        desired_xmat = status['desired_xmat']
        desired_xquat = status['desired_xquat']
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
            self.compliant_xquat = copy.deepcopy(desired_xquat)
            self.compliant_xvel = copy.deepcopy(desired_xvel)
            self.compliant_xacc = copy.deepcopy(desired_xacc)
        # xposture_error, xvel_error, w_c dot, related to base frame
        # xposture_error = np.concatenate([desired_xpos - self.compliant_xpos,
        #                                  orientation_error_axis_angle_with_mat(desired_xmat, self.compliant_xmat)])
        # xposture_error = np.concatenate([desired_xpos - self.compliant_xpos,
        #                                  orientation_error_quat_with_mat(desired_xmat, self.compliant_xmat)])
        xposture_error = np.concatenate([desired_xpos - self.compliant_xpos,
                                         orientation_error_quat_with_quat(desired_xquat, self.compliant_xquat)])
        xvel_error = desired_xvel - self.compliant_xvel
        T = np.multiply(B, xvel_error) + np.multiply(K, xposture_error) + contact_force
        solved_acc = desired_xacc + np.dot(np.linalg.inv(M), T)  # compliant_xacc
        self.compliant_xacc = solved_acc
        self.compliant_xvel += self.compliant_xacc * timestep
        self.compliant_xpos += self.compliant_xvel[:3] * timestep
        # 旋转矩阵微分与角速度关系
        w_global = self.compliant_xvel[3:]
        W = np.array([[0., -w_global[2], w_global[1]],
                      [w_global[2], 0., -w_global[0]],
                      [-w_global[1], w_global[0], 0.]])
        self.compliant_xmat += W @ self.compliant_xmat * timestep
        U, S, VT = np.linalg.svd(self.compliant_xmat)  # 旋转矩阵正交化防止矩阵蠕变
        self.compliant_xmat = U @ VT
        # 四元数微分与角速度关系，转置为了迎合行向量的四元数
        w_local = np.linalg.inv(self.compliant_xmat) @ self.compliant_xvel[3:]
        W = np.array([[0., w_local[2], -w_local[1], w_local[0]],
                      [-w_local[2], 0., w_local[0], w_local[1]],
                      [w_local[1], -w_local[0], 0., w_local[2]],
                      [-w_local[0], -w_local[1], -w_local[2], 0.]])
        self.compliant_xquat += W @ self.compliant_xquat * timestep / 2.
        if self.compliant_xquat[3] < 0:
            self.compliant_xquat[3] = -self.compliant_xquat[3]
        self.compliant_xquat = self.compliant_xquat / np.linalg.norm(self.compliant_xquat)  # 四元数单位化防止矩阵蠕变

        compliant_status = copy.deepcopy(status)
        compliant_status.update(controller_parameter=self.computed_torque_controller_para,
                                desired_xpos=self.compliant_xpos,
                                desired_xmat=self.compliant_xmat,
                                desired_xquat=self.compliant_xquat,
                                desired_xvel=self.compliant_xvel,
                                desired_xacc=self.compliant_xacc)

        tau = self.computed_torque_controller.step(compliant_status)

        return tau


if __name__ == "__main__":
    time_w = 4
    time_a = 1
    f = 500
    pos0, mat0 = np.array([-0.4, -0.15, 0.565]), np.array([0, -1, 0, -1, 0, 0, 0, 0, -1])
    pos2, mat2 = np.array([-0.4, 0.15, 0.565]), np.array([0, -1, 0, -1, 0, 0, 0, 0, -1])

    # 曲线
    # acc, vel, pos = get_trajectory_para(time_w, time_a, f)
    # plt.figure(1)
    # plt.plot(acc)
    # plt.plot(vel)
    # plt.plot(pos)
    # plt.grid()
    # plt.legend(['a', 'v', 'p'])
    # plt.title('curve')
    # plt.show()

    d_acc1, d_vel1, d_pos1 = position_interpolation_line(pos0, pos2, time_w, time_a, f)
    # d_pos1, d_vel1, d_acc1 = position_interpolation_circle(pos0, pos2, pos1, time_w, time_a, f)
    d_mat2, d_quat2, d_vel2, d_acc2 = orientation_interpolation(mat0, mat2, time_w, time_a, f)

    i = 1

    # 位置
    plt.figure(i)
    plt.plot(d_pos1)
    plt.grid()
    plt.legend(['x', 'y', 'z'])
    plt.title('pos')
    i += 1

    plt.figure(i)
    plt.plot(d_vel1)
    plt.grid()
    plt.legend(['x', 'y', 'z'])
    plt.title('vel1')
    i += 1

    plt.figure(i)
    plt.plot(d_acc1)
    plt.grid()
    plt.legend(['x', 'y', 'z'])
    plt.title('acc1')
    i += 1

    plt.figure(i)
    plt.plot(d_acc1[:, 0])
    plt.plot(d_vel1[:, 0])
    plt.plot(d_pos1[:, 0])
    plt.grid()
    plt.legend(['a', 'v', 'p'])
    plt.title('pos in one direction')
    i += 1

    # 姿态
    plt.figure(i)
    plt.plot(d_mat2)
    plt.grid()
    plt.title('mat')
    i += 1

    plt.figure(i)
    plt.plot(d_quat2[:, 0])
    plt.plot(d_quat2[:, 1])
    plt.plot(d_quat2[:, 2])
    plt.plot(d_quat2[:, 3])
    plt.grid()
    plt.legend(['x', 'y', 'z', 'w'])
    plt.title('quat')
    i += 1

    plt.figure(i)
    plt.plot(d_vel2)
    plt.grid()
    plt.legend(['x', 'y', 'z'])
    plt.title('vel2')
    i += 1

    plt.figure(i)
    plt.plot(d_acc2)
    plt.grid()
    plt.legend(['x', 'y', 'z'])
    plt.title('acc2')
    i += 1

    plt.show()
