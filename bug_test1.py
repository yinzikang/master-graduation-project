#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Write target here

Write detailed description here

Write typical usage example here

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
3/2/23 10:29 AM   yinzikang      1.0         None
"""
import numpy as np

from gym_custom.utils.mujoco_viewer.mujoco_viewer import MujocoViewer
from sb3_contrib import QRDQN
import mujoco
from gym_custom.envs.jk5_env_v5 import Jk5StickRobot
from gym_custom.envs.controller import *
from gym_custom.envs.transformations import quaternion_from_matrix, random_rotation_matrix, quaternion_from_matrix, \
    quaternion_multiply
import matplotlib.pyplot as plt

# policy_kwargs = dict(n_quantiles=50)
# model = QRDQN("MlpPolicy", "CartPole-v1", policy_kwargs=policy_kwargs, verbose=1)
# model.learn(total_timesteps=10000, log_interval=4)
# model.save("qrdqn_cartpole")
# 姿态误差测试
# for _ in range(10):
#     mat1 = random_rotation_matrix()
#     mat2 = random_rotation_matrix()
#     quat1 = quaternion_from_matrix(mat1)
#     quat2 = quaternion_from_matrix(mat2)
#     # print(quat1)
#     # print(quat2)
#     o_a_m = orientation_error_axis_angle_with_mat(mat2[:3, :3], mat1[:3, :3])
#     o_q_m = orientation_error_quat_with_mat(mat2[:3, :3], mat1[:3, :3])
#     o_q_q = orientation_error_quat_with_quat(quat2, quat1)
#     # print(o_a_m)
#     print(o_q_m)
#     print(o_q_q)
#     # if max(abs(o_q_m - o_q_q)) > 0.001:
#     #     print(quat1)
#     #     print(quat2)
#     #     print(o_q_m)
#     #     print(o_q_q)
#     #     print(o_q_m - o_q_q)

# for _ in range(10):
#     mat1 = random_rotation_matrix()
#     mat2 = random_rotation_matrix()
#     quat1 = quaternion_from_matrix(mat1)
#     quat2 = quaternion_from_matrix(mat2)
#     # mat计算
#     mat44 = np.eye(4)
#     mat44[:3, :3] = np.linalg.inv(mat2[:3, :3]) @ mat1[:3, :3]
#     quat_from_mat = quaternion_from_matrix(mat44)
#     if quat_from_mat[3] < 0:
#         quat_from_mat = - quat_from_mat
#     o_q_m = mat2[:3, :3] @ quat_from_mat[:3]
#     # quat计算
#     quat_from_quat = quaternion_multiply(quaternion_inverse(quat2), quat1)
#     if quat_from_quat[3] < 0:
#         quat_from_quat = - quat_from_quat
#     quat_from_quat[3] = 0
#     o_q_q = quaternion_multiply(quaternion_multiply(quat2, quat_from_quat), quaternion_inverse(quat2))
#     o_q_q = o_q_q[:3]
#     # print(quat_from_mat)
#     # print(quat_from_quat)
#     print(o_q_m)
#     print(o_q_q)

# pos = np.array([1, 2, 3])
# pos4 = np.array([1, 2, 3, 0])
# mat = random_rotation_matrix()
# quat = quaternion_from_matrix(mat)
# print(mat[:3, :3] @ pos)
# print(quaternion_multiply(quat, quaternion_multiply(pos4, quaternion_inverse(quat)))[:3])
#
#
# quat_i1 = quaternion_from_matrix(np.linalg.inv(mat))
# quat_i2 = quaternion_inverse(quat)
# print(quat)
# print(quat_i1)
# print(quat_i2)

# 积分测试
compliant_xvel_w = np.array([0, 0, 0, 0, 0, 5], dtype=np.float64)
timestep = 0.01
desired_xmat = np.array([[0, -1, 0],
                         [-1, 0, 0],
                         [0, 0, -1]], dtype=np.float64)
# desired_xmat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
# desired_xmat = np.array([[0, -1, 0],
#                          [1, 0, 0],
#                          [0, 0, 1]], dtype=np.float64)
desired_xquat = mat33_to_quat(desired_xmat)
compliant_xmat = desired_xmat.copy()
compliant_xquat = mat33_to_quat(compliant_xmat)

W = np.array([[0., -compliant_xvel_w[5], compliant_xvel_w[4]],
              [compliant_xvel_w[5], 0., -compliant_xvel_w[3]],
              [-compliant_xvel_w[4], compliant_xvel_w[3], 0.]])
compliant_xmat_1 = compliant_xmat + W @ compliant_xmat * timestep
U, S, VT = np.linalg.svd(compliant_xmat_1)  # 旋转矩阵正交化防止矩阵蠕变
compliant_xmat_1 = U @ VT

compliant_xvel_l = np.linalg.inv(compliant_xmat) @ compliant_xvel_w[3:]
W = np.array([[0., compliant_xvel_l[2], -compliant_xvel_l[1], compliant_xvel_l[0]],
              [-compliant_xvel_l[2], 0., compliant_xvel_l[0], compliant_xvel_l[1]],
              [compliant_xvel_l[1], -compliant_xvel_l[0], 0., compliant_xvel_l[2]],
              [-compliant_xvel_l[0], -compliant_xvel_l[1], -compliant_xvel_l[2], 0.]])
compliant_xquat_1 = compliant_xquat + W @ compliant_xquat * timestep / 2.
if compliant_xquat[3] < 0:
    compliant_xquat[3] = -compliant_xquat[3]
compliant_xquat = compliant_xquat / np.linalg.norm(compliant_xquat)  # 四元数单位化防止矩阵蠕变
w = np.concatenate((compliant_xvel_w[3:], np.array([0])), axis=0)
compliant_xquat_2 = compliant_xquat + quaternion_multiply(w, compliant_xquat) * timestep / 2.

#
# print(compliant_xquat_dot1)
# print(compliant_xquat_dot2)
print('______')

print(mat33_to_quat(compliant_xmat_1))
print(compliant_xquat_1)
print(compliant_xquat_2)
print('-----------')
print(compliant_xmat_1)
print(quaternion_matrix(compliant_xquat_1)[:3, :3])
print(quaternion_matrix(compliant_xquat_2)[:3, :3])
# print('-----------')
# print(orientation_error_quat_with_mat(desired_xmat, compliant_xmat_1))
# print(orientation_error_quat_with_quat(desired_xquat, compliant_xquat_1))
# print(orientation_error_quat_with_mat(desired_xmat, compliant_xmat_1) -
#       orientation_error_quat_with_quat(desired_xquat, compliant_xquat_1))
