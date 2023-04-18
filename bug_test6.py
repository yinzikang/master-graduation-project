#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Write target here

Write detailed description here

Write typical usage example here

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
4/16/23 3:29 PM   yinzikang      1.0         None
"""
import numpy
import numpy as np

from gym_custom.envs.transformations import quaternion_from_matrix, quaternion_matrix, random_rotation_matrix, \
    quaternion_multiply, quaternion_conjugate, quaternion_inverse
from gym_custom.envs.controller import mat33_to_quat

# delta_x = numpy.random.random(3)
# k = numpy.random.random(3)
# K = numpy.diag(k)
# print(numpy.multiply(k, delta_x))
# print(numpy.dot(K, delta_x))
# print(K @ delta_x)

R44 = random_rotation_matrix()
# R44 = numpy.array([[0, -1, 0, 0],
#                    [-1, 0, 0, 0],
#                    [0, 0, -1, 0],
#                    [0, 0, 0, 1]])
R33 = R44[:3, :3]
Q = quaternion_from_matrix(R44)
# k = numpy.random.random(3)
k = numpy.array([1, 2, 3])
K = numpy.concatenate((k, numpy.array([0])))
K1 = numpy.diag(k)
K2 = numpy.eye(4)
K2[:3, :3] = K1.copy()

# 本身
# print(quaternion_matrix(mat33_to_quat(K1)))  # 输出与K1不一致，因为输入的矩阵不满足旋转矩阵格式

# 向量旋转
# ## 以下两个结果一致
# print(R33 @ k)
# print(R33 @ numpy.transpose(k))
# ## 以下四个结果一致
# print(quaternion_multiply(quaternion_multiply(Q, K), quaternion_inverse(Q)))
# print(quaternion_multiply(quaternion_multiply(Q, K), quaternion_conjugate(Q)))
# print(quaternion_multiply(Q, quaternion_multiply(K, quaternion_inverse(Q))))
# print(quaternion_multiply(Q, quaternion_multiply(K, quaternion_conjugate(Q))))

# 矩阵旋转
print(K1)
print(R33 @ K1)
l = np.array([[Q[3], -Q[2], Q[1], Q[0]],
              [Q[2], Q[3], -Q[0], Q[1]],
              [-Q[1], Q[0], Q[3], Q[2]],
              [-Q[0], -Q[1], -Q[2], Q[3]]])
print(l @ K2)
# print(quaternion_matrix(quaternion_multiply(Q, mat33_to_quat(K1)))[:3,:3])

# 矩阵多次旋转
# print(k)
# print(R33 @ K1 @ numpy.transpose(R33))  # 需要对角矩阵作为输入
# # 转换出来结果是四维向量？？？没法用四元数吧
# # 只能用旋转矩阵形式，动力学方程里面全是矩阵，KX都是矩阵，
# print(quaternion_matrix(quaternion_multiply(quaternion_multiply(Q, mat33_to_quat(K1)), quaternion_inverse(Q)))[:3,:3])
