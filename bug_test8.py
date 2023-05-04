#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""对比旋转矩阵对向量、姿态的旋转

Write detailed description here

Write typical usage example here

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
5/4/23 11:57 AM   yinzikang      1.0         None
"""
from gym_custom.envs.transformations import random_vector, random_quaternion, \
    quaternion_matrix, quaternion_from_matrix, \
    quaternion_multiply, quaternion_conjugate
import numpy as np

# 初始的向量与姿态
v = random_vector(3)
v_plus = np.concatenate((v, np.zeros(1)))
p = random_quaternion()
m = quaternion_matrix(p)[:3, :3]

# 旋转变换
quat = random_quaternion()
mat = quaternion_matrix(quat)

# 向量旋转结果
v1 = quaternion_multiply(quat, quaternion_multiply(v_plus, quaternion_conjugate(quat)))
v2 = mat[:3, :3] @ v
print(v1[:3] - v2)

# 姿态旋转结果
p1 = quaternion_matrix(quaternion_multiply(quat, p))[:3, :3]
p2 = mat[:3, :3] @ m
print(p1 - p2)
