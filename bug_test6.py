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
from gym_custom.envs.transformations import quaternion_from_matrix, random_rotation_matrix
# delta_x = numpy.random.random(3)
# k = numpy.random.random(3)
# K = numpy.diag(k)
# print(numpy.multiply(k, delta_x))
# print(numpy.dot(K, delta_x))
# print(K @ delta_x)

R = random_rotation_matrix()[:3,:3]
k = numpy.random.random(3)
K = numpy.diag(k)
print(K)
print(R@numpy.linalg.inv(R))
print(R@K@numpy.linalg.inv(R))
