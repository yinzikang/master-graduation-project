#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Write target here

Write detailed description here

Write typical usage example here

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
3/28/23 9:59 PM   yinzikang      1.0         None
"""
import numpy as np
from module.transformations import random_rotation_matrix, random_quaternion

mat = random_rotation_matrix()
quat = random_quaternion()
print(mat)
print(quat)
print(np.linalg.norm(quat))
