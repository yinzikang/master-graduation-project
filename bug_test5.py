#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Write target here

Write detailed description here

Write typical usage example here

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
4/15/23 9:14 PM   yinzikang      1.0         None
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

# desired_xmat = np.array([[0, -1, 0],[-1, 0, 0],[0, 0, -1]], dtype=np.float64)
# desired_xmat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
# desired_xmat = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
desired_xmat = random_rotation_matrix()[:3,:3]

desired_xquat = mat33_to_quat(desired_xmat)
compliant_xmat = desired_xmat.copy()
compliant_xquat = mat33_to_quat(compliant_xmat)
compliant_xvel_w = np.array([1, 2, 5], dtype=np.float64)
compliant_xvel_l = np.linalg.inv(compliant_xmat) @ compliant_xvel_w
timestep = 0.01

W1 = np.array([[0., -compliant_xvel_w[2], compliant_xvel_w[1], compliant_xvel_w[0]],
               [compliant_xvel_w[2], 0., -compliant_xvel_w[0], compliant_xvel_w[1]],
               [-compliant_xvel_w[1], compliant_xvel_w[0], 0., compliant_xvel_w[2]],
               [-compliant_xvel_w[0], -compliant_xvel_w[1], -compliant_xvel_w[2], 0.]])
print(W1 @ compliant_xquat)

W2 = np.array([[0., compliant_xvel_l[2], -compliant_xvel_l[1], compliant_xvel_l[0]],
               [-compliant_xvel_l[2], 0., compliant_xvel_l[0], compliant_xvel_l[1]],
               [compliant_xvel_l[1], -compliant_xvel_l[0], 0., compliant_xvel_l[2]],
               [-compliant_xvel_l[0], -compliant_xvel_l[1], -compliant_xvel_l[2], 0.]])
print(W2 @ compliant_xquat)

w = np.concatenate((compliant_xvel_w, np.array([0])), axis=0)
print(quaternion_multiply(w, compliant_xquat))
