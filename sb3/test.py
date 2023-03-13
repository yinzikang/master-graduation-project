#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Write target here

Write detailed description here

Write typical usage example here

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
3/2/23 10:29 AM   yinzikang      1.0         None
"""
from utils.mujoco_viewer.mujoco_viewer import MujocoViewer
from sb3_contrib import QRDQN
import mujoco


# policy_kwargs = dict(n_quantiles=50)
# model = QRDQN("MlpPolicy", "CartPole-v1", policy_kwargs=policy_kwargs, verbose=1)
# model.learn(total_timesteps=10000, log_interval=4)
# model.save("qrdqn_cartpole")


model = mujoco.MjModel.from_xml_path(filename='robot/jk5_door_v1.xml')
data = mujoco.MjData(model)
viewer = MujocoViewer(model, data,mode='window')

while True:
    viewer.render()
