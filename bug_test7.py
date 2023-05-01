#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""不是用来测bug的，是为了测机器人与橱柜的相对位置使得机器人好运行一些

Write detailed description here

Write typical usage example here

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
5/1/23 5:04 PM   yinzikang      1.0         None
"""

from gym_custom.envs.jk5_env_v8 import Jk5StickRobotWithController
from gym_custom.envs.env_kwargs import env_kwargs
from gym_custom.envs.controller import trajectory_planning_circle
from eval_everything import eval_robot
import numpy as np

robot_frequency = 500
time_whole = 4
time_acceleration = 0.5
step_num = robot_frequency * time_whole

# cabinet_pos = np.array([0.8, -0.1135, 0.2])
cabinet_pos = np.array([0.8, -0.2, 0.3])
r = np.sqrt(0.34 ** 2 + 0.025 ** 2)
center = cabinet_pos + np.array([-0.2 + 0.0075, -0.19, 0.22])
rbt_tool = np.array([-0.011, -0.004, 0])
# xpos_init, xmat_init = rbt_tool + center + np.array([-0.025, 0.34, 0]), np.array([0, 0, 1, 0, -1, 0, 1, 0, 0])
# xpos_end, xmat_end = rbt_tool + center + np.array([-0.34, -0.025, 0]), np.array([0, 0, 1, 0, -1, 0, 1, 0, 0])
xpos_init, xmat_init = rbt_tool + center + np.array([-0.34, -0.025, 0]), np.array([0, 0, 1, 0, -1, 0, 1, 0, 0])
xpos_end, xmat_end = rbt_tool + center + np.array([-0.025, 0.34, 0]), np.array([0, 0, 1, 0, -1, 0, 1, 0, 0])
xpos_mid = rbt_tool + center + np.array([-r / np.sqrt(2), r / np.sqrt(2), 0])
desired_xposture_list, desired_xvel_list, desired_xacc_list = trajectory_planning_circle(xpos_init, xmat_init,
                                                                                         xpos_end, xmat_end,
                                                                                         xpos_mid,
                                                                                         time_whole, time_acceleration,
                                                                                         robot_frequency)

test_name = 'cabinet door close with plan test'
buffer = dict()
_, rbt_controller_kwargs, _ = env_kwargs(test_name)
env1 = Jk5StickRobotWithController(**rbt_controller_kwargs)

qpos_init_list = env1.inverse_kinematics(env1.qpos_init_list, xpos_init - rbt_tool, xmat_init)
print(qpos_init_list)
rbt_controller_kwargs['qpos_init_list'] = qpos_init_list
rbt_controller_kwargs['desired_xposture_list'] = desired_xposture_list
rbt_controller_kwargs['desired_xvel_list'] = desired_xvel_list
rbt_controller_kwargs['desired_xacc_list'] = desired_xacc_list
env2 = Jk5StickRobotWithController(**rbt_controller_kwargs)
env2.reset()

for status_name in env2.status_list:
    buffer[status_name] = [env2.status[status_name]]
for _ in range(rbt_controller_kwargs['step_num']):
    # env2.data.qpos[7] = np.pi/2
    env2.step()
    env2.render(pause_start=True)
    for status_name in env2.status_list:
        buffer[status_name].append(env2.status[status_name])
for status_name in env2.status_list:
    buffer[status_name] = np.array(buffer[status_name])

eval_robot(buffer, True)
