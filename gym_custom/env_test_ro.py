#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""用于测试自定义mujoco环境的bug

Write detailed description here

Write typical usage example here

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
3/8/23 4:12 PM   yinzikang      1.0         None
"""

from gym_custom.envs.jk5_env_v8 import Jk5StickRobotWithController
from gym_custom.envs.env_kwargs import env_kwargs
from eval_everything import eval_robot
import numpy as np

# test_name = 'cabinet surface with plan v7'
# test_name = 'cabinet drawer open with plan'
test_name = 'cabinet door open with plan test'
# test_name = 'cabinet door close with plan test'
test_times = 1
plot_flag = True

for i in range(test_times):
    buffer = dict()
    rbt_kwargs, rbt_controller_kwargs, rl_kwargs = env_kwargs(test_name)
    env = Jk5StickRobotWithController(**rbt_controller_kwargs)
    env.reset()
    for status_name in env.status_list:
        buffer[status_name] = [env.status[status_name]]
    for _ in range(rbt_controller_kwargs['step_num']):
        env.step()
        # env.render(pause_start=True)
        for status_name in env.status_list:
            buffer[status_name].append(env.status[status_name])
    for status_name in env.status_list:
        buffer[status_name] = np.array(buffer[status_name])

    if plot_flag:
        eval_robot(buffer, plot_flag)
