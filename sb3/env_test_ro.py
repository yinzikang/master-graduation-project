#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""用于测试自定义mujoco环境的bug

Write detailed description here

Write typical usage example here

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
3/8/23 4:12 PM   yinzikang      1.0         None
"""

from module.jk5_env_v5 import Jk5StickRobotWithController
# from module.jk5_env_v6 import Jk5StickRobotWithController
from module.env_kwargs import load_env_kwargs
import matplotlib.pyplot as plt
import numpy as np

test_times = 2
plot_flag = True

for i in range(test_times):
    buffer = dict()
    rbt_kwargs, rbt_controller_kwargs, rl_kwargs = load_env_kwargs('cabinet drawer open with plan')
    env = Jk5StickRobotWithController(**rbt_controller_kwargs)
    env.reset()
    for status_name in env.status_list:
        buffer[status_name] = [env.status[status_name]]
    for _ in range(rbt_controller_kwargs['step_num']):
        env.step()
        # env.render(pause_start=True)
        # print(env.status["qpos"]*180/np.pi)
        for status_name in env.status_list:
            buffer[status_name].append(env.status[status_name])
    for status_name in env.status_list:
        buffer[status_name] = np.array(buffer[status_name])

    if plot_flag:
        fig_title = rbt_controller_kwargs['controller'].__name__ + ' '

        i = 0

        i += 1
        plt.figure(i)
        plt.plot(buffer["xpos"])
        plt.plot(buffer["desired_xpos"])
        plt.legend(['x', 'y', 'z', 'dx', 'dy', 'dz'])
        plt.title(fig_title + 'xpos')
        plt.grid()

        i += 1
        plt.figure(i)
        plt.plot((buffer["xpos"] - buffer["desired_xpos"]))
        plt.legend(['x', 'y', 'z'])
        plt.title(fig_title + 'xpos error')
        plt.grid()

        i += 1
        plt.figure(i)
        plt.plot(buffer["xvel"][:, :3])
        plt.plot(buffer["desired_xvel"][:, :3])
        plt.legend(['x', 'y', 'z', 'dx', 'dy', 'dz'])
        plt.title(fig_title + 'xpos_vel')
        plt.grid()

        i += 1
        plt.figure(i)
        plt.plot(np.diff(buffer["xvel"][:, :2], axis=0))
        plt.plot(np.diff(buffer["desired_xvel"][:, :2], axis=0))
        # plt.legend(['x', 'y', 'z', 'dx', 'dy', 'dz'])
        plt.legend(['x', 'y', 'dx', 'dy'])
        plt.title(fig_title + 'xpos_acc')
        plt.grid()

        # i += 1
        # plt.figure(i)
        # plt.plot(buffer["xquat"])
        # plt.plot(buffer["desired_xquat"])
        # plt.legend(['x', 'y', 'z', 'w', 'dx', 'dy', 'dz', 'dw'])
        # plt.title(fig_title + 'xquat')
        # plt.grid()

        # i += 1
        # plt.figure(i)
        # plt.plot(buffer["xvel"][:, 3:])
        # plt.plot(buffer["desired_xvel"][:, 3:])
        # plt.legend(['x', 'y', 'z', 'dx', 'dy', 'dz'])
        # plt.title(fig_title + 'xmat_vel')
        # plt.grid()

        # i += 1
        # plt.figure(i)
        # orientation_error_buffer = []
        # for j in range(len(buffer["xquat"])):
        #     orientation_error_buffer.append(orientation_error_quat_with_mat(buffer["desired_xmat"][j], buffer["xmat"][j]))
        # plt.plot(orientation_error_buffer)
        # plt.legend(['x', 'y', 'z'])
        # plt.title(fig_title + 'orientation_error')
        # plt.grid()

        # i += 1
        # plt.figure(i)
        # plt.plot(buffer["qpos"])
        # plt.legend(['j1', 'j2', 'j3', 'j4', 'j5', 'j6'])
        # plt.title(fig_title + 'qpos')
        # plt.grid()
        #
        # i += 1
        # plt.figure(i)
        # plt.plot(buffer["qvel"])
        # plt.legend(['j1', 'j2', 'j3', 'j4', 'j5', 'j6'])
        # plt.title(fig_title + 'qvel')
        # plt.grid()

        # i += 1
        # plt.figure(i)
        # plt.plot((np.array(xpos_buffer) - np.array(desired_xpos_buffer))[1:, :] /
        #          np.array(contact_force_buffer)[1:, :3])
        # plt.legend(['x', 'y', 'z'])
        # plt.title(fig_title + '1/stiffness')
        # plt.grid()

        i += 1
        plt.figure(i)
        plt.plot(buffer["contact_force"])
        plt.legend(['x', 'y', 'z', 'rx', 'ry', 'rz'])
        plt.title(fig_title + 'contact_force')
        plt.grid()

        # i += 1
        # plt.figure(i)
        # plt.plot(buffer["touch_force"])
        # # plt.legend(['x', 'y', 'z', 'rx', 'ry', 'rz'])
        # plt.title(fig_title + 'touch_force')
        # plt.grid()
        #
        # i += 1
        # plt.figure(i)
        # plt.plot(buffer["tau"])
        # plt.legend(['j1', 'j2', 'j3', 'j4', 'j5', 'j6'])
        # plt.title(fig_title + 'tau')
        # plt.grid()

        plt.show()
