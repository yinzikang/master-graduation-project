#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""用于测试自定义mujoco环境的bug

Write detailed description here

Write typical usage example here

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
3/8/23 4:12 PM   yinzikang      1.0         None
"""

from module.jk5_env_v3 import TrainEnv, load_env_kwargs

env_kwargs = load_env_kwargs('desk')
env = TrainEnv(**env_kwargs)

obs = env.reset()
for i in range(4 * env_kwargs['rl_frequency']):
    action = env.action_space.sample()
    obs, rewards, dones, info = env.step(action)
    env.render(pause_start=True)
