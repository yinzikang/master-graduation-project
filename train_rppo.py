#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Write target here

Write detailed description here

Write typical usage example here

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
3/29/23 7:50 PM   yinzikang      1.0         None
"""

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.evaluation import evaluate_policy
from module.jk5_env_v6 import TrainEnvVariableStiffness as TrainEnv
from module.env_kwargs import load_env_kwargs

_, _, env_kwargs = load_env_kwargs('cabinet surface with plan')
env = TrainEnv(**env_kwargs)
model = RecurrentPPO("MlpLstmPolicy", env, verbose=1)
model.learn(5000)
model.save("ppo_recurrent")

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=20, warn=False)
print(mean_reward)
