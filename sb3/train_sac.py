#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""利用sb3 sac进行测试

Write detailed description here

Write typical usage example here

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
3/2/23 3:46 PM   yinzikang      1.0         None
"""
import torch as th
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from module.jk5_env_v5 import TrainEnv
from module.env_kwargs import load_env_kwargs

_, _, env_kwargs = load_env_kwargs('cabinet surface with plan')
env = TrainEnv(**env_kwargs)
policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=[dict(pi=[32, 32], vf=[32, 32])])

model = SAC('MlpPolicy', env, learning_rate=0.0003, buffer_size=1000000, learning_starts=100, batch_size=1024,
            tau=0.005,
            gamma=0.99, train_freq=(1, "episode"), gradient_steps=-1, action_noise=None, replay_buffer_class=None,
            replay_buffer_kwargs=None, optimize_memory_usage=False, ent_coef='auto', target_update_interval=1,
            target_entropy='auto', use_sde=False, sde_sample_freq=-1, use_sde_at_warmup=False, tensorboard_log='./log',
            policy_kwargs=None, verbose=1, seed=None, device='cuda', _init_setup_model=True)
model.learn(total_timesteps=1000000, log_interval=4)
model.save('test')

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(mean_reward, std_reward)
