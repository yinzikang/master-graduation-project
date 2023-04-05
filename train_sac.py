#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""利用sb3 sac进行测试

Write detailed description here

Write typical usage example here

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
3/2/23 3:46 PM   yinzikang      1.0         None
"""
import time

import gym
import torch as th
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from gym_custom.envs.env_kwargs import env_kwargs

env_name = 'TrainEnvVariableStiffnessAndPosture-v6'
test_name = 'cabinet surface with plan'
rl_name = 'SAC'
time_name = time.strftime("%m-%d-%H-%M")
path_name = 'train_results/' + test_name + '/' + rl_name + '/' + time_name + '/'
_, _, rl_kwargs = env_kwargs(test_name, save_flag=True, save_path=path_name)
train_env = make_vec_env(env_id=env_name, n_envs=4, env_kwargs=rl_kwargs)
eval_env = gym.make(env_name, **rl_kwargs)

total_timesteps = 1_000_000
policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[dict(pi=[256, 256], vf=[256, 256])])
replay_buffer_kwargs = dict(n_sampled_goal=4, goal_selection_strategy="future")
checkpoint_callback = CheckpointCallback(save_freq=int(total_timesteps / 10), save_path=path_name, name_prefix="model",
                                         save_replay_buffer=False, save_vecnormalize=False)
eval_callback = EvalCallback(eval_env, best_model_save_path=path_name, log_path=path_name,
                             eval_freq=int(total_timesteps / 10))
callback = CallbackList([checkpoint_callback, eval_callback])

model = SAC('MlpPolicy', train_env, learning_rate=0.0003, policy_kwargs=None, verbose=1, seed=None,
            device='cuda', _init_setup_model=True, tensorboard_log='log/' + test_name + '/' + rl_name + '/' + time_name,
            use_sde=True, sde_sample_freq=-1,
            # off policy特有
            buffer_size=1_000_000, learning_starts=80, batch_size=2048, tau=0.005, gamma=0.99, train_freq=1,
            gradient_steps=-1, action_noise=None, replay_buffer_class=None, replay_buffer_kwargs=None,
            optimize_memory_usage=False, use_sde_at_warmup=True,
            # 算法特有参数
            ent_coef='auto', target_update_interval=4, target_entropy='auto')
model.learn(total_timesteps=total_timesteps, callback=callback, log_interval=4, tb_log_name="",
            reset_num_timesteps=True, progress_bar=True)
model.save(path=path_name + 'model', exclude=None, include=None)
mean_reward, std_reward = evaluate_policy(model=model, env=eval_env, n_eval_episodes=1, deterministic=True,
                                          render=False, callback=None, reward_threshold=None,
                                          return_episode_rewards=False, warn=True)
print(mean_reward, std_reward)
