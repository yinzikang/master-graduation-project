#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""利用sb3 ppo进行测试

Write detailed description here

Write typical usage example here

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
3/2/23 3:46 PM   yinzikang      1.0         None
"""
import time
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
import gym
from gym_custom.envs.env_kwargs import env_kwargs
from rnn_feature_extractor import LSTMFeatureExtractor
from stable_baselines3.common.torch_layers import FlattenExtractor

env_name = 'TrainEnvVariableStiffnessAndPostureAndSM-v8'
test_name = 'cabinet surface with plan v7'
# test_name = 'cabinet drawer open with plan'
rl_name = 'PPO'
time_name = time.strftime("%m-%d-%H-%M")
path_name = 'train_results/' + test_name + '/' + rl_name + '/' + time_name + '/'
_, _, rl_kwargs = env_kwargs(test_name, save_flag=True, save_path=path_name)
env_num = 8
episode_length = 80
train_env = make_vec_env(env_id=env_name, n_envs=env_num, env_kwargs=rl_kwargs)
eval_env =  make_vec_env(env_id=env_name, n_envs=env_num, env_kwargs=rl_kwargs)

total_timesteps = episode_length * env_num * 2 ** 10  # 11: 655_360, 12: 1310720, 13: 2621440
policy_kwargs = dict(features_extractor_class=LSTMFeatureExtractor,
                     features_extractor_kwargs=dict(features_dim=64, num_layers=2),
                     share_features_extractor=True,
                     activation_fn=th.nn.ReLU,
                     net_arch=dict(pi=[64, 64], vf=[64, 64]))
# replay_buffer_kwargs = dict(n_sampled_goal=4, goal_selection_strategy="future")
checkpoint_callback = CheckpointCallback(save_freq=int(total_timesteps / 10 / env_num),
                                         save_path=path_name, name_prefix="model",
                                         save_replay_buffer=False, save_vecnormalize=False)
eval_callback = EvalCallback(eval_env, best_model_save_path=path_name, log_path=path_name,
                             eval_freq=int(total_timesteps / 10 / env_num))
callback = CallbackList([checkpoint_callback, eval_callback])

model = PPO('MlpPolicy', train_env, learning_rate=0.0003, policy_kwargs=policy_kwargs, verbose=1, seed=None,
            device='cuda', _init_setup_model=True, tensorboard_log='log/' + test_name + '/' + rl_name + '/' + time_name,
            use_sde=False, sde_sample_freq=-1,
            # on policy特有 除以多少就是多少次更新
            n_steps=int(total_timesteps / 1024), batch_size=int(total_timesteps / 1024 / 8), gamma=0.99,
            gae_lambda=0.98,
            ent_coef=0.0, vf_coef=0.0, max_grad_norm=0.5,
            # 算法特有参数，n_epochs=n_steps/batch_size
            n_epochs=16, clip_range=0.2, clip_range_vf=None, normalize_advantage=True, target_kl=0.01)
model.learn(total_timesteps=total_timesteps, callback=callback, log_interval=4, tb_log_name="",
            reset_num_timesteps=True, progress_bar=True)
model.save(path=path_name + 'model')
mean_reward, std_reward = evaluate_policy(model=model, env=eval_env, n_eval_episodes=10, deterministic=True,
                                          render=False, callback=None, reward_threshold=None,
                                          return_episode_rewards=False, warn=True)
print(mean_reward, std_reward)
