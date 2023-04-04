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
import torch as th
from stable_baselines3 import SAC, PPO, TD3, DDPG, DQN, A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.her import HerReplayBuffer
from module.jk5_env_v6 import TrainEnvVariableStiffness as TrainEnv
from module.env_kwargs import env_kwargs

test_name = 'cabinet surface with plan'
rl_name = 'SAC'
time_name = time.strftime("%m-%d-%H-%M")
path_name = 'train_results/' + test_name + '/' + rl_name + '/' + time_name + '/'
_, _, env_kwargs = env_kwargs(test_name, save_flag=True, save_path=path_name)
env = TrainEnv(**env_kwargs)
policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=[dict(pi=[128, 128], vf=[128, 128])])
# replay_buffer_kwargs = dict(
#     n_sampled_goal=n_sampled_goal,
#     goal_selection_strategy="future",
# )
model = SAC('MlpPolicy', env, learning_rate=0.0003, buffer_size=1_000_000, learning_starts=80, batch_size=2048,
            tau=0.005, gamma=0.99,
            train_freq=(1, "episode"), gradient_steps=-1, action_noise=None,
            replay_buffer_class=None, replay_buffer_kwargs=None, optimize_memory_usage=False,
            ent_coef='auto', target_update_interval=4, target_entropy='auto',  # 算法特有参数
            use_sde=True, sde_sample_freq=-1, use_sde_at_warmup=True,
            tensorboard_log='log/' + test_name + '/', policy_kwargs=None, verbose=1, seed=None, device='cuda',
            _init_setup_model=True)

model.learn(total_timesteps=80, callback=None, log_interval=4, tb_log_name=rl_name, reset_num_timesteps=True,
            progress_bar=False)

model.save(path=path_name + 'model', exclude=None, include=None)

mean_reward, std_reward = evaluate_policy(model=model, env=env, n_eval_episodes=1, deterministic=True,
                                          render=False, callback=None, reward_threshold=None,
                                          return_episode_rewards=False, warn=True)
print(mean_reward, std_reward)
