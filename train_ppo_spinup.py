#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""用来对比sb3的ppo

Write detailed description here

Write typical usage example here

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2/9/23 10:05 AM   yinzikang      1.0         None
"""
import time
import argparse
import gym
import gym_custom
from spinup.algos.pytorch.vpg.core import MLPActorCritic
from spinup.utils.mpi_tools import mpi_fork
from spinup.utils.run_utils import setup_logger_kwargs
from spinup.algos.pytorch.ppo.ppo import ppo
from gym_custom.envs.env_kwargs import env_kwargs

env_name = 'TrainEnvVariableStiffnessAndPosture-v6'
test_name = 'cabinet surface with plan'
rl_name = 'PPO'
time_name = time.strftime("%m-%d-%H-%M")
path_name = 'train_results/' + test_name + '/' + rl_name + '/' + time_name + '/'
_, _, rl_kwargs = env_kwargs(test_name, save_flag=False, save_path=None)

step_num = 80
# 网络
parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='TrainEnvVariableStiffnessAndPosture-v6')
parser.add_argument('--hid', type=int, default=64)  # 隐藏层神经元个数
parser.add_argument('--l', type=int, default=2)  # 隐藏层层数
# 训练
parser.add_argument('--max_ep_len', type=int, default=step_num)  # 每个episode长度
epoch_num = 2_000 # 2000*80*4 = 640_000
parser.add_argument('--epochs', type=int, default=epoch_num)  # epoch数量
cpu_num = 4
parser.add_argument('--cpu', type=int, default=cpu_num)
parser.add_argument('--steps_per_epoch', type=int, default=step_num * cpu_num)  # 经验池大小
parser.add_argument('--seed', '-s', type=int, default=7)
# 超参数
parser.add_argument('--gamma', type=float, default=0.99)  # forgetting factor
parser.add_argument('--lam', type=int, default=0.97)  # GAE-Lambda
parser.add_argument('--pi_lr', type=int, default=3e-5)  # 策略网络学习率
parser.add_argument('--vf_lr', type=int, default=1e-4)  # 价值网络学习率
parser.add_argument('--train_pi_iters', type=int, default=80)  # 相比vpg增加
parser.add_argument('--train_v_iters', type=int, default=80)
parser.add_argument('--clip_ratio', type=float, default=0.2)  # 相比vpg增加
parser.add_argument('--target_kl', type=float, default=0.01)  # 相比vpg增加
# 输出与可视化
parser.add_argument('--exp_name', type=str, default='ppo')
parser.add_argument('--data_dir', type=str, default='./spinup/')
parser.add_argument('--datestamp', type=bool, default=True)
parser.add_argument('--save_freq', type=int, default=40)  # log保存频率
parser.add_argument('--view_freq', type=int, default=100)  # view可视化频率
args = parser.parse_args()
# 并行
mpi_fork(args.cpu)  # run parallel code with mpi
# 多个key args
ac_kwargs = dict(hidden_sizes=[args.hid] * args.l)
logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed, args.data_dir, args.datestamp)

ppo(lambda: gym.make(args.env, **rl_kwargs),
    actor_critic=MLPActorCritic, ac_kwargs=ac_kwargs,
    seed=args.seed, gamma=args.gamma, lam=args.lam,
    steps_per_epoch=args.steps_per_epoch, epochs=args.epochs, max_ep_len=args.max_ep_len,
    pi_lr=args.pi_lr, vf_lr=args.vf_lr, train_pi_iters=args.train_pi_iters, train_v_iters=args.train_v_iters,
    clip_ratio=args.clip_ratio, target_kl=args.target_kl,
    logger_kwargs=logger_kwargs, save_freq=args.save_freq)
