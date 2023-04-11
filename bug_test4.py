import gym

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy


import gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from gym_custom.envs.env_kwargs import env_kwargs

# # Parallel environments
# env = make_vec_env("CartPole-v1", n_envs=4)
# eval_env = gym.make("CartPole-v1")
# model = PPO.load("ppo_cartpole")
# mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
# print(mean_reward)
# Parallel environments
env_name = 'TrainEnvVariableStiffnessAndPosture-v6'
test_name = 'cabinet surface with plan'
path1 = "model"
path2 = "train_results/cabinet surface with plan/PPO/04-10-22-22/model"

_, _, rl_kwargs = env_kwargs(test_name, save_flag=False)
env = make_vec_env(env_name, n_envs=4,env_kwargs=rl_kwargs)
# env = gym.make(env_name)

# model = PPO("MlpPolicy", env, verbose=1)
# mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
# print(mean_reward)


model = PPO.load(path2)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(mean_reward)
