import gym

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy


import gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
train_env = make_vec_env("CartPole-v1", n_envs=4)
eval_env = gym.make("CartPole-v1")
model = PPO("MlpPolicy", train_env, verbose=1)
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(mean_reward)

model.learn(total_timesteps=25000)
model.save("ppo_cartpole")
del model # remove to demonstrate saving and loading

model = PPO.load("ppo_cartpole")
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
print(mean_reward)
