import gym
import gym_custom
from gym_custom.envs.env_kwargs import env_kwargs
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
# 环境加载
env_name = 'TrainEnvVariableStiffnessAndPosture-v6'
test_name = 'cabinet surface with plan'
rl_name = 'PPO'
time_name = '04-10-11-54'
path_name = test_name + '/' + rl_name + '/' + time_name + '/'
itr = 262144
mode = 3
save_fig = False
plot_fig = False
render = False

if mode == 1:  # 评估中间模型
    logger_path = "eval_results/" + path_name + "model_" + str(itr)
    modeL_path = "train_results/" + path_name + "model_" + str(itr) + '_steps'
elif mode == 2:  # 评估最后模型
    logger_path = "eval_results/" + path_name + "model"
    modeL_path = "train_results/" + path_name + "model"
elif mode == 3:  # 评估最优模型
    logger_path = "eval_results/" + path_name + "best_model"
    modeL_path = "train_results/" + path_name + "best_model"

# Create environment
_, _, rl_kwargs = env_kwargs(test_name, save_flag=False)
env = make_vec_env(env_id=env_name, n_envs=4, env_kwargs=rl_kwargs)
# env = gym.make(env_name, **rl_kwargs)
# env = DummyVecEnv([lambda: Monitor(gym.make(env_name, **rl_kwargs))])
# env = Monitor(env)
# Load the model
model = PPO.load(modeL_path, env=env, print_system_info=True)

# Evaluate the model
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(mean_reward)