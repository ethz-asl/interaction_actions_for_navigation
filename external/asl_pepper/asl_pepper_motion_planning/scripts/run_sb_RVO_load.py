import gym
import numpy as np

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, A2C

# from pepper_2d_iarlenv import MultiIARLEnv, parse_iaenv_args, check_iaenv_args
from pepper_2d_simulator import populate_PepperRLEnv_args, check_PepperRLEnv_args
from pepper_2d_iarlenv import parse_training_args
from gymified_pepper_envs import PepperRLVecEnvRVO, evaluate_model

# Env
args, _ = parse_training_args(
  ignore_unknown=False,
  env_populate_args_func=populate_PepperRLEnv_args,
  env_name="PepperRLVecEnvRVO"
)
args.no_ros = True
args.continuous = True
args.n_envs = 8
check_PepperRLEnv_args(args)

env = PepperRLVecEnvRVO(args)

model = PPO2.load(args.resume_from)

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render(RENDER_LIDAR=True)

print("Rendering done")

mean_reward = evaluate_model(model, env, num_steps=10000)

env.close()
