import gym
import os

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, A2C

# from pepper_2d_iarlenv import MultiIARLEnv, parse_iaenv_args, check_iaenv_args
from pepper_2d_simulator import populate_PepperRLEnv_args, check_PepperRLEnv_args
from pepper_2d_iarlenv import parse_training_args
from gymified_pepper_envs import PepperRLVecEnvRVO, evaluate_model

# args

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

assert args.model == "PPO2"
assert args.policy == "MlpPolicy"
model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log=args.summary_root_dir+'/')


try:
    model.learn(total_timesteps=args.total_timesteps, tb_log_name=args.run_name)
except KeyboardInterrupt:
    pass

if not os.path.exists(args.checkpoint_root_dir):
    os.makedirs(args.checkpoint_root_dir)
model.save(args.checkpoint_path)
print("Model '{}' saved".format(args.checkpoint_path))

mean_reward = evaluate_model(model, env, num_steps=100)

env.close()
