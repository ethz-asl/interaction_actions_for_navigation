#!/usr/bin/env python3

from mpi4py import MPI
from baselines.common import set_global_seeds
from baselines import bench
import os.path as osp
from baselines import logger
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from pepper_2d_simulator import PepperRLEnv
from map2d import Map2D

import numpy as np
from gym.spaces.box import Box


def train(num_timesteps, seed, resume):
    map_folder = "."
    map_name = "empty"
    map_ = Map2D(map_folder, map_name)
    print("Map '{}' loaded.".format(map_name))
    # RL multi-agent simulator
    import pposgd_simple
    import cnn_policy
    import baselines.common.tf_util as U
    rank = MPI.COMM_WORLD.Get_rank()
    sess = U.single_threaded_session()
    sess.__enter__()
    if rank == 0:
        logger.configure()
    else:
        logger.configure(format_strs=[])
    env = PepperRLEnv(args)
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank() if seed is not None else None
    set_global_seeds(workerseed)
    def policy_fn(name, ob_space, ac_space): #pylint: disable=W0613
        return cnn_policy.CnnPolicy(name=name, ob_space=ob_space, ac_space=ac_space)
    pposgd_simple.learn(env, policy_fn,
        max_timesteps=int(num_timesteps * 1.1),
        timesteps_per_actorbatch=256,
        clip_param=0.2, entcoeff=0.01,
        optim_epochs=4,
        optim_stepsize=1e-3, # original 1e-3
        optim_batchsize=64,
        gamma=0.99, lam=0.95,
        schedule='linear',
        resume_training=resume,
    )
    env.close()

def parse_args():
    import argparse
    ## Arguments
    parser = argparse.ArgumentParser(description='Keeps track of git repositories.')
    parser.add_argument(
            '--resume',
            dest='resume',
            action='store_true',
            help='loads saved models and continues training from there.',
    )
    parser.add_argument(
            '--seed',
            dest="seed",
            type=int,
            default=None,
            help='seed value',
            )
    parser.add_argument(
            '--timesteps',
            dest="num_timesteps",
            type=int,
            default=1000000,
            help='timesteps to train for',
            )

    ARGS = parser.parse_args()
    return ARGS


if __name__ == '__main__':
    args = parse_args()
    train(num_timesteps=args.num_timesteps, seed=args.seed, resume=args.resume)

