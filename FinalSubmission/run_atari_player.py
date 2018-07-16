#!/usr/bin/env python3

import time
from mpi4py import MPI
from baselines.common import set_global_seeds
from baselines import bench
import os.path as osp
import tensorflow as tf
from baselines import logger
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.cmd_util import atari_arg_parser

def play(env_id, num_timesteps, seed):
    from baselines.ppo1 import pposgd_simple, cnn_policy
    import baselines.common.tf_util as U
    rank = MPI.COMM_WORLD.Get_rank()
    sess = U.single_threaded_session()
    sess.__enter__()
    if rank == 0:
        logger.configure()
    else:
        logger.configure(format_strs=[])
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)

    env = make_atari(env_id)

    def policy_fn(name, ob_space, ac_space): #pylint: disable=W0613
        return cnn_policy.CnnPolicy(name=name, ob_space=ob_space, ac_space=ac_space)

    env = bench.Monitor(env, logger.get_dir() and osp.join(logger.get_dir(), str(rank)))
    env.seed(workerseed)

    env = wrap_deepmind(env)

    obs = env.reset()
    env.seed(workerseed)

    print("Loading model======================")

    pi = policy_fn('pi', env.observation_space, env.action_space)
    tf.train.Saver().restore(sess, 'models/sim2_RGB_GS_downsampling_def_ts')


    done = False
    for eps in range(3):
        while not done:
            action = pi.act(True, obs)[0]
            obs, reward, done, info = env.step(action)
            env.render()
            time.sleep(0.01)
        done=False
        eps+=1



def main():
    args = atari_arg_parser().parse_args()
    play(args.env, num_timesteps=args.num_timesteps, seed=args.seed)

if __name__ == '__main__':
    main()
