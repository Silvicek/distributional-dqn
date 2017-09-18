import gym
import distdeepq
import numpy as np
import matplotlib.pyplot as plt

from baselines.common.atari_wrappers_deprecated import wrap_dqn, ScaledFloatFrame
import argparse
from baselines.common.misc_util import boolean_flag
import json
import baselines.common.tf_util as U
import os


def parse_args():
    parser = argparse.ArgumentParser("Run an already learned DQN model.")
    # Environment
    parser.add_argument("--env", type=str, required=True, help="name of the game")
    parser.add_argument("--model-dir", type=str, default=None, help="load model from this directory. ")
    parser.add_argument("--nb-episodes", type=int, default=1000, help="number of episodes to compute statistics over")
    parser.add_argument("--cvar-alpha", type=float, default=0., help="if set to 0, the training cvar will be used")
    boolean_flag(parser, "stochastic", default=True, help="whether or not to use stochastic actions according to models eps value")
    boolean_flag(parser, "dueling", default=False, help="whether or not to use dueling model")

    return parser.parse_args()


def make_env(game_name):
    env = gym.make(game_name + "NoFrameskip-v4")
    env = wrap_dqn(env)
    return env


def main():
    with U.make_session(4) as sess:
        args = parse_args()
        env = make_env(args.env)

        model_parent_path = distdeepq.parent_path(args.model_dir)
        old_args = json.load(open(model_parent_path + '/args.json'))

        act = distdeepq.build_act(
            make_obs_ph=lambda name: U.Uint8Input(env.observation_space.shape, name=name),
            p_dist_func=distdeepq.models.atari_model(),
            num_actions=env.action_space.n,
            dist_params={'Vmin': old_args['vmin'],
                         'Vmax': old_args['vmax'],
                         'nb_atoms': old_args['nb_atoms']},
            risk_alpha=old_args['cvar_alpha'])
        U.load_state(os.path.join(args.model_dir, "saved"))

        nb_episodes = args.nb_episodes
        history = np.zeros(nb_episodes)

        for ix in range(nb_episodes):
            obs, done = env.reset(), False
            episode_rew = 0
            while not done:
                action = act(np.array(obs)[None], stochastic=args.stochastic)[0]
                obs, rew, done, info = env.step(action)
                episode_rew += rew
            print("{:4d} Episode reward: {:.3f}".format(ix, episode_rew))

            history[ix] = episode_rew

        if args.cvar_alpha == 0:
            plot_distribution(history, alpha=old_args['cvar_alpha'], nb_bins=old_args['nb_atoms'])
        else:
            plot_distribution(history, alpha=args.cvar_alpha, nb_bins=old_args['nb_atoms'])


if __name__ == '__main__':
    main()
