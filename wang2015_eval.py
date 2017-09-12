import argparse
import gym
import numpy as np
import os
import json
import baselines.common.tf_util as U

import distdeepq
from baselines.common.misc_util import get_wrapper_by_name, SimpleMonitor, boolean_flag, set_global_seeds
from baselines.common.atari_wrappers_deprecated import wrap_dqn
import gym
import distdeepq
import numpy as np
import matplotlib.pyplot as plt
from baselines.common.atari_wrappers_deprecated import wrap_dqn


def cvar_from_histogram(alpha, pdf, bins):
    bins = np.array([(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)])

    threshold = 0.
    cvar = 0.
    var = 0.
    for n, bin in zip(pdf, bins):

        threshold += n
        if threshold >= alpha:
            n_rest = alpha - (threshold - n)
            cvar += n_rest * bin
            var = bin
            break

        cvar += n * bin

    return var, cvar / alpha


def plot_distribution(samples, alpha, nb_bins):
    n, bins, patches = plt.hist(samples, nb_bins, normed=1, facecolor='green', alpha=0.75)
    pdf = n * np.diff(bins)
    var, cvar = cvar_from_histogram(alpha, pdf, bins)

    y_lim = 1.1*np.max(n)

    plt.vlines([var], 0, y_lim)
    plt.vlines([cvar], 0, y_lim/3, 'r')

    # plt.xlabel('Smarts')
    # plt.ylabel('Probability')
    # plt.title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=100,\ \sigma=15$')
    axes = plt.gca()
    axes.set_ylim([0., 1.1*np.max(n)])
    plt.grid(True)

    print('Mean={:.1f}, VaR={:.1f}, CVaR={:.1f}'.format(np.mean(samples), var, cvar))

    plt.show()


def make_env(game_name):
    env = gym.make(game_name + "NoFrameskip-v4")
    env_monitored = SimpleMonitor(env)
    env = wrap_dqn(env_monitored)
    return env_monitored, env


def parse_args():
    parser = argparse.ArgumentParser("Evaluate an already learned DQN model.")
    # Environment
    parser.add_argument("--env", type=str, required=True, help="name of the game")
    parser.add_argument("--model-dir", type=str, default=None, help="load model from this directory. ")
    parser.add_argument("--nb-episodes", type=int, default=1000, help="statistics over how many episodes?")
    boolean_flag(parser, "stochastic", default=True, help="whether or not to use stochastic actions according to models eps value")
    boolean_flag(parser, "dueling", default=False, help="whether or not to use dueling model")

    return parser.parse_args()


def wang2015_eval(game_name, act, stochastic):
    print("==================== wang2015 evaluation ====================")
    episode_rewards = []

    for num_noops in range(1, 31):
        env_monitored, eval_env = make_env(game_name)
        eval_env.unwrapped.seed(1)

        # get_wrapper_by_name(eval_env, "NoopResetEnv").override_num_noops = num_noops
        # XXX: whats this

        eval_episode_steps = 0
        done = True
        while True:
            if done:
                obs = eval_env.reset()
            eval_episode_steps += 1
            action = act(np.array(obs)[None], stochastic=stochastic)[0]

            obs, reward, done, info = eval_env.step(action)
            if done:
                obs = eval_env.reset()
            if len(info["rewards"]) > 0:
                episode_rewards.append(info["rewards"][0])
                break
            if info["steps"] > 108000:  # 5 minutes of gameplay
                episode_rewards.append(env_monitored._current_reward)
                break
        print("Num steps in episode {} was {} yielding {} reward".format(
              num_noops, eval_episode_steps, episode_rewards[-1]), flush=True)
    print("Evaluation results: " + str(np.mean(episode_rewards)))
    print("=============================================================")
    return np.mean(episode_rewards)


def measure_performance(env, act, stochastic, nb_episodes, alpha, nb_atoms):

    history = np.zeros(nb_episodes)

    for ix in range(nb_episodes):
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            action = act(np.array(obs)[None], stochastic=stochastic)[0]

            obs, reward, done, info = env.step(action)
            episode_rew += reward
        print("{:4d} Episode reward: {:.3f}".format(ix, episode_rew))

        history[ix] = episode_rew

    plot_distribution(history, alpha=alpha, nb_bins=nb_atoms)


def main():
    set_global_seeds(1)
    args = parse_args()

    with U.make_session(4) as sess:  # noqa
        _, env = make_env(args.env)
        model_parent_path = distdeepq.parent_path(args.model_dir)
        old_args = json.load(open(model_parent_path + '/args.json'))

        act = distdeepq.build_act(
            make_obs_ph=lambda name: U.Uint8Input(env.observation_space.shape, name=name),
            p_dist_func=distdeepq.models.atari_model(),
            num_actions=env.action_space.n,
            dist_params={'Vmin': old_args['vmin'],
                         'Vmax': old_args['vmax'],
                         'nb_atoms': old_args['nb_atoms']},
            risk_alpha=1.0)
        U.load_state(os.path.join(args.model_dir, "saved"))
        # wang2015_eval(args.env, act, stochastic=args.stochastic)
        measure_performance(env, act, args.stochastic, args.nb_episodes,
                            old_args['cvar_alpha'], old_args['nb_atoms'])


if __name__ == '__main__':
    main()
