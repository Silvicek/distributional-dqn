import argparse
import gym
import numpy as np
import os
import json
import baselines.common.tf_util as U

import distdeepq
from baselines.common.misc_util import get_wrapper_by_name, SimpleMonitor, boolean_flag, set_global_seeds
from baselines.common.atari_wrappers import wrap_deepmind


def make_env(game_name):
    env = gym.make(game_name + "NoFrameskip-v4")
    env_monitored = SimpleMonitor(env)
    env = wrap_deepmind(env_monitored)
    return env_monitored, env


def parse_args():
    parser = argparse.ArgumentParser("Evaluate an already learned DQN model.")
    # Environment
    parser.add_argument("--env", type=str, required=True, help="name of the game")
    parser.add_argument("--model-dir", type=str, default=None, help="load model from this directory. ")
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
                         'nb_atoms': old_args['nb_atoms']})
        U.load_state(os.path.join(args.model_dir, "saved"))
        wang2015_eval(args.env, act, stochastic=args.stochastic)


if __name__ == '__main__':
    main()
