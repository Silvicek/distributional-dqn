import gym

import distdeepq
from distdeepq.static import build_z
from baselines.common.atari_wrappers_deprecated import wrap_dqn, ScaledFloatFrame

import numpy as np
import matplotlib.pyplot as plt


def plot_distribution(sample, dist_params):
    z, _ = build_z(**dist_params)
    print(z.shape, sample[0].shape)

    fig, ax = plt.subplots()
    ax.barv(z, sample[0])
    # plt.vlines(z, np.zeros_like(z), sample[0])

    # axes = plt.gca()
    # axes.set_ylim([0., 1.1*np.max(n)])
    # plt.grid(True)

    # print('Mean={:.1f}, VaR={:.1f}, CVaR={:.1f}'.format(np.mean(samples), var, cvar))

    plt.show()


def main():
    env = gym.make("PongNoFrameskip-v4")
    env = ScaledFloatFrame(wrap_dqn(env))
    act = distdeepq.load("pong_model.pkl")
    print(act)
    import tensorflow as tf
    sess = tf.get_default_session()
    p_out = tf.get_default_graph().get_tensor_by_name("distdeepq/q_func/cnn_softmax:0")

    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            env.render()
            obs, rew, done, _ = env.step(act(obs[None])[0])
            p = sess.run(p_out, {"distdeepq/observation:0": obs[None]})[0]
            plot_distribution(p, act.get_dist_params())
            episode_rew += rew
        print("Episode reward", episode_rew)


if __name__ == '__main__':
    main()
