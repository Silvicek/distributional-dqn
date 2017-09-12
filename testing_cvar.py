import gym
import distdeepq
import numpy as np
import matplotlib.pyplot as plt

from baselines.common.atari_wrappers_deprecated import wrap_dqn, ScaledFloatFrame
# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('environment')
# parser.add_argument('--is_atari', type=int, default=0)


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


def main():
    env = gym.make("CartPole-v0")
    act = distdeepq.load("cartpole_model.pkl")

    nb_episodes = 1000

    history = np.zeros(nb_episodes)

    for ix in range(nb_episodes):
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            obs, rew, done, _ = env.step(act(obs[None])[0])
            episode_rew += rew
        print("{:4d} Episode reward: {:.3f}".format(ix, episode_rew))

        history[ix] = episode_rew

    plot_distribution(history, alpha=0.05, nb_bins=51)


if __name__ == '__main__':
    main()
