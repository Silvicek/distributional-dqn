import numpy as np
import matplotlib.pyplot as plt
from distdeepq.static import build_z
import tensorflow as tf


class PlotMachine:

    def __init__(self, dist_params, nb_actions):

        self.z, self.dz = build_z(**dist_params, numpy=True)

        plt.ion()

        self.fig, self.ax = plt.subplots()

        self.bars = [self.ax.bar(self.z, np.ones_like(self.z)*0.25, self.dz * 0.9) for _ in range(nb_actions)]

        self.sess = tf.get_default_session()
        self.p_out = tf.get_default_graph().get_tensor_by_name("distdeepq/q_func/cnn_softmax:0")

    def make_pdf(self, obs):
        return self.sess.run(self.p_out, {"distdeepq/observation:0": obs})[0]

    def plot_distribution(self, obs):
        pdf_act = self.make_pdf(obs)

        for rects, sample in zip(self.bars, pdf_act):
            for rect, y in zip(rects, sample):
                rect.set_height(y)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


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


def plot_distribution_with_cvar(samples, alpha, nb_bins):
    n, bins, patches = plt.hist(samples, nb_bins, normed=1, facecolor='green', alpha=0.75)
    pdf = n * np.diff(bins)
    var, cvar = cvar_from_histogram(alpha, pdf, bins)

    y_lim = 1.1*np.max(n)

    plt.vlines([var], 0, y_lim)
    plt.vlines([cvar], 0, y_lim/3, 'r')

    axes = plt.gca()
    axes.set_ylim([0., 1.1*np.max(n)])
    plt.grid(True)

    print('Mean={:.1f}, VaR={:.1f}, CVaR={:.1f}'.format(np.mean(samples), var, cvar))

    plt.show()


def cvar_from_samples(samples, alpha, nb_bins):
    n, bins, patches = plt.hist(samples, nb_bins, normed=1, facecolor='green', alpha=0.75)
    pdf = n * np.diff(bins)
    var, cvar = cvar_from_histogram(alpha, pdf, bins)
    return var, cvar

