import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


class PlotMachine:

    def __init__(self, dist_params, nb_actions, action_set=None):
        nb_atoms = dist_params['nb_atoms']
        tau = np.arange(0, nb_atoms + 1) / nb_atoms

        # extend tau with [0, tau, 1]
        self.tau = np.zeros(nb_atoms+2)
        self.tau[1:-1] = (tau[1:] + tau[:-1]) / 2
        self.tau[-1] = 1.

        self.limits = None

        self.fig, self.ax = plt.subplots()
        self.fig.canvas.draw()

        self.plots = [self.ax.step(np.arange(0, nb_atoms+2), self.tau)[0] for _ in range(nb_actions)]

        if action_set is not None:
            plt.legend(action_set, loc='upper left')

        self.sess = tf.get_default_session()
        self.act = tf.get_default_graph().get_tensor_by_name("distdeepq/q_func/quantiles:0")

    def plot_distribution(self, obs):
        quant_out = self.sess.run(self.act, {"distdeepq/observation:0": obs})[0]

        if self.limits is None:
            self.limits = [np.min(quant_out), np.max(quant_out)]
        else:
            self.limits = [min(np.min(quant_out), self.limits[0]), max(np.max(quant_out), self.limits[1])]

        self.ax.set_xlim(self.limits)
        for line, quant in zip(self.plots, quant_out):
            x_data = np.zeros(len(quant)+2)
            x_data[1:-1] = quant
            x_data[0] = self.limits[0]
            x_data[-1] = self.limits[-1]
            line.set_xdata(x_data)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(1e-10)

