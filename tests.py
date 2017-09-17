from distdeepq.static import build_z
import numpy as np
import matplotlib.pyplot as plt


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()




class PlotMachine:

    def __init__(self, dist_params, nb_actions):

        self.z, self.dz = build_z(**dist_params, numpy=True)

        h_plots = int(np.sqrt(nb_actions))
        v_plots = int(np.ceil(nb_actions / h_plots))

        plt.ion()
        self.fig, self.ax = plt.subplots(v_plots, h_plots, sharex=True, sharey=True)

    def plot_distribution(self, sample):

        for i, ax in enumerate(self.ax.flatten()):
            ax.clear()

            ax.bar(self.z, sample[i], self.dz*0.9)
        # plt.vlines(z, np.zeros_like(z), sample[0])

        # axes = plt.gca()
        # axes.set_ylim([0., 1.1*np.max(n)])
        # plt.grid(True)

        # print('Mean={:.1f}, VaR={:.1f}, CVaR={:.1f}'.format(np.mean(samples), var, cvar))

        plt.pause(0.05)

nb_actions = 6

dist_params={'Vmin': -10, 'Vmax': 10, 'nb_atoms': 51}

plotter = PlotMachine(dist_params, nb_actions)

for i in range(10):

    p = np.apply_along_axis(softmax, -1, np.random.normal(size=[nb_actions, 51]))

    plotter.plot_distribution(p)

while True:
    plt.pause(0.05)