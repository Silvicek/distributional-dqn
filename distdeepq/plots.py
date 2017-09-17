import numpy as np
import matplotlib.pyplot as plt
from distdeepq.static import build_z
import tensorflow as tf

class PlotMachine:

    def __init__(self, dist_params, nb_actions):

        self.z, self.dz = build_z(**dist_params, numpy=True)

        h_plots = int(np.sqrt(nb_actions))
        v_plots = int(np.ceil(nb_actions / h_plots))

        plt.ion()
        # self.fig, self.ax = plt.subplots(v_plots, h_plots, sharex=True, sharey=True)
        self.fig, self.ax = plt.subplots()

        self.sess = tf.get_default_session()
        self.p_out = tf.get_default_graph().get_tensor_by_name("distdeepq/q_func/cnn_softmax:0")

    def make_pdf(self, obs):
        return self.sess.run(self.p_out, {"distdeepq/observation:0": obs})[0]

    def plot_distribution(self, obs):
        pdf_act = self.make_pdf(obs)

        # multiple plots
        # for sample, ax in zip(pdf_act, self.ax.flatten()):
        #     ax.clear()
        #     ax.bar(self.z, sample, self.dz)

        # single plot
        self.ax.clear()
        for sample in pdf_act:
            self.ax.bar(self.z, sample, self.dz * 0.9)

        plt.pause(0.001)

