import tensorflow as tf
import numpy as np
import os


def build_z(Vmin, Vmax, nb_atoms, numpy=False):
    dz = (Vmax - Vmin) / (nb_atoms - 1)
    if numpy:
        z = np.arange(Vmin, Vmax + dz / 2, dz)
    else:
        z = tf.range(Vmin, Vmax + dz / 2, dz, dtype=tf.float32, name='z')  # TODO: reuse?

    return z, dz


def parent_path(path):
    if path.endswith('/'):
        path = path[:-1]
    return os.path.join(*os.path.split(path)[:-1])

atari_actions = ['noop', 'fire', 'up', 'right', 'left',
                 'down', 'up-right', 'up-left', 'down-right', 'down-left',
                 'up-fire', 'right-fire', 'left-fire', 'down-fire', 'up-right-fire',
                 'up-left-fire', 'down-right-fire', 'down-left-fire']


def actions_from_env(env):
    """ Propagate through all wrappers to get action indices. """
    import gym
    while True:
        if isinstance(env, gym.Wrapper):
            env = env.env
        else:
            break
    if isinstance(env, gym.Env):
        if hasattr(env, 'ale'):
            actions = env.ale.getMinimalActionSet()
            return [atari_actions[i] for i in actions]


