import tensorflow as tf
import numpy as np
import os
import gym


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
    while True:
        if isinstance(env, gym.Wrapper):
            env = env.env
        else:
            break
    if isinstance(env, gym.Env):
        if hasattr(env, 'ale'):
            actions = env.ale.getMinimalActionSet()
            return [atari_actions[i] for i in actions]


class ActionRandomizer(gym.ActionWrapper):

    def __init__(self, env=None, random_p=0.1):
        super(ActionRandomizer, self).__init__(env)
        self.random_p = random_p

    # def _step(self, action):
    #     if np.random.rand() < self.random_p:
    #         action = np.random.randint(self.action_space.n)
    #     return self.env.step(action)

    def _action(self, action):
        if np.random.rand() < self.random_p:
            action = np.random.randint(self.action_space.n)
        return action


def make_env(game_name):
    from baselines.common.atari_wrappers_deprecated import wrap_dqn
    from baselines.common.misc_util import SimpleMonitor
    env = gym.make(game_name + "NoFrameskip-v4")
    monitored_env = SimpleMonitor(env)
    env = wrap_dqn(monitored_env)
    env = ActionRandomizer(env)
    return env, monitored_env
