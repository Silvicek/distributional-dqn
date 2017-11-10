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


def make_env(game_name):
    from baselines.common.atari_wrappers import wrap_deepmind, make_atari
    env = make_atari(game_name + "NoFrameskip-v4")
    monitored_env = SimpleMonitor(env)
    # TODO: port to c51
    env = wrap_deepmind(monitored_env, frame_stack=True, scale=True)
    return env, monitored_env


# hard copy from old baselines.common.misc_util
# TODO: remove?
import time


class SimpleMonitor(gym.Wrapper):
    def __init__(self, env):
        """Adds two qunatities to info returned by every step:
            num_steps: int
                Number of steps takes so far
            rewards: [float]
                All the cumulative rewards for the episodes completed so far.
        """
        super().__init__(env)
        # current episode state
        self._current_reward = None
        self._num_steps = None
        # temporary monitor state that we do not save
        self._time_offset = None
        self._total_steps = None
        # monitor state
        self._episode_rewards = []
        self._episode_lengths = []
        self._episode_end_times = []

    def _reset(self):
        obs = self.env.reset()
        # recompute temporary state if needed
        if self._time_offset is None:
            self._time_offset = time.time()
            if len(self._episode_end_times) > 0:
                self._time_offset -= self._episode_end_times[-1]
        if self._total_steps is None:
            self._total_steps = sum(self._episode_lengths)
        # update monitor state
        if self._current_reward is not None:
            self._episode_rewards.append(self._current_reward)
            self._episode_lengths.append(self._num_steps)
            self._episode_end_times.append(time.time() - self._time_offset)
        # reset episode state
        self._current_reward = 0
        self._num_steps = 0

        return obs

    def _step(self, action):
        obs, rew, done, info = self.env.step(action)
        self._current_reward += rew
        self._num_steps += 1
        self._total_steps += 1
        info['steps'] = self._total_steps
        info['rewards'] = self._episode_rewards
        return (obs, rew, done, info)

    def get_state(self):
        return {
            'env_id': self.env.unwrapped.spec.id,
            'episode_data': {
                'episode_rewards': self._episode_rewards,
                'episode_lengths': self._episode_lengths,
                'episode_end_times': self._episode_end_times,
                'initial_reset_time': 0,
            }
        }

    def set_state(self, state):
        assert state['env_id'] == self.env.unwrapped.spec.id
        ed = state['episode_data']
        self._episode_rewards = ed['episode_rewards']
        self._episode_lengths = ed['episode_lengths']
        self._episode_end_times = ed['episode_end_times']

