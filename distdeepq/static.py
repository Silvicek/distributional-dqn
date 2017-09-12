import tensorflow as tf
import os

def build_z(Vmin, Vmax, nb_atoms):
    dz = (Vmax - Vmin) / (nb_atoms - 1)
    z = tf.range(Vmin, Vmax + dz / 2, dz, dtype=tf.float32, name='z')  # TODO: reuse?

    # z = tf.Variable(initial_value=tf.range(Vmin, Vmax + dz / 2, dz), trainable=False, dtype=tf.float32)
    # z = np.arange(Vmin, Vmax + dz / 2, dz, dtype=np.float32)
    # print('Z=', z)
    return z, dz


def parent_path(path):
    if path.endswith('/'):
        path = path[:-1]
    return os.path.join(*os.path.split(path)[:-1])