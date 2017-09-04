import tensorflow as tf


def build_z(Vmin, Vmax, nb_atoms):
    dz = (Vmax - Vmin) / (nb_atoms - 1)
    z = tf.range(Vmin, Vmax + dz / 2, dz, dtype=tf.float32, name='z')  # TODO: reuse?

    # z = tf.Variable(initial_value=tf.range(Vmin, Vmax + dz / 2, dz), trainable=False, dtype=tf.float32)
    # z = np.arange(Vmin, Vmax + dz / 2, dz, dtype=np.float32)
    # print('Z=', z)
    return z, dz


# TODO: CVaR - finish actor (build_act, actor_params)
# TODO: save more often
# TODO: Atari
# TODO: visualize histogram


