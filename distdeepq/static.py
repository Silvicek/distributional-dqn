import tensorflow as tf
# Vmin = -21
# Vmax = 21
# nb_atoms = 51
# dz = (Vmax-Vmin) / (nb_atoms-1)
# z = tf.range(Vmin, Vmax + dz/2, dz, dtype=tf.float32, name='z')
#
# with tf.Session() as sess:
#     zz = sess.run(z)
#     print(zz.shape, zz)


def build_z(Vmin, Vmax, nb_atoms):
    dz = (Vmax - Vmin) / (nb_atoms - 1)
    # z = tf.range(Vmin, Vmax + dz / 2, dz, dtype=tf.float32, name='z')
    # z = tf.Variable(initial_value=tf.range(Vmin, Vmax + dz / 2, dz), trainable=False, dtype=tf.float32)
    z = tf.range(Vmin, Vmax + dz / 2, dz, dtype=tf.float32, name='z')  # TODO: reuse?
    # print('Z=', z)
    return z, dz


# TODO: breakout

