import tensorflow as tf
import tensorflow.contrib.layers as layers


def atari_model():
    model = cnn_to_dist_mlp(
            convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
            hiddens=[512])
    return model


def _dist_mlp(hiddens, inpt, num_actions, nb_atoms, scope, reuse=False, layer_norm=False):
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        for hidden in hiddens:
            out = layers.fully_connected(out, num_outputs=hidden, activation_fn=None)
            if layer_norm:
                out = layers.layer_norm(out, center=True, scale=True)
            out = tf.nn.relu(out)

        out = layers.fully_connected(out, num_outputs=num_actions * nb_atoms, activation_fn=None)

        out = tf.reshape(out, shape=[-1, num_actions, nb_atoms])
        out = tf.nn.softmax(out, dim=-1, name='softmax')
        return out


def dist_mlp(hiddens=[], layer_norm=False):
    """This model takes as input an observation and returns values of all actions.

    Parameters
    ----------
    hiddens: [int]
        list of sizes of hidden layers

    Returns
    -------
    p_dist_func: function
        p_dist_function for DistDQN algorithm.
    """
    return lambda *args, **kwargs: _dist_mlp(hiddens, layer_norm=layer_norm, *args, **kwargs)


def _cnn_to_dist_mlp(convs, hiddens, dueling, inpt, num_actions, nb_atoms, scope, reuse=False, layer_norm=False):
    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        with tf.variable_scope("convnet"):
            for num_outputs, kernel_size, stride in convs:
                out = layers.convolution2d(out,
                                           num_outputs=num_outputs,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           activation_fn=tf.nn.relu)
        conv_out = layers.flatten(out)
        with tf.variable_scope("action_value"):
            action_out = conv_out
            for hidden in hiddens:
                action_out = layers.fully_connected(action_out, num_outputs=hidden, activation_fn=None)
                if layer_norm:
                    action_out = layers.layer_norm(action_out, center=True, scale=True)
                action_out = tf.nn.relu(action_out)
            action_scores = layers.fully_connected(action_out, num_outputs=num_actions * nb_atoms, activation_fn=None)

        if dueling:
            raise ValueError('Dueling not supported')
            # with tf.variable_scope("state_value"):
            #     state_out = conv_out
            #     for hidden in hiddens:
            #         state_out = layers.fully_connected(state_out, num_outputs=hidden, activation_fn=None)
            #         if layer_norm:
            #             state_out = layers.layer_norm(state_out, center=True, scale=True)
            #         state_out = tf.nn.relu(state_out)
            #     state_score = layers.fully_connected(state_out, num_outputs=1, activation_fn=None)
            # action_scores_mean = tf.reduce_mean(action_scores, 1)
            # action_scores_centered = action_scores - tf.expand_dims(action_scores_mean, 1)
            # q_out = state_score + action_scores_centered
        else:
            out = tf.reshape(action_scores, shape=[-1, num_actions, nb_atoms])
            out = tf.nn.softmax(out, dim=-1, name='softmax')
        return out


def cnn_to_dist_mlp(convs, hiddens, dueling=False, layer_norm=False):
    """This model takes as input an observation and returns values of all actions.

    Parameters
    ----------
    convs: [(int, int int)]
        list of convolutional layers in form of
        (num_outputs, kernel_size, stride)
    hiddens: [int]
        list of sizes of hidden layers
    dueling: bool
        if true double the output MLP to compute a baseline
        for action scores

    Returns
    -------
    q_func: function
        q_function for DQN algorithm.
    """

    return lambda *args, **kwargs: _cnn_to_dist_mlp(convs, hiddens, dueling, layer_norm=layer_norm, *args, **kwargs)

