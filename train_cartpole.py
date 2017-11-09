import gym
import numpy as np
import tensorflow as tf
import distdeepq


def callback(lcl, glb):
    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    return is_solved


def main():
    env = gym.make("CartPole-v0")
    env.seed(1337)
    np.random.seed(1337)
    tf.set_random_seed(1337)

    model = distdeepq.models.dist_mlp([64])
    act = distdeepq.learn(
        env,
        quant_func=model,
        lr=1e-3,
        max_timesteps=100000,
        buffer_size=50000,
        exploration_fraction=0.2,
        exploration_final_eps=0.02,
        print_freq=10,
        callback=callback,
        batch_size=32,
        dist_params={'nb_atoms': 1, 'huber_loss': True}
    )
    print("Saving model to cartpole_model.pkl")
    act.save("cartpole_model.pkl")


if __name__ == '__main__':
    main()
