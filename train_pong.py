import distdeepq


def main():
    env, _ = distdeepq.make_env("Pong")

    model = distdeepq.models.cnn_to_dist_mlp(
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[256],
        dueling=False
    )
    act = distdeepq.learn(
        env,
        quant_func=model,
        lr=5e-5,
        max_timesteps=2000000,
        buffer_size=10000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        train_freq=4,
        learning_starts=10000,
        target_network_update_freq=1000,
        gamma=0.99,
        prioritized_replay=False,
        batch_size=1,
        dist_params={'nb_atoms': 1, 'huber_loss': False}
    )
    act.save("pong_model.pkl")
    env.close()


if __name__ == '__main__':
    main()
