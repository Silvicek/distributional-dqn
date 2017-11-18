import distdeepq
import numpy as np


def main():
    env, _ = distdeepq.make_env("Pong")
    act = distdeepq.load("pong_model.pkl")
    print(act)
    action_set = distdeepq.actions_from_env(env)
    plot_machine = distdeepq.PlotMachine(act.get_dist_params(), env.action_space.n, action_set)

    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            env.render()
            obs, rew, done, _ = env.step(act(np.array(obs)[None])[0])
            plot_machine.plot_distribution(np.array(obs)[None])
            episode_rew += rew
        print("Episode reward", episode_rew)


if __name__ == '__main__':
    main()
