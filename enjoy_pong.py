import gym

import distdeepq
from baselines.common.atari_wrappers_deprecated import wrap_dqn, ScaledFloatFrame



def main():
    env = gym.make("PongNoFrameskip-v4")
    env = ScaledFloatFrame(wrap_dqn(env))
    act = distdeepq.load("pong_model.pkl")
    print(act)

    plot_machine = distdeepq.PlotMachine(act.get_dist_params(), env.action_space.n)

    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            env.render()
            obs, rew, done, _ = env.step(act(obs[None])[0])
            plot_machine.plot_distribution(obs[None])
            episode_rew += rew
        print("Episode reward", episode_rew)


if __name__ == '__main__':
    main()
