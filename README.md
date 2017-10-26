# Distributional DQN
Implementation of 'A Distributional Perspective on Reinforcement Learning' based on OpenAi DQN baseline

Requires the installation of https://github.com/openai/baselines

### Usage:
For simple benchmarking:

    python3 train_[{cartpole, pong}].py
    python3 enjoy_[{cartpole, pong}].py

For full Atari options see help

    python3 train_atari.py --help

after learning, you can visualize the distributions by running

    python3 enjoy_atari.py --visual ...

-----------------

Some features not supported (prioritized replay, double q-learning, dueling)
