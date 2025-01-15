## The game

Reinforcement Learning on a 2D Python [Race Game](https://github.com/tdostilio/Race_Game). The game is essentially a maze for a small car that has to find the trophy.

The Reinforcement Learning Environment:

![RaceGame Environment](doc/rl_environment.png "RaceGame Environment"){width=50%}

I restructured the game into an openAI-gym by adding 1st person view for an agent (8 whiskers to measure distance to surrounding objects) and by designing a reward map to keep it away from walls and obstacles. This yielded 10 state observations (distances, car's speed, car's orientation) that were used to choose one of 5 actions (no action, left, right, forward, reverse). As I started with simple Qlearning to learn the basics and it was quite a challenge to get it to learn, I added checkpoints that would reward the agent once passed into the next "level". This led to a reward map and reward distribution like so:

![Reward Map and Distribution](doc/reward_map.png "Reward Map and Distribution"){width=50%}

Reward function design and agent strategy optimization require mutual optimization. I went through various stages of reward maps that are calculated from weighted sub-maps that describe winning/losing and distances to obstacles and checkpoints:

![Weighted Sub-maps](doc/reward_submaps.png "Weighted Sub-maps"){width=50%}

## Qlearning

![Q-table agent exploring the maze](doc/recording_success.gif "Q-table agent exploring the maze"){width=50%}

Before going into any machine learning approaches, I Implemented traditional [Qlearning](https://en.wikipedia.org/wiki/Q-learning) via a Q-table that represents the state and action space in a table. Naturally, this is not the best choice for a continous environment but it helps in refining the environment and reward shaping. We can log the learning process to see if the agent actually improves in solving the maze:

![Logging the learning process](doc/reward_submaps.png "Logging the learning process"){width=25%}

The agent learns to avoid walls and to navigate the maze, resulting in increasing runs that actually reach the trophy (no one said doing so in reverse was not allowed). However, it gets stuck in local minima of the Q-table quite often which leads to rocking back and forth without actually advancing.

My key takeaways so far:
* Meaningful discretization of continuous state - each bin needs a meaning
* Reward shaping until smooth and meaningful everywhere
* Explicit guidance: level checkpoints really helpful
* Adding minimal resting speed helps with being stuck in local minima

## Roadmap

* Generalizability: Random level construction
* DeepQ or Policy-Gradient approaches via Neural Nets
* Add memory: Transformer-based or Replay Buffer
* Top view: CNN-based architecture


## Sources

* [RaceGame by tdostilio](https://github.com/tdostilio/Race_Game)
* [Medium: Ultimate Guide to Reinforcement Learning Part 1 — Creating a Game](https://towardsdatascience.com/ultimate-guide-for-reinforced-learning-part-1-creating-a-game-956f1f2b0a91)
* [Python Programming Tutorial Series on Qlearning](https://pythonprogramming.net/q-learning-reinforcement-learning-python-tutorial/)

