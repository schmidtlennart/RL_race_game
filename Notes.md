"Even with a discount factor only slightly lower than 1, Q-function learning leads to propagation of errors and instabilities when the value function is approximated with an artificial neural network.[7] In that case, starting with a lower discount factor and increasing it towards its final value accelerates learning.[8]"

**Current To Dos:**

### ToDO

### Code:

Prio 1:

* smoothen out distance penalty
* append logging if loaded
* Walls as invisible pads
* add orientation (y position, dist to checkpoint...) to state (?)

* add x,y-distance to checkpoint to state (? - policy vs observations)
* `reward_list` to `reward_dict`, named, all other code dynamically

* For NN: Direction as sin/cos (?)
* Reward map: Wall+pad penalty have to stay the same everywhere, separate reward maps for all components
* Add later exploration impulses: Annilated / Sinus-based decay or coupled with progress in learning Q
* Why collisions: Plot min(distances) ~ Q, does Q decrease? Does Q decrease towards end of episode?
* Analyse Q table via 2/3D-PCA - any paths? (good if smooth edges between paths) or bad: isolated maxima/uniform patterns/jumps?. Maybe mark maxQ paths

Later:

* Fix Penalty for top wall (move trophy?)


Viz:

* Live graph of logging metrics

With DeepQ via NN:

* replay buffer
* Policy-gradient approaches