"Even with a discount factor only slightly lower than 1, Q-function learning leads to propagation of errors and instabilities when the value function is approximated with an artificial neural network.[7] In that case, starting with a lower discount factor and increasing it towards its final value accelerates learning.[8]"

**Current To Dos:**

### ToDO

### Code:

Prio 1:

* Check binning: 0 of direction, speed as float
* For NN/meaningful binning: Direction as sin/cos (?)

* add orientation (y position, xy dist to checkpoint...) to state (? - policy vs observations)


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


Done:

* smoothen out distance penalty
* dynamic reward_dict