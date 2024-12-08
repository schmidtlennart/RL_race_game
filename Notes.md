"Even with a discount factor only slightly lower than 1, Q-function learning leads to propagation of errors and instabilities when the value function is approximated with an artificial neural network.[7] In that case, starting with a lower discount factor and increasing it towards its final value accelerates learning.[8]"

**Current To Dos:**

### ToDO

### Code:

* `reward_list` to `reward_dict`, named, all other code dynamically

* Reward map: Wall+pad penalty have to stay the same everywhere, separate reward maps for all components
* Add later exploration impulses: Annilated / Sinus-based decay or coupled with progress in learning Q
* Reduce action space: Drop 0:None
* Why collisions: Plot min(distances) ~ Q, does Q decrease? Does Q decrease towards end of episode?
* Logging: Plot (cumulative) Q and reward, max Q, Record path/final y-location
* Check initialization values vs/and reward value ranges
* Analyse Q table via 2/3D-PCA - any paths? (good if smooth edges between paths) or bad: isolated maxima/uniform patterns/jumps?. Maybe mark maxQ paths

Later:

* Fix Penalty for top wall (move trophy?)


With DeepQ via NN:

* replay buffer
* Policy-gradient approaches