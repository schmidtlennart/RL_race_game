"Even with a discount factor only slightly lower than 1, Q-function learning leads to propagation of errors and instabilities when the value function is approximated with an artificial neural network.[7] In that case, starting with a lower discount factor and increasing it towards its final value accelerates learning.[8]"

**Current To Dos:**

### ToDO

##### Code:

Prio 1:

* reduce backwarsd speed so that it advances forward preferrably
* reduce narrowness in parts of the map (including trophy)
* Re-Check binning: 0 of direction, speed as float
* add logging: sd, mean, median, q10. Also log growth of Qtable, n checkpoints reached, win/loss, TD error
* For NN/meaningful binning: Direction as sin/cos (?)

* add orientation state (y position, xy dist/orientation toof trophy to checkpoint...) to state (? - policy vs observations)
* Add later exploration impulses: Annilated / Sinus-based decay or coupled with progress in learning Q
* Analyse Q table via 2/3D-PCA - any paths? (good if smooth edges between paths) or bad: isolated maxima/uniform patterns/jumps?. Maybe mark maxQ paths
* time penalty? -1 each step

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
* debugging binning (still not ideal ideal)


# Key Takeaways

* Most important:
  * Meaningful discretization of continuous state - each bin needs a meaning
  * Reward shaping until smooth and meaningful everywhere
* Overwriting Q-values if win/loss or level checkpoint gives the push
* Explicit guidance: level checkpoints really helpful
* Adding minimal resting speed helps with being stuck in local minima