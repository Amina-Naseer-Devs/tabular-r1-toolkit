# Tabular RL Toolkit

A small learning project built from scratch to understand the core ideas of
reinforcement learning on small problems: MDPs, model-based planning, model-free
learning, Bayesian inference, and POMDP belief tracking. The algorithms use only NumPy.

## Modules

| Module         | What it does                                                                                                 | Concept shown                         |
| -------------- | ------------------------------------------------------------------------------------------------------------ | ------------------------------------- |
| `mdp.py`       | `TabularMDP` class: transitions `P[s,a,s']`, rewards `R[s,a]`, discount `γ`. The gridworld is built with it. | MDPs                                  |
| `planning.py`  | Value iteration: apply the Bellman update until values stop changing, then read off the best policy.         | Bellman updates, model-based planning |
| `qlearning.py` | Q-learning from sampled steps, without knowing `P` and `R`.                                                  | Model-free learning                   |
| `bayes.py`     | Beta-Bernoulli conjugate update and Thompson sampling on a 2-arm bandit.                                     | Bayesian inference, conjugate priors  |
| `pomdp.py`     | Generic belief update (Bayes filter) for any small POMDP. The Tiger problem is given as data.                | POMDPs                                |

## Setup

```bash
pip install numpy matplotlib jupyter
jupyter notebook