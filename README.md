**Theek hai, ab bilkul clear.**

Neeche **ek hi box** mein **poora complete README** hai.  
Bas isko copy karke GitHub pe `README.md` mein paste kar do.

```markdown
# Tabular RL Toolkit

A from-scratch educational toolkit for understanding core Reinforcement Learning concepts using only NumPy.  
This project covers Tabular MDPs, Value Iteration, Q-Learning, Bayesian inference with Thompson Sampling, and POMDP belief updates.

## Introduction

Reinforcement Learning can feel abstract when you only read theory. This project makes the core ideas concrete by implementing them on small, understandable problems.

You will see:

- How the Bellman equation becomes a working algorithm (Value Iteration)
- How a model-free method (Q-Learning) can discover the same optimal policy as a model-based method
- How conjugate priors make Bayesian updates simple
- How an agent maintains a belief over hidden states (POMDP)

All algorithms are implemented from scratch using only NumPy.

## Modules

| Module         | What it does                                                                 | Concept Shown                        |
|----------------|------------------------------------------------------------------------------|--------------------------------------|
| `mdp.py`       | General `TabularMDP` class + simple gridworld                                | Markov Decision Processes            |
| `planning.py`  | Value Iteration (Bellman optimality updates)                                 | Model-based planning                 |
| `qlearning.py` | Tabular Q-Learning from experience                                           | Model-free learning                  |
| `bayes.py`     | Beta-Bernoulli conjugate prior + Thompson Sampling bandit                    | Bayesian inference & exploration     |
| `pomdp.py`     | Generic belief update (Bayes filter) + classic Tiger problem                 | Partially Observable MDPs            |

## Key Result

**Value Iteration** (needs the model) and **Q-Learning** (needs only experience) find the **same optimal policy** on the same gridworld. This is the strongest demonstration that both approaches are solving the same underlying problem.

## Results

### 1. Value Iteration

Gridworld with γ = 0.9, goal reward = +1, step cost = -0.01

| State                     | Value  | Policy |
|---------------------------|--------|--------|
| 0 (three steps from goal) | 0.7910 | Right  |
| 1 (two steps)             | 0.8900 | Right  |
| 2 (next to goal)          | 1.0000 | Right  |
| 3 (goal)                  | 0.0000 | —      |

These values match the hand-calculated Bellman updates:
- Next to goal: `V = 1.0`
- Two steps away: `V = -0.01 + 0.9 × 1.0 = 0.89`
- Three steps away: `V = -0.01 + 0.9 × 0.89 = 0.791`

### 2. Q-Learning vs Value Iteration

Q-Learning (8000 episodes) never sees the transition matrix `P` or reward function `R`, yet learns the identical policy:

```
Value Iteration policy : [1, 1, 1, -1]
Q-Learning policy      : [1, 1, 1, -1]
```

### 3. Bayesian Inference (Conjugate Prior)

Starting with `Beta(1,1)` (no prior knowledge):
- After observing 7 successes and 3 failures
- Posterior becomes `Beta(8,4)`
- Mean = 8/12 ≈ **0.667**

This shows why conjugate priors are powerful — the update is simply adding counts.

### 4. POMDP Belief Update (Tiger Problem)

Belief update rule:

`b'(s') ∝ P(o | s', a) × Σ_s P(s' | s, a) × b(s)`

Starting from uniform belief `[0.5, 0.5]` and repeatedly hearing “left”:

| Listens | Belief (tiger-left) |
|---------|---------------------|
| 1       | 0.8500              |
| 2       | 0.9698              |

The belief code is completely generic. The Tiger problem is just data.

## Project Structure

```
tabular-rl/
├── mdp.py
├── planning.py
├── qlearning.py
├── bayes.py
├── pomdp.py
├── notebooks/
│   ├── 01_mdp_and_value_iteration.ipynb
│   ├── 02_qlearning.ipynb
│   ├── 03_bayes_thompson.ipynb
│   └── 04_pomdp_belief.ipynb
├── tests/
│   └── test_basic.py
├── requirements.txt
└── README.md
```

## Setup & Usage

```bash
pip install -r requirements.txt
jupyter notebook
```

Then open the notebooks in order:

1. `01_mdp_and_value_iteration.ipynb`
2. `02_qlearning.ipynb`
3. `03_bayes_thompson.ipynb`
4. `04_pomdp_belief.ipynb`

You can also run the basic tests:

```bash
python tests/test_basic.py
```

## What I Learned

- How the Bellman optimality equation turns into a practical planning algorithm
- Why model-based and model-free methods can converge to the same solution
- How conjugate priors make Bayesian updating extremely simple
- How an agent can reason under partial observability using belief states

## Future Work

- Bayes-Adaptive MDPs (BAMDP)
- Interactive POMDPs (IPOMDP)
- Value iteration / point-based solvers for larger POMDPs


## Author

**Amina Naseer**  
BS Software Engineering  
National University of Modern Languages (NUML)
```
