import numpy as np
from mdp import TabularMDP
from typing import List, Tuple

def q_learning(
    mdp: TabularMDP,
    num_episodes: int = 5000,
    alpha: float = 0.1,
    epsilon: float = 0.1,
    epsilon_decay: float = 0.9995,
    min_epsilon: float = 0.01,
    seed: int = 42
) -> Tuple[np.ndarray, List[float]]:
    """
    Tabular Q-Learning (model-free)
    
    Returns:
        Q          : action-value table (n_states x n_actions)
        returns    : list of total return per episode (for learning curve)
    """
    np.random.seed(seed)
    
    Q = np.zeros((mdp.n_states, mdp.n_actions))
    returns = []
    
    for ep in range(num_episodes):
        s = 0  # start state
        total_return = 0.0
        done = False
        steps = 0
        max_steps = 100
        
        while not done and steps < max_steps:
            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                a = np.random.choice(mdp.get_actions(s))
            else:
                a = np.argmax(Q[s])
            
            # Take action
            next_s, reward = mdp.step(s, a)
            total_return += reward
            
            # Q-learning update
            if mdp.is_terminal(next_s):
                td_target = reward
                done = True
            else:
                td_target = reward + mdp.gamma * np.max(Q[next_s])
            
            Q[s, a] = Q[s, a] + alpha * (td_target - Q[s, a])
            
            s = next_s
            steps += 1
        
        returns.append(total_return)
        
        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
    
    return Q, returns


def extract_policy(Q: np.ndarray, mdp: TabularMDP) -> np.ndarray:
    """Greedy policy from Q-table"""
    policy = np.full(mdp.n_states, -1, dtype=int)
    for s in range(mdp.n_states):
        if mdp.is_terminal(s):
            continue
        policy[s] = np.argmax(Q[s])
    return policy


def print_q_and_policy(mdp: TabularMDP, Q: np.ndarray, policy: np.ndarray):
    action_names = {0: "Left ", 1: "Right", -1: " ---- "}
    
    print("State |    Q(Left)   |   Q(Right)  | Policy")
    print("------|--------------|-------------|--------")
    for s in range(mdp.n_states):
        if mdp.is_terminal(s):
            print(f"  {s}   |       -      |      -      |  ---- ")
        else:
            print(f"  {s}   |  {Q[s,0]:10.4f}  | {Q[s,1]:10.4f}  | {action_names[policy[s]]}")