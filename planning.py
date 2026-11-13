import numpy as np
from mdp import TabularMDP

def value_iteration(
    mdp: TabularMDP,
    theta: float = 1e-8,
    max_iterations: int = 1000
) -> tuple[np.ndarray, np.ndarray]:
    """
    Value Iteration
    Returns: V (values), policy
    """
    V = np.zeros(mdp.n_states)
    
    for it in range(max_iterations):
        delta = 0.0
        V_new = np.zeros_like(V)
        
        for s in range(mdp.n_states):
            if mdp.is_terminal(s):
                V_new[s] = 0.0
                continue
            
            q_values = []
            for a in mdp.get_actions(s):
                q = mdp.R[s, a] + mdp.gamma * np.dot(mdp.P[s, a], V)
                q_values.append(q)
            
            V_new[s] = max(q_values)
            delta = max(delta, abs(V_new[s] - V[s]))
        
        V = V_new
        if delta < theta:
            break
    
    # Greedy policy
    policy = np.full(mdp.n_states, -1, dtype=int)
    
    for s in range(mdp.n_states):
        if mdp.is_terminal(s):
            continue
        
        best_q = -np.inf
        best_a = 0
        for a in mdp.get_actions(s):
            q = mdp.R[s, a] + mdp.gamma * np.dot(mdp.P[s, a], V)
            if q > best_q:
                best_q = q
                best_a = a
        policy[s] = best_a
    
    return V, policy


def print_values_and_policy(mdp: TabularMDP, V: np.ndarray, policy: np.ndarray):
    action_names = {0: "Left ", 1: "Right", -1: " ---- "}
    
    print("State |   Value   | Policy")
    print("------|-----------|--------")
    for s in range(mdp.n_states):
        print(f"  {s}   |  {V[s]:7.4f}  | {action_names.get(policy[s], policy[s])}")