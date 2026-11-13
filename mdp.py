import numpy as np
from typing import Optional, List, Tuple

class TabularMDP:
    """
    General Tabular Markov Decision Process.
    
    P[s, a, s'] = probability of going to s' when taking action a in state s
    R[s, a]     = expected reward for taking action a in state s
    """
    
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        gamma: float = 0.9,
        P: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
        terminal_states: Optional[List[int]] = None
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        
        if P is None:
            self.P = np.zeros((n_states, n_actions, n_states))
        else:
            self.P = np.array(P, dtype=float)
            
        if R is None:
            self.R = np.zeros((n_states, n_actions))
        else:
            self.R = np.array(R, dtype=float)
            
        self.terminal_states = terminal_states if terminal_states is not None else []
    
    def is_terminal(self, s: int) -> bool:
        return s in self.terminal_states
    
    def get_actions(self, s: int) -> List[int]:
        if self.is_terminal(s):
            return []
        return list(range(self.n_actions))
    
    def step(self, s: int, a: int) -> Tuple[int, float]:
        """Sample next state and reward"""
        if self.is_terminal(s):
            return s, 0.0
        next_s = np.random.choice(self.n_states, p=self.P[s, a])
        reward = self.R[s, a]
        return int(next_s), float(reward)


def make_simple_gridworld(gamma: float = 0.9) -> TabularMDP:
    """
    Simple chain:
        0 → 1 → 2 → 3 (Goal)
    
    Actions: 0 = Left, 1 = Right
    Living reward = -0.01
    Goal reward = +1
    """
    n_states = 4
    n_actions = 2
    terminal = [3]
    
    P = np.zeros((n_states, n_actions, n_states))
    R = np.zeros((n_states, n_actions))
    
    # State 0
    P[0, 0, 0] = 1.0
    P[0, 1, 1] = 1.0
    R[0, :] = -0.01
    
    # State 1
    P[1, 0, 0] = 1.0
    P[1, 1, 2] = 1.0
    R[1, :] = -0.01
    
    # State 2 (next to goal)
    P[2, 0, 1] = 1.0
    P[2, 1, 3] = 1.0
    R[2, 0] = -0.01
    R[2, 1] = +1.0
    
    # State 3 (Goal) - absorbing
    P[3, :, 3] = 1.0
    R[3, :] = 0.0
    
    return TabularMDP(n_states, n_actions, gamma, P, R, terminal)