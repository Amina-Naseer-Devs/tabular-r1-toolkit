import numpy as np
from typing import List, Dict, Optional, Tuple

class TabularPOMDP:
    """
    Small Tabular POMDP.
    
    Components:
    - States
    - Actions
    - Observations
    - Transition: P[s, a, s']
    - Observation: O[a, s', o]   (probability of observing o after taking a and arriving in s')
    - Reward: R[s, a]
    - Discount gamma
    """
    
    def __init__(
        self,
        states: List[str],
        actions: List[str],
        observations: List[str],
        gamma: float = 0.95
    ):
        self.states = states
        self.actions = actions
        self.observations = observations
        self.n_states = len(states)
        self.n_actions = len(actions)
        self.n_obs = len(observations)
        self.gamma = gamma
        
        self.state_to_idx = {s: i for i, s in enumerate(states)}
        self.action_to_idx = {a: i for i, a in enumerate(actions)}
        self.obs_to_idx = {o: i for i, o in enumerate(observations)}
        
        # P[s, a, s']
        self.P = np.zeros((self.n_states, self.n_actions, self.n_states))
        # O[a, s', o]
        self.O = np.zeros((self.n_actions, self.n_states, self.n_obs))
        # R[s, a]
        self.R = np.zeros((self.n_states, self.n_actions))
    
    def belief_update(self, belief: np.ndarray, action: str, observation: str) -> np.ndarray:
        """
        Bayes filter (Belief update):
        
        b'(s') ∝ O(o | s', a) * Σ_s P(s' | s, a) * b(s)
        
        Returns normalized new belief.
        """
        a = self.action_to_idx[action]
        o = self.obs_to_idx[observation]
        
        # Predict: sum over previous states
        predicted = np.zeros(self.n_states)
        for sp in range(self.n_states):
            predicted[sp] = np.sum(self.P[:, a, sp] * belief)
        
        # Update with observation likelihood
        unnormalized = self.O[a, :, o] * predicted
        
        # Normalize
        total = np.sum(unnormalized)
        if total == 0:
            raise ValueError("Impossible observation: probability is 0 under the current belief.")
        return unnormalized / total
    
    def print_belief(self, belief: np.ndarray):
        for s, b in zip(self.states, belief):
            print(f"  {s}: {b:.4f}")


def make_tiger_pomdp(gamma: float = 0.95) -> TabularPOMDP:
    """
    Classic Tiger POMDP.
    
    States: tiger-left, tiger-right
    Actions: listen, open-left, open-right
    Observations: hear-left, hear-right
    
    Listen is noisy (85% correct).
    Opening the correct door: +10
    Opening the wrong door: -100
    Listening cost: -1
    """
    states = ["tiger-left", "tiger-right"]
    actions = ["listen", "open-left", "open-right"]
    observations = ["hear-left", "hear-right"]
    
    pomdp = TabularPOMDP(states, actions, observations, gamma)
    
    # Transitions
    # listen → state does not change
    pomdp.P[0, 0, 0] = 1.0  # tiger-left, listen → tiger-left
    pomdp.P[1, 0, 1] = 1.0  # tiger-right, listen → tiger-right
    
    # open-left or open-right → reset to 50-50 (new random tiger location)
    # (common simple version)
    for s in range(2):
        for a in [1, 2]:  # open-left, open-right
            pomdp.P[s, a, 0] = 0.5
            pomdp.P[s, a, 1] = 0.5
    
    # Observations (only matter after listen)
    # After listen:
    #   85% chance to hear the correct side
    pomdp.O[0, 0, 0] = 0.85  # listen, tiger-left, hear-left
    pomdp.O[0, 0, 1] = 0.15  # listen, tiger-left, hear-right
    pomdp.O[0, 1, 0] = 0.15  # listen, tiger-right, hear-left
    pomdp.O[0, 1, 1] = 0.85  # listen, tiger-right, hear-right
    
    # After opening doors, observation is not very meaningful
    # (we set uniform)
    for a in [1, 2]:
        pomdp.O[a, :, :] = 0.5
    
    # Rewards
    # listen always costs -1
    pomdp.R[:, 0] = -1.0
    
    # open-left
    pomdp.R[0, 1] = -100.0  # tiger is left → bad
    pomdp.R[1, 1] = +10.0    # tiger is right → good
    
    # open-right
    pomdp.R[0, 2] = +10.0    # tiger is left → good
    pomdp.R[1, 2] = -100.0   # tiger is right → bad
    
    return pomdp