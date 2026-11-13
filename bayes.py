import numpy as np
from typing import Tuple, List

class BetaBernoulli:
    """
    Beta-Bernoulli conjugate prior.
    
    Prior: Beta(α, β)
    After observing successes (s) and failures (f):
        Posterior = Beta(α + s, β + f)
    """
    
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        self.alpha = alpha
        self.beta = beta
    
    def update(self, successes: int, failures: int):
        """Update posterior with new observations"""
        self.alpha += successes
        self.beta += failures
    
    def mean(self) -> float:
        """Posterior mean = α / (α + β)"""
        return self.alpha / (self.alpha + self.beta)
    
    def sample(self) -> float:
        """Sample from posterior Beta distribution (for Thompson Sampling)"""
        return np.random.beta(self.alpha, self.beta)
    
    def __repr__(self):
        return f"Beta(α={self.alpha:.1f}, β={self.beta:.1f}) | mean={self.mean():.3f}"


def thompson_sampling_bandit(
    true_probs: List[float],
    num_rounds: int = 1000,
    seed: int = 42
) -> Tuple[List[int], List[float], List[BetaBernoulli]]:
    """
    2-arm (or multi-arm) Bernoulli Bandit with Thompson Sampling.
    
    Args:
        true_probs: list of true success probabilities for each arm
        num_rounds: number of rounds to play
    
    Returns:
        actions: list of chosen arms
        rewards: list of observed rewards
        posteriors: final Beta posteriors for each arm
    """
    np.random.seed(seed)
    n_arms = len(true_probs)
    
    # Start with uniform prior Beta(1,1) for each arm
    posteriors = [BetaBernoulli(1.0, 1.0) for _ in range(n_arms)]
    
    actions = []
    rewards = []
    
    for t in range(num_rounds):
        # Thompson Sampling: sample from each posterior and pick arm with highest sample
        samples = [post.sample() for post in posteriors]
        arm = int(np.argmax(samples))
        
        # Pull the arm (Bernoulli reward)
        reward = 1 if np.random.rand() < true_probs[arm] else 0
        
        # Update posterior
        if reward == 1:
            posteriors[arm].update(successes=1, failures=0)
        else:
            posteriors[arm].update(successes=0, failures=1)
        
        actions.append(arm)
        rewards.append(reward)
    
    return actions, rewards, posteriors


def demonstrate_conjugate_update():
    """
    Worked example from the project description:
    Start with Beta(1,1). See 7 successes and 3 failures.
    Posterior should be Beta(8,4), mean = 0.667
    """
    print("=== Conjugate Prior Worked Example ===")
    prior = BetaBernoulli(1.0, 1.0)
    print(f"Prior: {prior}")
    
    prior.update(successes=7, failures=3)
    print(f"After 7 successes + 3 failures: {prior}")
    print(f"Expected: Beta(8, 4), mean ≈ 0.667")
    print(f"Match: {abs(prior.mean() - 0.667) < 0.001}")