import sys
sys.path.append("..")

import numpy as np
from mdp import make_simple_gridworld
from planning import value_iteration
from qlearning import q_learning, extract_policy
from bayes import BetaBernoulli
from pomdp import make_tiger_pomdp

def test_value_iteration_values():
    mdp = make_simple_gridworld(gamma=0.9)
    V, policy = value_iteration(mdp)
    
    assert abs(V[2] - 1.0) < 1e-4
    assert abs(V[1] - 0.89) < 1e-3
    assert abs(V[0] - 0.791) < 1e-3
    assert list(policy[:3]) == [1, 1, 1]
    print("Value Iteration test passed")

def test_qlearning_policy_agreement():
    mdp = make_simple_gridworld(gamma=0.9)
    _, vi_policy = value_iteration(mdp)
    Q, _ = q_learning(mdp, num_episodes=6000, alpha=0.1, epsilon=0.2)
    ql_policy = extract_policy(Q, mdp)
    
    assert np.array_equal(vi_policy, ql_policy)
    print("Q-Learning policy agreement test passed")

def test_conjugate_prior():
    prior = BetaBernoulli(1.0, 1.0)
    prior.update(successes=7, failures=3)
    assert prior.alpha == 8.0
    assert prior.beta == 4.0
    assert abs(prior.mean() - 0.6667) < 1e-3
    print("Conjugate prior test passed")

def test_tiger_belief_update():
    pomdp = make_tiger_pomdp()
    b = np.array([0.5, 0.5])
    b1 = pomdp.belief_update(b, "listen", "hear-left")
    b2 = pomdp.belief_update(b1, "listen", "hear-left")
    
    assert abs(b1[0] - 0.85) < 1e-4
    assert abs(b2[0] - 0.9698) < 1e-3
    print("Tiger belief update test passed")

if __name__ == "__main__":
    test_value_iteration_values()
    test_qlearning_policy_agreement()
    test_conjugate_prior()
    test_tiger_belief_update()
    print("\nAll basic tests passed!")