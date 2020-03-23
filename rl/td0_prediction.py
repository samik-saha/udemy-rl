import numpy as np
import matplotlib.pyplot as plt
from grid_world import Grid, standard_grid, negative_grid
from iterative_policy_evaluation import print_values, print_policy

SMALL_ENOUGH = 10E-4
GAMMA = 0.9
ALPHA = 0.1 # learning rate
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')

def random_action(a, eps = 0.2):
    p = np.random.random()
    if p < (1 - eps):
        return a
    else:
        return np.random.choice(ALL_POSSIBLE_ACTIONS)

def play_game(grid: Grid, policy: dict):
    # returns a list of states and corresponding rewards
    # start at the designated start state
    s = (2,0)
    grid.set_state(s)
    states_and_rewards = [(s,0)] #list of tuples (state, reward)
    while not grid.game_over():
        a = policy[s]
        a = random_action(a)
        r = grid.move(a)
        s = grid.current_state()
        states_and_rewards.append((s, r))
    return states_and_rewards

if __name__ == '__main__':
    grid = standard_grid()

    print('rewards:')
    print_values(grid.rewards, grid)

    policy = {
        (2, 0): 'U',
        (1, 0): 'U',
        (0, 0): 'R',
        (0, 1): 'R',
        (0, 2): 'R',
        (1, 2): 'R',
        (2, 1): 'R',
        (2, 2): 'R',
        (2, 3): 'U',
    }

    # initialize V(s) as returns
    V = {}
    states = grid.all_states()
    for s in states:
        V[s] = 0

    # repeat until convergence
    for it in range(2000):
        # generate an episode using pi
        states_and_rewards = play_game(grid, policy)
        for t in range(len(states_and_rewards)-1):
            s, _ = states_and_rewards[t]
            s2, r = states_and_rewards[t+1]

            # we will update V(s) as we experience the episode
            V[s] = V[s] + ALPHA*(r + GAMMA * V[s2] - V[s])
    print('values:')
    print_values(V, grid)

    print('policy:')
    print_policy(policy, grid)
