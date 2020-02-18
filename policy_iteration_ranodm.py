import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid
from iterative_policy_evaluation import print_values, print_policy

SMALL_ENOUGH = 10E-4
GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')

# this is deterministic
# all p(s', r|s,a) = 1 or 0

if __name__ == '__main__':
    grid = negative_grid(step_cost = -1.0)

    #print rewards
    print('rewards')
    print_values(grid.rewards, grid)

    # state -> action
    # we'll randomly choose an action and update as we learn
    policy = {}
    for s in grid.actions.keys():
        policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)
    
    # initial policy
    print('initial policy')
    print_policy(policy, grid)

    # initialize V(s)
    V = {}
    states = grid.all_states()
    for s in states:
        if s in grid.actions:
            V[s] = np.random.random()
        else:
            V[s] = 0 # terminal state
    
    # repeat until convergence
    while True:
        # policy evaluation step
        while True:
            biggest_change = 0
            for s in states:
                old_v = V[s]
                new_v = 0
                if s in policy:
                    for a in ALL_POSSIBLE_ACTIONS:
                        if a == policy[s]:
                            p = 0.5
                        else:
                            p = 0.5/3
                    
                        grid.set_state(s)
                        r = grid.move(a)
                
                        new_v += p*(r + GAMMA * V[grid.current_state()])
                    V[s] = new_v
                    biggest_change = max(biggest_change, np.abs(old_v - V[s]))
            
            if biggest_change < SMALL_ENOUGH:
                break
        
        # policy improvement step
        is_policy_converged = True
        for s in states:
            if s in policy:
                old_a = policy[s]
                new_a = None
                best_value = float('-inf')
                # loop through all possible actions
                for a in ALL_POSSIBLE_ACTIONS:
                    v = 0
                    for a2 in ALL_POSSIBLE_ACTIONS:
                        if a == a2:
                            p=0.5
                        else:
                            p = 0.5/3
            
                        grid.set_state(s)
                        r = grid.move(a2)
                        v += p*(r + GAMMA * V[grid.current_state()])
                    if v>best_value:
                        best_value = v
                        new_a = a
                policy[s] = new_a
                if new_a != old_a:
                    is_policy_converged = False

        if is_policy_converged:
            break

    print('values:')
    print_values(V, grid)
    print('policy:')
    print_policy(policy, grid)

