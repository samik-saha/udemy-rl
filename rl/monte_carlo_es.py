import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid
from iterative_policy_evaluation import print_values, print_policy

SMALL_ENOUGH = 10E-4
GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')

# Only policy evaluation not optimization

def play_game(grid, policy):
    # returns a list of states and corresponding returns

    # reset game to start at a random position
    # we need to do this, because given our current deterministic
    # policy we would never end up at certain states

    start_states = list(grid.actions.keys())
    start_idx = np.random.choice(len(start_states))
    grid.set_state(start_states[start_idx])

    s = grid.current_state()
    a = np.random.choice(ALL_POSSIBLE_ACTIONS)

    # each triple is s(t), a(t) and r(t)
    # but r(t) results from takig action a(t-1) from state s(t-1) and landing in s[t]
    states_actions_rewards = [(s,a,0)] 
    seen_states = set()
    seen_states.add(grid.current_state())
    num_steps = 0

    while True:
        r = grid.move(a)
        num_steps += 1
        s = grid.current_state()

        if s in seen_states:
            reward = -10./num_steps
            states_actions_rewards.append((s, None, reward))
            break # end episode
        elif grid.game_over():
            states_actions_rewards.append((s, None, r))
            break
        else:
            a = policy[s]
            states_actions_rewards.append((s,a,r))
        seen_states.add(s)

    # calculate the returns by working backwards
    G = 0
    states_actions_returns = []
    first = True
    for s, a, r in reversed(states_actions_rewards):
        # the value of the terminal state is 0 by definition
        # we should ignore the first state we encounter
        # and ignore the last G, which is meaningless, since it does not correspond
        if first:
            first = False
        else:
            states_actions_returns.append((s, a, G))
        G = r + GAMMA * G
    states_actions_returns.reverse()
    return states_actions_returns

def max_dict(d: dict):
    max_key = None
    max_val = float('-inf')

    for k,v in d.items():
        if v > max_val:
            max_val = v
            max_key = k
    
    return max_key, max_val

if __name__ == '__main__':
    grid = negative_grid(step_cost = -0.1)

    print('rewards:')
    print_values(grid.rewards, grid)

    # state -> action
    policy = {}
    for s in grid.actions.keys():
        policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)

    # initialize Q(s, a) and returns
    Q = {}
    returns = {}
    states = grid.all_states()
    for s in states:
        if s in grid.actions:
            Q[s] = {}
            for a in ALL_POSSIBLE_ACTIONS:
                Q[s][a] = 0
                returns[(s,a)] = []
        else:
            pass # terminal state or state we cannot get to

    # repeat
    deltas = []
    for t in range(2000):
        if t % 1000 == 0:
            print(t)

        # generate an episode using pi
        biggest_change = 0
        states_actions_returns = play_game(grid, policy)
        seen_state_action_pairs = set()

        for s, a, G in states_actions_returns:
            # check if we have already seen s
            # called "first-visit" MC policy evaluation
            sa = (s, a)
            if sa not in seen_state_action_pairs:
                old_q = Q[s][a]
                returns[sa].append(G)
                Q[s][a] = np.mean(returns[sa])
                biggest_change = max(biggest_change, np.abs(old_q - Q[s][a]))
                seen_state_action_pairs.add(sa)

        deltas.append(biggest_change)

        # update policy
        for s in policy.keys():
            policy[s] = max_dict(Q[s])[0]
        
    plt.plot(deltas)
    plt.show()

    print('final policy')
    print_policy(policy, grid)

    V = {}
    for s, Qs in Q.items():
        V[s] = max_dict(Q[s])[1]
    
    
    print('final values:')
    print_values(V, grid)
