import gym
import numpy as np
import matplotlib.pyplot as plt
from gym import wrappers

def take_action(s, w):
    return 1 if s.dot(w) > 0 else 0

def play_episode(env, params):
    observation = env.reset()
    done = False
    t = 0

    while not done and t < 10000:
        t += 1
        a = take_action(observation, params)
        observation, reward, done, info = env.step(a)
        if done:
            break
        
    return t

def play_multiple_episodes(env, T, params):
    episode_length = np.zeros(T)
    for i in range(T):
        episode_length[i] = play_episode(env, params)

    avg = episode_length.mean()
    print (f'Average Episode Length: {avg}')
    return avg

def random_search(env):
    episode_lengths = []
    best = 0
    best_params = None
    
    for i in range(100):
        params = np.random.random(4) * 2 - 1
        avg_length = play_multiple_episodes(env, 100, params)
        episode_lengths.append(avg_length)
        if avg_length > best:
            best = avg_length
            best_params = params
        
    return episode_lengths, params


if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    episode_lengths, params = random_search(env)
    plt.plot(episode_lengths)
    plt.show()

    env = wrappers.Monitor(env, 'rl2/cartpole')
    play_episode(env,params)