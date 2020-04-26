import gym
env = gym.make('CartPole-v0')
env.reset() # puts in the start state
box = env.observation_space