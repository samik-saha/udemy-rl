import numpy as np
import matplotlib.pyplot as plt

class Bandit:
    def __init__(self, m, upper_limit):
        self.m = m
        self.mean = upper_limit
        self.N = 1
    
    def pull(self):
        return np.random.randn() + self.m

    def update(self, x):
        self.N += 1
        self.mean = (1 - 1.0/self.N) * self.mean + 1.0/self.N * x

def run_experiment(m1, m2, m3, N):
    bandits = [Bandit(m1,10), Bandit(m2,10), Bandit(m3,10)]

    data = np.empty(N)

    for i in range(N):
        j = np.argmax([b.mean + np.sqrt(2*np.log(N)/(b.N+0.0001))for b in bandits])
        x = bandits[j].pull()
        bandits[j].update(x)

        #for the plot
        data[i] = x
    cumulative_average = np.cumsum(data) / (np.arange(N)+1)

    #plot moving average ctr
    plt.plot(cumulative_average)
    plt.plot(np.ones(N)*m1)
    plt.plot(np.ones(N)*m2)
    plt.plot(np.ones(N)*m3)
    plt.xscale('log')
    plt.show()

    for b in bandits:
        print(b.mean)

    return cumulative_average

if __name__ == '__main__':
    c_1 = run_experiment(1.0, 2.0, 3.0, 100000)

    #log scale plot
    plt.plot(c_1, label='ucb1 - log plot')
    plt.legend()
    plt.xscale('log')
    plt.show()

    #linear plot
    plt.plot(c_1, label='ucb1 - linear plot')
    plt.legend()
    plt.show()

