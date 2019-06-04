# adaptive regularization classification
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def generate_sample(sample_size):
    x = np.random.randn(sample_size, 3)
    x[:sample_size//2,0] -= 15
    x[sample_size//2:,0] -= 5
    y = np.concatenate([np.ones(sample_size//2, dtype=int),-np.ones(sample_size//2, dtype=int)])
    x[1:3,0] += 10
    x[:,2] = 1
    index = np.random.permutation(np.arange(sample_size))
    x = x[index]
    y = y[index]
    return x,y

def train(mu, sigma, x, y):
    gamma = 1
    mu = mu + (y*max(0, 1-mu.T.dot(x)*y)/(x.T.dot(sigma).dot(x)+gamma))*sigma.dot(x)
    sigma = sigma - sigma.dot(x).dot(x.T).dot(sigma)/(x.T.dot(sigma).dot(x)+gamma)
    return mu,sigma

if __name__ == '__main__':
    sample_size = 50
    x, y = generate_sample(sample_size)
    mu = np.zeros((len(x[-1]), 1))
    sigma = np.eye(len(x[-1]), len(x[-1]))
    for i in range(len(x)):
        mu, sigma = train(mu, sigma, x[i][:,None], y[i])
    plt.figure()
    plt.scatter(x[y==1,0], x[y==1,1])
    plt.scatter(x[y==-1,0], x[y==-1,1])
    plt.ylim([-2,2])
    plt.plot([-20,0], -(mu[2]+np.array([-20,0])*mu[0])/mu[1], color='green')
    plt.show()
