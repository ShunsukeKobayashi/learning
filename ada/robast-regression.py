from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

def generate_sample(xmin, xmax, sample_size):
    x = np.linspace(start=xmin, stop=xmax, num=sample_size)
    y = x+0.2*np.random.rand(sample_size)
    y[sample_size-1] = y[sample_size-2] = y[1] = -4
    return x, y

def model_function(theta, x):
    return theta[0]+theta[1]*x

def design_matrix(x):
    return np.concatenate([np.ones(len(x))[:,None], x[:,None]], axis=1)

def tukey_wait(r, eta):
    if abs(r) <= eta:
        return (1-(1-r**2/eta**2))**2
    else:
        return 0

def calc_new_theta(phi, wait, y):
    return np.linalg.inv(phi.T.dot(wait).dot(phi)).dot(phi.T).dot(wait).dot(y)

def regression(x, y, theta, eta):
    estimated_y = model_function(theta, x)
    wait = np.diag(list(map((lambda x: tukey_wait(x, eta)), y-estimated_y)))
    phi = design_matrix(x)
    return theta, calc_new_theta(phi, wait, y)

if __name__ == '__main__':
    sample_size = 10
    xmin, xmax = -3, 3
    eta = 1
    x, y = generate_sample(xmin, xmax, sample_size)
    theta_new = np.array([0,0]) # parameters
    step = 1
    N = 1000
    while(step <= N):
        theta_old, theta_new = regression(x, y, theta_new, eta)
        step = step+1
        if np.sum(np.abs(theta_new-theta_old)) < 1e-1:
            print("break at step",step)
            break
    X = np.linspace(-3, 3, 1000)
    prediction = model_function(theta_new, X)
    plt.clf()
    plt.scatter(x, y, c="green", marker="o")
    plt.plot(X, prediction)
    # plt.show()
    plt.savefig("robast-regression.png")
