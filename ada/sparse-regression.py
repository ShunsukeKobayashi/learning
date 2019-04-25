from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

np.random.seed(0)  # set the random seed for reproducibility

def generate_sample(xmin, xmax, sample_size):
    x = np.linspace(start=xmin, stop=xmax, num=sample_size)
    pix = np.pi * x
    target = np.sin(pix) / pix + 0.1 * x
    noise = 0.05 * np.random.normal(loc=0., scale=1., size=sample_size)
    return x, target + noise

def calc_design_matrix(x, c, h):
    return np.exp(-(x[None] - c[:, None]) ** 2 / (2 * h ** 2))

def calcMSE(theta, z):
    return hoge

if __name__ == "__main__":
    # create sample
    sample_size = 50
    xmin, xmax = -3, 3
    x, y = generate_sample(xmin=xmin, xmax=xmax, sample_size=sample_size)

    period = 1000
    Lambda = 0.03
    h = 1

    theta = np.zeros(sample_size)
    z = np.zeros(sample_size)
    u = np.zeros(sample_size)

    time = 1
    # update parameters
    while(time < period):
        K = calc_design_matrix(x, x, h)
        I = np.identity(sample_size)
        y_hat = K.dot(theta)
        error = (y_hat-y).T.dot(y_hat-y)+Lambda*np.linalg.norm(z, ord=1)+u.T.dot(theta-z)+(theta-z).T.dot(theta-z)*0.5
        theta = np.linalg.inv(K.T.dot(K)+I).dot(K.T.dot(y)+z-u)
        z = np.max(np.array([np.zeros(sample_size), theta+u-Lambda*np.ones(sample_size)]), axis=0) - \
        np.max(np.array([np.zeros(sample_size), -theta-u-Lambda*np.ones(sample_size)]), axis=0)
        u = u+theta-z
        time = time+1

    # create data to visualize the prediction
    X = np.linspace(start=xmin, stop=xmax, num=sample_size)
    K = calc_design_matrix(X, X, h)
    prediction = K.dot(theta)

    # count how many variables = 0
    print("all variables:",len(theta))
    print("zero variables:",np.sum(np.abs(theta)<1e-6))

    # visualization
    plt.clf()
    plt.scatter(x, y, c="green", marker="o")
    plt.plot(X, prediction)
    # plt.show()
    plt.savefig("sparse-regression.png")
