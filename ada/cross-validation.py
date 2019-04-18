from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib

matplotlib.use('TkAgg')
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

if __name__ == '__main__':
    # create sample
    sample_size = 50
    xmin, xmax = -3, 3
    x, y = generate_sample(xmin=xmin, xmax=xmax, sample_size=sample_size)

    h = 10 # Gauss kernel band-width
    l = 0.03 # regularization term

    # S-fold cross-validation
    s = 5
    validation_times = int(sample_size/s)
    error = np.zeros(validation_times)
    for i in range(validation_times):
        train_x = np.concatenate((x[:s*i],x[s*(i+1):]))
        train_y = np.concatenate((y[:s*i],y[s*(i+1):]))
        test_x = x[s*i:s*(i+1)]
        test_y = y[s*i:s*(i+1)]

        # calculate design matrix
        k = calc_design_matrix(train_x, train_x, h)

        # solve the least square problem

        theta = np.linalg.solve(
            k.T.dot(k) + l * np.identity(len(k)),
            k.T.dot(train_y[:, None]))

        # evaluate test_data
        test_K = calc_design_matrix(train_x, test_x ,h)
        prediction = test_K.dot(theta)
        error[i] = (prediction.T+-1*test_y).dot((prediction.T + -1*test_y).T)
    print(np.sum(error))

    # create data to visualize the prediction
    X = np.linspace(start=xmin, stop=xmax, num=5000)
    K = calc_design_matrix(train_x, X, h)
    prediction = K.dot(theta)

    # visualization
    plt.clf()
    plt.scatter(x, y, c='green', marker='o')
    plt.plot(X, prediction)
    plt.savefig('l'+str(l)+'h'+str(h)+'.png')
