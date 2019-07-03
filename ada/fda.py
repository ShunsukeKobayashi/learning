# Fisher Discriminant Analysis.
from scipy.linalg import eigh
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
np.random.seed(46)


def generate_data(sample_size=100, pattern='two_cluster'):
    if pattern not in ['two_cluster', 'three_cluster']:
        raise ValueError('Dataset pattern must be one of '
                         '[two_cluster, three_cluster].')
    x = np.random.normal(size=(sample_size, 2))
    if pattern == 'two_cluster':
        x[:sample_size // 2, 0] -= 4
        x[sample_size // 2:, 0] += 4
    else:
        x[:sample_size // 4, 0] -= 4
        x[sample_size // 4:sample_size // 2, 0] += 4
    y = np.ones(sample_size, dtype=np.int64)
    y[sample_size // 2:] = 2
    return x, y


def fda(x, y):
    x = x - np.mean(x,axis=0)
    x_1 = x[np.where(y==1)]
    n_y1 = len(x_1)
    mu_y1 = np.reshape(np.mean(x_1,axis=0),(1,2))
    x_2 = x[np.where(y==2)]
    n_y2 = len(x_2)
    mu_y2 = np.reshape(np.mean(x_2,axis=0),(1,2))
    S_w = (x_1-mu_y1).T.dot(x_1-mu_y1)+(x_2-mu_y2).T.dot(x_2-mu_y2)
    S_b = n_y1*mu_y1.T.dot(mu_y1)+n_y2*mu_y2.T.dot(mu_y2)
    eigvals, eigvecs = eigh(S_b, S_w, eigvals_only=False)
    #昇順に並び替える
    index = np.argsort(eigvals)
    eigvals = eigvals[index]
    eigvecs = eigvecs[1]
    T = np.reshape(eigvecs,(1,2))
    return T

def visualize(x, y, T):
    plt.figure(1, (6, 6))
    plt.clf()
    plt.xlim(-7., 7.)
    plt.ylim(-7., 7.)
    plt.plot(x[y == 1, 0], x[y == 1, 1], 'bo', label='class-1')
    plt.plot(x[y == 2, 0], x[y == 2, 1], 'rx', label='class-2')
    plt.plot(np.array([-T[:, 0], T[:, 0]]) * 100,
             np.array([-T[:, 1], T[:, 1]]) * 100, 'k-')
    plt.legend()
    plt.savefig('lecture11-h1.png')


sample_size = 100
x, y = generate_data(sample_size=sample_size, pattern='two_cluster')
# x, y = generate_data(sample_size=sample_size, pattern='three_cluster')
T = fda(x, y)
visualize(x, y, T)
