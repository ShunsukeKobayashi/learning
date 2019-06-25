# Locality preserving projection
from scipy.linalg import eigh
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def generate_sample(n):
    x = np.hstack((2*np.random.randn(n,1), np.random.randn(n,1)))
    # x = np.hstack((2*np.random.randn(n,1), 2*np.round(np.random.randn(n,1))-1+np.random.randn(n,1)/3.))
    return x

def calculate_similarity_matrix(x1):
    # distance-based
    t = 1
    distance = (x[None][:,:,0]-x[:,None][:,:,0])**2+(x[None][:,:,1]-x[:,None][:,:,1])**2
    W = np.exp(-distance/(2*t**2))
    return W

def get_T_lpp(x, W):
    w_i = np.sum(W, axis=1)
    D = np.diag(w_i)
    L = D-W
    A = x.T.dot(L).dot(x)
    B = x.T.dot(D).dot(x)
    eigvals, eigvecs = eigh(A, B, eigvals_only=False)
    x_s = eigvecs.T.dot(eigvecs).dot(x.T)
    #昇順に並び替える
    index = np.argsort(eigvals)
    eigvals = eigvals[index]
    eigvecs = eigvecs[index]
    T_lpp = eigvecs
    return T_lpp

def f_lpp(x, T_lpp):
    y = T_lpp[0][1]/T_lpp[0][0]*x
    return y

def plot_data(x, T_lpp):
    plt.figure()
    ax = plt.subplot()
    ax.set_ylim([-6,6])
    ax.set_xlim([-6,6])
    plt.scatter(x[:,0], x[:,1])
    plt_x = np.linspace(-6,6)
    plt_y = f_lpp(plt_x, T_lpp)
    plt.plot(plt_x,plt_y)
    plt.savefig('result.png')
    plt.show()

if __name__ == '__main__':
    sample_size = 100
    x = generate_sample(sample_size)
    W = calculate_similarity_matrix(x)
    T_lpp = get_T_lpp(x, W)
    plot_data(x, T_lpp)
