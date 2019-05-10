import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
import scipy.linalg

from scipy.io import loadmat

def read_data(datapath):
    dataset = loadmat(datapath)
    x1 = dataset['x1'][0]
    x2 = dataset['x2'][0]
    return x1, x2

def gaussian_parzen_window(x, xi, hn):
    return 1./hn*1./np.sqrt(2.*np.pi)*np.exp(-((x-xi)/hn)**2./2.)

def box_parzen_window(x, xi, hn):
    return 1./hn*(np.abs(x-xi)/hn<=0.5)

def plot_probabilities(dataset):
    h1l=np.array([5.,1., 0.5, 0.1])
    x=np.linspace(-3.,3.,num=500)
    plt.figure()
    for i in range(len(h1l)):
        h1 = h1l[i]
        n = len(dataset[0])
        hn = h1/np.sqrt(n)
        p = np.zeros((2,len(x)))
        for j, data in enumerate(dataset):
            for xi in data:
                p[j] = p[j]+gaussian_parzen_window(x, xi, hn)
            p[j] = p[j]/n
            plt.subplot(len(h1l), 2, 2*i+1)
            plt.plot(x, p[j])
        for j in range(len(dataset)):
            postp = np.zeros((2,len(x)))
            postp[j] = p[j]/2./(p[0]/2.+p[1]/2.)
            plt.subplot(len(h1l), 2, 2*(i+1))
            plt.plot(x, postp[j])
    # plt.show()
    plt.figure()
    for i in range(len(h1l)):
        h1 = h1l[i]
        n = len(dataset[0])
        hn = h1/np.sqrt(n)
        p = np.zeros((2,len(x)))
        for j, data in enumerate(dataset):
            for xi in data:
                p[j] = p[j]+box_parzen_window(x, xi, hn)
            p[j] = p[j]/n
            plt.subplot(len(h1l), 2, 2*i+1)
            plt.plot(x, p[j])
        for j in range(len(dataset)):
            postp = np.zeros((2,len(x)))
            postp[j] = p[j]/2./(p[0]/2.+p[1]/2.)
            plt.subplot(len(h1l), 2, 2*(i+1))
            plt.plot(x, postp[j])
    # plt.show()
    plt.figure()
    nl=np.array([1, 4, 16, 64])
    for i in range(len(nl)):
        n=nl[i]
        k=np.sqrt(n)
        p=np.zeros((2,len(x)))
        for j, data in enumerate(dataset):
            s = np.random.choice(data, n, replace=False)
            for l in range(len(x)):
                # r: sorted list by the distance to x[l]
                r=sorted(abs(s-x[l]))
        	# r[int(k)-1]: k-th distance
                p[j][l]=k/(n*2.*r[int(k)-1])
            ax=plt.subplot(len(nl), 2, 2*i+1)
            plt.plot(x,p[j])
            if k==1:
                ax.set_ylim([0, 10])
        for j in range(len(dataset)):
            postp = np.zeros((2,len(x)))
            postp[j] = p[j]/2./(p[0]/2.+p[1]/2.)
            plt.subplot(len(nl), 2, 2*(i+1))
            plt.plot(x, postp[j])
    plt.show()


if __name__ == '__main__':
    x1, x2 = read_data('data.mat')
    x = np.stack((x1, x2))
    plot_probabilities(x)
