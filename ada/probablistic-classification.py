import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

sample_size = 300
class_size = 3
h = 1e0 # Gauss kernel band-width
l = 1e0 # regularization term

def generate_sample(sample_size, class_size):
    test_y = np.repeat(np.arange(1,class_size+1), sample_size/class_size)
    test_x = np.repeat(np.arange(-3, 3+1e-10, 6./(class_size-1)), sample_size/class_size)+np.random.randn(sample_size)
    return test_x, test_y

def calc_design_matrix(x, c, h):
    return np.exp(-(x[None]-c[:,None])**2/(2*h**2))

def train(x, y):
    phi = calc_design_matrix(x, x, h)
    theta = np.zeros((class_size, sample_size))
    for i in range(class_size):
        pi = np.where(y == (i+1), 1, 0)
        theta[i] = np.linalg.inv(phi.T.dot(phi)+l*np.eye(len(x))).dot(phi.T).dot(pi)
    return theta

def classify(test_x, theta, x):
    phi = calc_design_matrix(x, test_x, h)
    p = np.zeros((class_size, 100))
    for i in range(class_size):
        p[i] = theta[i].dot(phi)
    p = np.where(p>0, p, 0)
    sum_p = np.sum(p, axis=0)
    p = p/sum_p
    return p

def figure(x, p, test_x, test_y):
    plt.figure()
    for i in range(class_size):
        plt.plot(x, p[i])
        plt.scatter(test_x[test_y == i+1], -i*2e-2*np.ones(len(test_x)//class_size))
    # plt.show()
    plt.savefig('result.png')

if __name__ == '__main__':
    test_x, test_y = generate_sample(sample_size, class_size)
    theta = train(test_x, test_y)
    x = np.linspace(-5, 5, 100)
    p = classify(test_x, theta, x)
    figure(x, p, test_x, test_y)
