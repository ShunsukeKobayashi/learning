import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
import pandas as pd

h = 5e0 # Gauss kernel band-width
l = 1e-2 # regularization term

def read_csv(file_path, num):
    csv_data = pd.read_csv('digits/'+file_path+str(num)+'.csv', header=None, dtype=float)
    csv_data = np.array(csv_data)
    return csv_data

def make_train_label(data):
    labels = -1*np.ones((10,len(data)))
    data_num = int(len(data)/10)
    for i in range(10):
        labels[i][i*data_num:(i+1)*data_num] = 1
    return labels

def make_test_label(data, num):
    label = num*np.ones(len(data))
    return label

def get_train_data(file_name):
    split_num = 1
    data = read_csv(file_name, 0)
    for i in range(1, 10):
        tmpdata = read_csv(file_name, i)
        data = np.concatenate((data, tmpdata))
    label = make_train_label(data[::split_num])
    return data[::split_num], label

def get_test_data(file_name):
    data = read_csv(file_name, 0)
    label = make_test_label(data, 0)
    for i in range(1, 10):
        tmpdata = read_csv(file_name, i)
        data = np.concatenate((data, tmpdata))
        tmplabel = make_test_label(tmpdata, i)
        label =  np.concatenate((label, tmplabel))
    return data, label

def calc_design_matrix(x, c, h):
    return np.exp(-np.sum((x[:,None]-c[None])**2, axis=-1)/(2*h**2))

def train(data, label):
    k = calc_design_matrix(data, data, h)
    phi = np.zeros((10, len(k)))
    for i in range(10):
        phi[i][:] = LA.inv(k.dot(k)+l*np.identity(len(k))).dot(k.T).dot(label[i][:])
    return k, phi

def classify(train, test, k, phi):
    k = calc_design_matrix(test, train, h)
    label = np.zeros(len(test))
    for i in range(len(label)):
        label[i] = np.argmax(k.dot(phi.T)[i])
    return label

def plot_data(index, data, est_label, test_label, title):
    plt.figure()
    plt.suptitle(title)
    for i,indice in enumerate(index):
        plt.subplot(5, 9, i+1)
        plt.axis('off')
        plt.imshow(data[indice].reshape(16, 16), cmap='gray')
        if test_label == []:
            plt.title('%s' % int(est_label[indice]))
        else:
            plt.title('%s(%s)' % (int(est_label[indice]), int(test_label[indice])))
    plt.savefig(title)

if __name__ == '__main__':
    train_data, train_label = get_train_data('digit_train')
    test_data, test_label = get_test_data('digit_test')
    k, phi = train(train_data, train_label)
    est_label = classify(train_data, test_data, k, phi)
    good_num = 0
    print('accuracy: %g' %(float(sum(est_label==test_label))/len(test_label)))
    goods = np.random.permutation(np.where(est_label==test_label)[-1])[range(45)]
    bads = np.random.permutation(np.where(~(est_label==test_label))[-1])[range(45)]
    plot_data(goods, test_data, est_label, [], 'good')
    plot_data(bads, test_data, est_label, test_label, 'bad')
    plt.show()
