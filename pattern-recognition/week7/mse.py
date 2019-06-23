from linear import *

import numpy as np
import numpy.linalg as LA
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import mnread

def train(data, label):
    ax= np.hstack((np.ones((len(data),1)), np.reshape(data,[data.shape[0],-1]))).T
    lset = set(label)
    l = np.empty((len(lset),data.shape[0]),dtype=float)
    for x in lset:
        l[x,:] = np.where(label==x, 1, 0)
    X = ax.T
    b = l.T
    alpha = 1e-10
    w = np.linalg.inv(X.T.dot(X)+alpha*np.eye(ax.shape[0])).dot(X.T).dot(b)
    return w

def classify(data, w):
    data= np.hstack((np.ones((len(data),1)), np.reshape(data,[data.shape[0],-1])))
    label = np.argmax(data.dot(w), axis=1)
    return label

def evaluation(tstlabel, estlabel):
    print('accuracy: %g' % (float(sum(estlabel==tstlabel))/len(tstlabel)))

    plt.figure()
    plt.suptitle('goods')
    goods = np.random.permutation(np.where(estlabel==tstlabel)[-1])[range(50)]
    for i,good in enumerate(goods):
        plt.subplot(5,10,i+1)
        plt.axis('off')
        plt.imshow(tstdata[good,:,:],cmap='gray')
        plt.title(estlabel[good])
    plt.savefig('good.png')
    plt.figure()
    plt.suptitle('bads')
    bads = np.random.permutation(np.where(~(estlabel==tstlabel))[-1])[range(50)]
    for i,bad in enumerate(bads):
        plt.subplot(5,10,i+1)
        plt.axis('off')
        plt.imshow(tstdata[bad,:,:],cmap='gray')
        plt.title('%s(%s)' % (estlabel[bad], tstlabel[bad]))
    plt.savefig('bad.png')
    plt.show()

if __name__ == '__main__':
    num = 1000
    trlabel = mnread.readlabel(mnread.trlabelfz)
    trdata = mnread.readim(mnread.trdatafz)
    tstlabel = mnread.readlabel(mnread.tstlabelfz)[0:num]
    tstdata = mnread.readim(mnread.tstdatafz)[0:num]

    w = train(trdata, trlabel)
    estlabel = classify(tstdata, w)
    evaluation(tstlabel, estlabel)
