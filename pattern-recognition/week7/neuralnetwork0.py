import numpy as np
import numpy.linalg as LA
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import mnread


nmid = 200

def sigmoid(u):
    return 1/(1+np.exp(-u))

def forward_propagation(x):
    global W1
    global W2
    u1 = W1.dot(x)
    z1 = sigmoid(u1)
    u2 = W2.dot(z1)
    y = sigmoid(u2)
    return y, z1

def back_propagation(x, z1, y, d):
    global W1
    global W2
    learning_rate = 1e-3
    # 出力層
    x = np.reshape(x,(785,1))
    y = np.reshape(y,(10,1))
    d = np.reshape(d,(10,1))
    z1 = np.reshape(z1,(nmid,1))

    delta2 = (y-d)*y*(1-y)
    grad_W2 = delta2.dot(z1.T)
    W2 -= learning_rate*grad_W2

    # 中間層
    delta1 = W2.T.dot(delta2)*(z1)*(1-z1)
    grad_W1 = delta1.dot(x.T)
    W1 -= learning_rate*grad_W1

def epoch(data,label):
    global W1
    global W2
    ax = np.hstack((np.ones((len(data),1)), np.reshape(data,[data.shape[0],-1]))).T
    lset = set(label)
    l = np.empty((len(lset),data.shape[0]),dtype=float)
    for x in lset:
        l[x,:] = np.where(label==x, 1, -1)
    for i in range(ax.shape[1]):
        y, z1 = forward_propagation(ax[0:,i])
        back_propagation(ax[:,i], z1, y, l[0:,i])

def classify(data):
    data = np.hstack((np.ones((len(data),1)), np.reshape(data,[data.shape[0],-1]))).T
    label, gomi = forward_propagation(data)
    label = np.argmax(label, axis=0)
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
    W1 = np.random.normal(loc=0, scale=(1/785)**0.5, size=(nmid,785))
    W2 = np.random.normal(loc=0, scale=(1/nmid)**0.5, size=(10,nmid))
    num = 1000
    trlabel = mnread.readlabel(mnread.trlabelfz)
    trdata = mnread.readim(mnread.trdatafz)
    tstlabel = mnread.readlabel(mnread.tstlabelfz)[0:num]
    tstdata = mnread.readim(mnread.tstdatafz)[0:num]
    for i in range(10):
        print(i)
        epoch(trdata, trlabel)
    estlabel = classify(tstdata)
    evaluation(tstlabel, estlabel)
