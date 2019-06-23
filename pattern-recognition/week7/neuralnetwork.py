import numpy as np
import matplotlib.pyplot as plt

# from linear import *
# from nonlinear import *
from slinear import *

nmid = 2

def sigmoid(u):
    return 1/(1+np.exp(-u))

def regularization(x, l):
    r_x = np.where(l == 1, x, -x)
    return r_x

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
    learning_rate = 1

    # 出力層
    x = np.reshape(x,(3,1))
    y = np.reshape(y,(2,1))
    d = np.reshape(d,(2,1))
    z1 = np.reshape(z1,(nmid,1))

    delta2 = (y-d)*y*(1-y)
    grad_W2 = delta2.dot(z1.T)
    W2 -= learning_rate*grad_W2

    # 中間層
    delta1 = W2.T.dot(delta2)*(z1)*(1-z1)
    grad_W1 = delta1.dot(x.T)
    W1 -= learning_rate*grad_W1

def epoch(x):
    global W1
    global W2
    for i in range(len(l)):
        y, z1 = forward_propagation(x[0:,i])
        d = np.vstack((np.where(l==1, 1, 0),np.where(l==-1, 1, 0)))
        back_propagation(x[:,i], z1, y, d[0:,i])

def plot(x, l):
    [xx,yy]=np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-2, 2, 100))
    xxyy=np.c_[np.ones(xx.size), np.reshape(xx,-1), np.reshape(yy,-1)].T
    pp,hoge = forward_propagation(xxyy)
    pp = pp[0,:]-pp[1,:]
    plt.figure()
    plt.plot(x[0,np.where(l==1)],x[1,np.where(l==1)],'bo')
    plt.plot(x[0,np.where(l==-1)],x[1,np.where(l==-1)],'bx')
    pp=np.reshape(pp, xx.shape)
    cs=plt.contour(xx, yy, pp, cmap='hsv') # 識別線
    plt.clabel(cs)
    plt.savefig('result'+str(nmid)+'.png')
    plt.show()

if __name__ == '__main__':
    # wを学習する
    ax = np.concatenate((np.ones((1,n)), x))
    rx = regularization(ax, l)
    W1 = np.random.normal(loc=0, scale=(1/3)**0.5, size=(nmid,3))
    W2 = np.random.normal(loc=0, scale=(1/nmid)**0.5, size=(2,nmid))
    for i in range(1000):
        print(i)
        epoch(ax)
    plot(x, l)
