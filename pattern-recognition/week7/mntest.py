import numpy as np
import numpy.linalg as LA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mnread

def train(label,data):
    # 各数字の多次元正規分布に対し、平均ベクトルと分散共分散行列を推定する
    data = np.reshape(data,[data.shape[0],-1]) # データ量行784列
    lset = set(label) # set型の生成
    p = np.empty(len(lset))
    mean = np.empty((len(lset),data.shape[1]),dtype=float) # 10行data列
    variance = np.empty((len(lset), data.shape[1], data.shape[1]), dtype=float)
    for x in lset:
        p[x] = data[np.where(label==x),:].shape[1]/data.shape[0]
        mean[x,:] = np.mean(data[np.where(label==x),:],axis=1) # labelがiのものを取り出し、データ毎の画素の平均値
        variance[x,:,:] = np.cov(data[np.where(label==x),:][0].T)
    return mean, variance, p

def classify(data, mean, variance, p):
    data = np.reshape(data,[data.shape[0],-1])
    label = np.empty(data.shape[0],dtype=int)
    # 識別関数を用いた推定
    # 分散共分散行列を決める
    alpha = 1e-10 # 正則化
    sigma = np.empty((mean.shape[0], data.shape[1], data.shape[1]))
    # 1. 標準偏差の２乗
    # for i in range(mean.shape[0]):
    #     sigma[:] = (np.var(data[:,:].T)+alpha)*np.eye(data.shape[1])
    # 2. 共通の共分散行列
    for i in range(mean.shape[0]):
        sigma[:] = np.cov(data[:,:].T)+alpha*np.eye(data.shape[1])
    # 3. 任意
    # sigma[:] = variance[:]+alpha*np.eye(data.shape[1])
    recognition = np.empty(mean.shape[0], dtype=float)
    for i in range(data.shape[0]):
        for j in range(mean.shape[0]):
            recognition[j] = -1./2.*(data[i,:]-mean[j,:]).T.dot(LA.inv(sigma[j])).dot(data[i,:]-mean[j,:]) \
            +(-1./2.*np.log(LA.norm(sigma[j], ord=1))+np.log(p[j]))
        label[i] = np.argmax(recognition)
        print(i)
    return label

if __name__ == '__main__':
    num = 1000
    trlabel = mnread.readlabel(mnread.trlabelfz)
    trdata = mnread.readim(mnread.trdatafz)
    tstlabel = mnread.readlabel(mnread.tstlabelfz)[0:num]
    tstdata = mnread.readim(mnread.tstdatafz)[0:num]

    mean, variance, p = train(trlabel, trdata)
    estlabel = classify(tstdata[0:num], mean, variance, p)
    # evaluation
    print('accuracy: %g' % (float(sum(estlabel==tstlabel)) / len(tstlabel)))

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
