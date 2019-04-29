import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt

d=2; n=500
# mean
m1=np.array([0.,2.])[:,np.newaxis]
m2=np.array([3.,0.])[:,np.newaxis]
# variance
g_cov1=np.array([[1.3, 1.2],[1.2, 1.7]])
g_cov2=np.array([[1.4, -0.18],[-0.18, 1.4]])
# probability
p1=0.3
p2=1-p1
# data constellation
x1=np.random.multivariate_normal(m1.reshape(m1.shape[0]), g_cov1, n).T
x2=np.random.multivariate_normal(m2.reshape(m2.shape[0]), g_cov2, n).T


def gausscontour(c,m,xx,yy):
    xt=xx-m[0,0]
    yt=yy-m[1,0]
    ic=np.linalg.inv(c)
    p=1./(2.*np.pi*np.sqrt(np.linalg.det(c))) * \
        np.exp(-1./2.*(ic[0,0]*xt*xt+(ic[0,1]+ic[1,0])*xt*yt+ic[1,1]*yt*yt))
    return p

def recognition(x, mu, sigma, p):
    N = np.array(mu*np.ones(len(x.T)))
    return np.diag(-1./2.*(x-N).T.dot(LA.inv(sigma)).dot(x-N)) \
    +(-d/2.*np.log(2*np.pi)-1./2.*np.log(LA.norm(sigma, ord=1))+np.log(p))*np.ones(len(x.T))

def classification(cov1, cov2, name):
    w=m1-m2

    # recognition function ここを一般化するところから
    x0=1./2.*(m1+m2)-1./np.linalg.norm(m1-m2)**2.*np.log(p1/p2)*(m1-m2);
    l1=(w.T.dot(x1-x0)>0)[-1]
    l2=(w.T.dot(x2-x0)>0)[-1]
    l1=(recognition(x1, m1, cov1, p1)-recognition(x1, m2, cov2, p2)>0)
    l2=(recognition(x2, m1, cov1, p1)-recognition(x2, m2, cov2, p2)>0)

    [xx,yy]=np.meshgrid(np.linspace(-2, 5), np.linspace(-2, 5))
    plt.figure()
    plt.axis('equal')
    contour1=gausscontour(cov1, m1, xx, yy)
    plt.contour(xx, yy, contour1, cmap='hsv')
    contour2=gausscontour(cov2, m2, xx, yy)
    plt.contour(xx, yy, contour2, cmap='hsv')
    # correct x1
    plt.plot(x1[0,np.where(l1)], x1[1,np.where(l1)], 'bo', ms=3)
    # wrong x1
    plt.plot(x1[0,np.where(~l1)], x1[1,np.where(~l1)], 'ro', ms=3)
    # correct x2
    plt.plot(x2[0,np.where(1-l2)], x2[1,np.where(1-l2)], 'r^', ms=3)
    # wrong x2
    plt.plot(x2[0,np.where(l2)], x2[1,np.where(l2)], 'b^', ms=3)
    xxyy=np.c_[np.reshape(xx,-1), np.reshape(yy,-1)].T
    pp = recognition(xxyy, m1, cov1, p1)-recognition(xxyy, m2, cov2, p2)
    pp=np.reshape(pp, xx.shape)
    cs=plt.contour(xx, yy, pp, cmap='hsv') # 識別線
    plt.clabel(cs)
    # plt.savefig(name+'cov_diag.eps')
    plt.show()

if __name__ == '__main__':
    sigma = 1.5
    # variance-covariance matrix
    cov1=np.diag(np.cov(np.hstack([x1,x2])))*np.eye(d)
    cov2=cov1
    classification(cov1, cov2, "1")
    cov1=np.cov(np.hstack([x1,x2]))
    cov2=cov1
    classification(cov1, cov2, "2")
    cov1=np.cov(x1)
    cov2=np.cov(x2)
    classification(cov1, cov2, "3")
