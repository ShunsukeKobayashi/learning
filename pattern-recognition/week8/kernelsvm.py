import numpy as np
import matplotlib.pyplot as plt
import cvxopt
from linear import x, l, n
from nonlinear import x, l, n
from slinear import x, l, n
from qlinear import x, l, n

def kernel(x,c):
    t = 0.01
    distance = (x[None][:,:,0]-c[:,None][:,:,0])**2+(x[None][:,:,1]-c[:,None][:,:,1])**2
    W = np.exp(-distance/(2*t**2))
    return W

def classify(x,c,t,alpha,l):
    term1 = np.sum(alpha*l.dot(kernel(x,c,t)))
    return term1

h = x*l
c = 1
qpP = cvxopt.matrix(kernel(x.T,x.T))
qpq = cvxopt.matrix(-np.ones(n), (n, 1))
qpG = cvxopt.matrix(np.vstack((-np.eye(n),np.eye(n))))
qph = cvxopt.matrix(np.hstack((np.zeros(n),c*np.ones(n))), (2*n, 1))
qpA = cvxopt.matrix(l.astype(float), (1, n))
qpb = cvxopt.matrix(0.)
cvxopt.solvers.options['abstol'] = 1e-5
cvxopt.solvers.options['reltol'] = 1e-10
cvxopt.solvers.options['show_progress'] = False
res=cvxopt.solvers.qp(qpP, qpq, qpG, qph, qpA, qpb)
alpha = np.reshape(np.array(res['x']),-1)
sv=np.where((alpha>1e-5) & (alpha<c), True, False)
isv=np.where(sv)[-1]

# make contour
[xx,yy]=np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-2, 2, 100))
xxyy=np.c_[np.reshape(xx,-1), np.reshape(yy,-1)]
term1 = kernel(x.T,xxyy).dot(alpha*l)
term2 = np.sum(kernel(x[:,:].T,x[:,isv].T).dot(alpha*l)-l[isv])/np.sum(sv)
pp = term1-term2
plt.figure()
plt.xlim([-1, 1])
plt.ylim([-1, 1])
plt.plot(x[0,np.where(l>0)], x[1,np.where(l>0)],'bx')
plt.plot(x[0,np.where(l<0)], x[1,np.where(l<0)],'rx')
# plt.plot(x[0,np.where((l>0) & sv)], x[1,np.where((l>0) & sv)],'bo')
# plt.plot(x[0,np.where((l>0) & ~sv)], x[1,np.where((l>0) & ~sv)],'bx')
# plt.plot(x[0,np.where((l<0) & sv)], x[1,np.where((l<0) & sv)],'ro')
# plt.plot(x[0,np.where((l<0) & ~sv)], x[1,np.where((l<0) & ~sv)],'rx')
pp=np.reshape(pp, xx.shape)
cs=plt.contour(xx, yy, pp, cmap='hsv') # 識別線
plt.clabel(cs)
plt.savefig('kernel_result.png')
plt.show()
