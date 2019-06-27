import numpy as np
import matplotlib.pyplot as plt
import cvxopt
from linear import x, l, n
from nonlinear import x, l, n
from slinear import x, l, n
from qlinear import x, l, n

h = x*l
c = 1
qpP = cvxopt.matrix(h.T.dot(h))
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

w=np.sum(x*(np.ones(n)*(l*alpha)),axis=1)
sv=np.where((alpha>1e-5) & (alpha<c), True, False)
isv=np.where(sv)[-1]
b=np.sum(w.T.dot(x[:,isv])-l[isv])/np.sum(sv)
plt.figure()
plt.xlim([-1, 1])
plt.ylim([-1, 1])
plt.plot(x[0,np.where((l>0) & sv)], x[1,np.where((l>0) & sv)],'bo')
plt.plot(x[0,np.where((l>0) & ~sv)], x[1,np.where((l>0) & ~sv)],'bx')
plt.plot(x[0,np.where((l<0) & sv)], x[1,np.where((l<0) & sv)],'ro')
plt.plot(x[0,np.where((l<0) & ~sv)], x[1,np.where((l<0) & ~sv)],'rx')
if abs(w[0])>abs(w[1]):
  plt.plot([-1, 1],[(b+1+w[0])/w[1], (b+1-w[0])/w[1]])
  plt.plot([-1, 1],[(b-1+w[0])/w[1], (b-1-w[0])/w[1]])
else:
  plt.plot([(b+1+w[1])/w[0], (b+1-w[1])/w[0]], [-1, 1])
  plt.plot([(b-1+w[1])/w[0], (b-1-w[1])/w[0]], [-1, 1])
plt.savefig('soft_result.png')
plt.show()
