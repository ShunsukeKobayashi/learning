import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
import scipy.linalg

from scipy.io import loadmat

def make_contour(mean, cov):
    icov = LA.inv(cov)
    xx,yy=np.meshgrid(np.linspace(-5,5,1e3),np.linspace(-5,5,1e3))
    xt=xx-mean[0,0]
    yt=yy-mean[0,1]
    p=1./(2.*np.pi*np.sqrt(LA.det(cov))) * np.exp(-1./2.*(icov[0,0]*xt*xt+(icov[0,1]+icov[1,0])*xt*yt+icov[1,1]*yt*yt))
    plt.contour(xx,yy,p,cmap='hsv',levels=7)

data = loadmat('data2.mat')
x = data['x']  # data matrix
cov1 = data['cov1']  # covariance matrix 1
cov2 = data['cov2']  # covariance matrix 2
m1 = data['m1']  # mean vector 1
m2 = data['m2']  # mean vector 2
plt.figure()
plt.axis('equal')
plt.scatter([x[:,0]],[x[:,1]],s=10)
make_contour(m1, cov1)
make_contour(m2, cov2)
plt.savefig('simudiagonal-before.eps')

phi = LA.eig(cov1)[1] # eigenvalue vector of cov1
theta = LA.inv(phi).dot(cov1).dot(phi)
theta_inv_root = LA.inv(scipy.linalg.sqrtm(theta))
Y = theta_inv_root.dot(phi.T).dot(x.T)
I = theta_inv_root.dot(phi.T).dot(cov1).dot(phi).dot(theta_inv_root)
K = theta_inv_root.dot(phi.T).dot(cov2).dot(phi).dot(theta_inv_root)

psi = LA.eig(K)[1] # eigenvalue vector of K
Lambda = LA.inv(psi).dot(K).dot(psi)
Z = psi.T.dot(Y)
plt.figure()
plt.axis('equal')
plt.scatter([Z[0,:]],[Z[1,:]],s=10)
m3 = m1.dot(psi.T.dot(theta_inv_root).dot(phi.T).T) # m1 after simultaneous-diagonalization
m4 = m2.dot(psi.T.dot(theta_inv_root).dot(phi.T).T) # m2 after simultaneous-diagonalization
make_contour(m3, I)
make_contour(m4, Lambda)
plt.savefig('simudiagonal-after.eps')

# plt.show()
