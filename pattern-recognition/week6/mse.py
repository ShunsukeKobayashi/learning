from linear import *
# from nonlinear import *
# from slinear import *

def plot(x, l, w):
    [xx,yy]=np.meshgrid(np.linspace(-2, 2), np.linspace(-2, 2))
    xxyy=np.c_[np.ones(xx.size), np.reshape(xx,-1), np.reshape(yy,-1)].T
    pp = w[0:3].T.dot(xxyy)
    plt.figure()
    plt.plot(x[0,np.where(l==1)],x[1,np.where(l==1)],'bo')
    plt.plot(x[0,np.where(l==-1)],x[1,np.where(l==-1)],'bx')
    pp=np.reshape(pp, xx.shape)
    cs=plt.contour(xx, yy, pp, cmap='hsv') # 識別線
    plt.clabel(cs)
    # plt.show()
    plt.savefig('result.png')

if __name__ == '__main__':
    ax = np.concatenate((np.ones((1,n)), x))
    X = ax.T
    b = l.T
    w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(b)
    print(w)
    plot(x, l, w)
