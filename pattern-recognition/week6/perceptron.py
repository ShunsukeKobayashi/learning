from linear import *
# from nonlinear import *
# from slinear import *

def initialization():
    rho = 5e-1
    theta = 1e-1
    k = 0
    w = np.random.randn(3)
    return w, rho, theta, k

def regularization(x, l):
    r_x = np.where(l == 1, x, -x)
    return r_x

def calculate_errors(x, l, est):
    j = []
    for i, data in enumerate(x[-1]):
        if(l[i] != est[i]):
            j.append(x.T[i])
    return np.array(j)

def classify_function(x, w, l):
    return np.where(w.T.dot(x) >= 0, 1, -1)

def update(w, j, rho):
    return w + np.sum(rho*j, axis=0)

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
    rx = regularization(ax, l)
    w, rho, theta, k = initialization()
    est = classify_function(ax, w, l)
    j = calculate_errors(rx, l, est)
    while k < 1e3:
        k += 1
        w = update(w, j ,rho)
        est = classify_function(ax, w, l)
        j = calculate_errors(rx, l, est)
        if np.sum(np.abs(rho*j))<=theta:
            break
    print(w)
    plot(x, l, w)
