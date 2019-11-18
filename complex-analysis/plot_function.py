"""
複素数の実部と虚部の等高線を描写
"""
import numpy as np
import matplotlib.pyplot as plt

def plot_contour(x, y, f):
    cont = plt.contour(x, y, f.real, 5, linestyles='solid')
    cont.clabel(fmt='%1.1f', fontsize=14)
    cont = plt.contour(x, y, f.imag, 5, linestyles='dashed')
    cont.clabel(fmt='%1.1f', fontsize=14)
    plt.show()

if __name__ == '__main__':
    x0, x1, dx = -10, 10, 0.1
    y0, y1, dy = -10, 10, 0.1

    x = np.arange(x0, x1, dx)
    y = np.arange(y0, y1, dy)
    x, y = np.meshgrid(x, y)
    c = x+y*1j
    # change below
    f = c**1
    # change above
    plot_contour(x, y, f)
