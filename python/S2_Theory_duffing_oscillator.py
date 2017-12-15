import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections as mc

params = {'text.usetex' : True,
          'font.size' : 8,
          'font.family' : 'lmodern',
          'text.latex.unicode' : True}
plt.rcParams['text.latex.preamble']=[r'\usepackage{lmodern}']
plt.rcParams.update(params)
colors = [ 'dimgrey', 'royalblue', 'orange', 'seagreen', 'y' ]

fig_width = 7.5*2./2.54

def duffing_oscillator(x, t=None):

    # --> Initialize variables.
    dx = np.zeros_like(x)

    # --> x-equation.
    dx[0] = x[1]

    # --> y-equation.
    dx[1] = -0.5*x[1] + x[0] - x[0]**3.

    return dx

def streamlines_and_isoclines(x, y, u, v, xf=None, yf=None, savename=None):

    fig, ax = plt.subplots(1, 2, figsize=(fig_width, fig_width/3))

    streamlines(ax[0], x, y, u, v)
    ax[0].set_xlabel(r'$x$')
    ax[0].set_ylabel(r'$y$', rotation=0)

    isoclines(ax[1], x, y, u, v)

    if (xf is not None) and (yf is not None):
        ax[0].plot(xf, yf, 'ro')
        ax[1].plot(xf, yf, 'ro')

    if savename is not None:
        plt.savefig('../imgs/'+savename+'.pdf', bbox_inches='tight', dpi=300)

    return

def streamlines(ax, x, y, u, v, density=1):

    #-->
    magnitude = np.sqrt(u**2 + v**2)

    ax.streamplot(x, y, u, v, color=magnitude, cmap=plt.cm.inferno, density=density)

    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())

    ax.set_aspect('equal')
    ax.set_title(r'Streamlines')

    return

def isoclines(ax, x, y, u, v):

    ax.quiver(x[::4, ::4], y[::4, ::4], u[::4, ::4], v[::4, ::4], color='k')
    c1 = ax.contour(x, y, u, levels=[0.], colors=colors[1], label=r'$\dot{x}=0$')
    c2 = ax.contour(x, y, v, levels=[0.], colors=colors[2], label=r'$\dot({y}=0$')

    fmt = {}
    strs = [r'$\dot{x}=0$']
    for l, s in zip(c1.levels, strs):
        fmt[l] = s
    ax.clabel(c1, c1.levels, inline=True, fmt=fmt, fontsize=8)

    fmt = {}
    strs = [r'$\dot{y}=0$']
    for l, s in zip(c2.levels, strs):
        fmt[l] = s
    ax.clabel(c2, c2.levels, inline=True, fmt=fmt, fontsize=8)

    ax.set_xlabel(r'$x$')

    ax.set_yticklabels([])
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())

    ax.set_aspect('equal')
    ax.set_title(r'Isoclines')

    return

def duffing_jacobian(x):

    x = np.asarray(x)

    # --> Initialize variable.
    A = np.zeros((x.shape[0], x.shape[0]))

    # --> x-equation.
    A[0, 0] = 0.
    A[0, 1] = 1.

    # --> y-equation.
    A[1, 0] = 1. - 3*x[0]**2
    A[1, 1] = -0.5

    return A

if __name__ == '__main__':

    # --> Definition of the grid for the phase plane.
    x = np.linspace(-2, 2, 100)
    x, y = np.meshgrid(x, x)

    # --> Compute the time derivative vectors.
    xdot, ydot = np.zeros_like(x), np.zeros_like(y)
    xdot[:], ydot[:] = duffing_oscillator([x[:], y[:]])

    magnitude = np.sqrt(xdot[:]**2 + ydot[:]**2)

    # --> Determine the fixed points.
    xf = np.array([0., 1., -1.])
    yf = np.zeros_like(xf)


    # --> Numerical integration.
    from scipy.integrate import odeint
    from numpy.random import uniform
    t = np.linspace(0, 100, 1000.)


    # --> Stable and unstable manifolds.

    fig = plt.figure(figsize=(fig_width/3, fig_width/3))
    ax = fig.gca()

    ax.streamplot(x, y, xdot, ydot, color='lightgrey', density=0.5)

    t = np.linspace(0, 35, 1000)

    from scipy.linalg import eig
    A = duffing_jacobian([xf[0], yf[0]])
    _, eigenvectors = eig(A)

    x0 = 1e-8*eigenvectors[:, 1]
    x = odeint(duffing_oscillator, x0, np.flipud(t))
    ax.plot(x[:, 0], x[:, 1], color='royalblue', lw=3)
    x = odeint(duffing_oscillator, -x0, np.flipud(t))
    ax.plot(x[:, 0], x[:, 1], color='royalblue', lw=3)

    x0 = 1e-8*eigenvectors[:, 0]
    x = odeint(duffing_oscillator, x0, t)
    ax.plot(x[:, 0], x[:, 1], color='orange', lw=3)
    x = odeint(duffing_oscillator, -x0, t)
    ax.plot(x[:, 0], x[:, 1], color='orange', lw=3)

    ax.plot(xf, yf, 'ro')

    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.locator_params(axis='both', nbins=5)

    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')

    plt.savefig('../imgs/duffing_oscillator_saddle_manifold.pdf', bbox_inches='tight', dpi=300)

    plt.show()
