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

if __name__ == '__main__':

    def Jacobian_matrix(mu, Re):

        # --> Initialize variable.
        A = np.zeros((2, 2))

        # --> Fill-in the matrix.
        A[0, 0] = 1./100. - 1./Re
        A[1, 0] = mu
        A[1, 1] = -2/100.

        return A

    def dynamical_system(x, t, A):

        dx = np.zeros_like(x)
        dx[0] = A[0, 0]*x[0] + A[0, 1]*x[1]
        dx[1] = A[1, 0]*x[0] + A[1, 1]*x[1]

        return dx


    # --> Normal case.
    L = Jacobian_matrix(0., 50.)

    # --> Non-normal case.
    A = Jacobian_matrix(1., 50.)

    # --> Compute eigenvalues and eigenvectors.
    from scipy.linalg import eig, expm
    eigenvalues, v, u = eig(A, right=True, left=True)

    angle = np.angle(u[0, 1] + 1j*u[1, 1])

    # --> Random initial perturbation.
    from numpy.random import uniform
    from scipy.linalg import norm, expm
    from scipy.integrate import odeint

    x0 = np.array([1., 1.])
    x0 /= norm(x0)

    t = np.linspace(0, 400, 1000)
    x1 = odeint(dynamical_system, x0, t, args=(A,))

    # --> Compute phase plane.
    y = np.linspace(-10, 20, 20)
    x = np.linspace(-1, 1., 20)
    x, y = np.meshgrid(x, y)

    xdot, ydot = np.zeros_like(x), np.zeros_like(y)
    xdot[:], ydot[:] = dynamical_system([x[:], y[:]], None, A)

    fig = plt.figure(figsize=(fig_width, fig_width/3))
    ax = fig.gca()

    ax.annotate("", xy=(10*u[0, 1], 10*u[1, 1]), xytext=(0, 0), arrowprops=dict(arrowstyle="-", lw=2))
    ax.annotate("", xy=(-10*u[0, 1], -10*u[1, 1]), xytext=(0, 0), arrowprops=dict(arrowstyle="-", lw=2))

    ax.annotate("", xy=(10*u[0, 0], 10*u[1, 0]), xytext=(0, 0), arrowprops=dict(arrowstyle="-", lw=2))
    ax.annotate("", xy=(-10*u[0, 0], -10*u[1, 0]), xytext=(0, 0), arrowprops=dict(arrowstyle="-", lw=2))


    ax.streamplot(x, y, xdot, ydot, color='gray', density=0.5, linewidth=0.5)
    ax.plot(x1[:, 0], x1[:, 1], color='royalblue', lw=2, ls='--')

    ax.plot(x1[0, 0], x1[0, 1], '>', ms=6, color='orange', label=r'Initial Condition')
    ax.plot(0., 0., 'o', ms=6, color='red', zorder=8, label=r'Stable fixed point')

    ax.set_xlim(-0.2, 1.)
    ax.set_ylim(-10, 20)

    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')

    ax.legend(loc=0)
    ax.locator_params(axis='y', nbins=4)
    plt.savefig('../imgs/S2_Theory_explanation_transient_growth.pdf', bbox_inches='tight', dpi=300)


    ###################################



    def optimal_gain(A, T):
        from scipy.linalg import expm, svdvals

        G = np.zeros_like(T)

        for i, t in enumerate(T):
            G[i] = svdvals(expm(A*t))[0]**2

        return G

    t = np.linspace(0, 400, 40001)
    G = optimal_gain(A, t)

    fig = plt.figure(figsize=(fig_width/3, fig_width/3))
    ax = fig.gca()

    ax.semilogy(t, G, color='royalblue')

    ax.set_ylim(0.8, 1000)

    ax.set_xlabel(r'$T$')
    ax.set_ylabel(r'$\mathcal{G}(T)$')

    plt.savefig('../imgs/S2_Theory_illustration_optimal_perturbation.pdf', bbox_inches='tight', dpi=300)


    # --> Optimal initial condition.
    from scipy.linalg import svd, expm
    T = t[G.argmax()]
    _, s, vh = svd(expm(A*T))

    x0 = -(vh.T)[:, 0]

    t = np.linspace(0, 400, 1000)
    x2 = odeint(dynamical_system, x0, t, args=(A,))

    # --> Compute phase plane.
    y = np.linspace(-10, 30, 40)
    x = np.linspace(-0.2, 1.2, 20)
    x, y = np.meshgrid(x, y)

    xdot, ydot = np.zeros_like(x), np.zeros_like(y)
    xdot[:], ydot[:] = dynamical_system([x[:], y[:]], None, A)

    fig = plt.figure(figsize=(fig_width, fig_width/3))
    ax = fig.gca()

    ax.annotate("", xy=(10*u[0, 1], 10*u[1, 1]), xytext=(0, 0), arrowprops=dict(arrowstyle="-", lw=2))
    ax.annotate("", xy=(-10*u[0, 1], -10*u[1, 1]), xytext=(0, 0), arrowprops=dict(arrowstyle="-", lw=2))

    ax.annotate("", xy=(10*u[0, 0], 10*u[1, 0]), xytext=(0, 0), arrowprops=dict(arrowstyle="-", lw=2))
    ax.annotate("", xy=(-10*u[0, 0], -10*u[1, 0]), xytext=(0, 0), arrowprops=dict(arrowstyle="-", lw=2))


    ax.streamplot(x, y, xdot, ydot, color='gray', density=0.5, linewidth=0.5)
    ax.plot(x2[:, 0], x2[:, 1], color='royalblue', lw=2, ls='--')
    ax.plot(x1[:, 0], x1[:, 1], color='orange', lw=2, ls='-.')

    ax.plot(x1[0, 0], x1[0, 1], 's', ms=6, color='orange', label=r'Random unit-norm initial condition')
    ax.plot(x2[0, 0], x2[0, 1], '>', ms=6, color='royalblue', label=r'Opt. initial condition')
    ax.plot(0., 0., 'o', ms=6, color='red', zorder=8, label=r'Fixed point')

    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-10, 30)

    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')

    ax.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.25))
    ax.locator_params(axis='y', nbins=4)
    plt.savefig('../imgs/S2_Theory_illustration_optimal_perturbation_bis.pdf', bbox_inches='tight', dpi=300)


    plt.show()
