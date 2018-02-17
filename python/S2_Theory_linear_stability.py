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

fig_width = 11.7/2.54

if __name__ == '__main__':

    def Jacobian_matrix(Re):

        # --> Initialize variable.
        A = np.zeros((2, 2))

        # --> Fill-in the matrix.
        A[0, 0] = 1./100. - 1./Re
        A[1, 0] = 1.
        A[1, 1] = -2/100.

        return A

    def dynamical_system(x, t, A):

        dx = A.dot(x)

        return dx.ravel()

    # --> Stable case.
    As = Jacobian_matrix(50.)

    # --> Unstable case.
    Au = Jacobian_matrix(125.)

    # --> Simulate both cases.
    from scipy.integrate import odeint
    from scipy.linalg import norm
    from numpy.random import uniform

    t = np.linspace(0, 400)
    x0 = uniform(low=-1., high=1., size=(2,))
    x0 /= norm(x0)

    xs = odeint(dynamical_system, x0, t, args=(As,))
    xu = odeint(dynamical_system, x0, t, args=(Au,))

    # --> Plot figure.
    fig = plt.figure(figsize=(fig_width/3, fig_width/3))
    ax = fig.gca()

    ax.semilogy(t, xs[:, 0]**2 + xs[:, 1]**2, color='orange', ls='--', label=r'$Re=50$')
    ax.semilogy(t, xu[:, 0]**2 + xu[:, 1]**2, color='royalblue', label=r'$Re=125$')

    # --> Decorators.
    ax.set_xlabel(r'$t$')
    ax.set_xlim(t.min(), t.max())

    ax.set_ylabel(r'$\| \mathbf{x} \|_2^2$')

    ax.legend(loc=0)

    plt.savefig('../imgs/S2_Theory_illustration_linear_stability.pdf', bbox_inches='tight', dpi=1200)

    plt.show()
