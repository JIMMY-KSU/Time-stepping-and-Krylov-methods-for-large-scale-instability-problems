#!/usr/bin/env python
"""Simulate linearized CGL and find BPOD modes and reduced-order model."""
import numpy as np
import scipy.linalg as spla

import modred as mr
import hermite as hr

plots = True
if plots:
    try:
        import matplotlib.pyplot as plt
    except:
        plots = False

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
    # Parameters for the subcritical case
    nx = 220
    dt = 1.0
    s = 1.6
    U = 2.0
    c_u = 0.2
    c_d = -1.0
    mu_0 = 0.23
    mu_2 = -0.01
    x_1 = -(-2 * (mu_0 - c_u ** 2) / mu_2) ** 0.5   # branch I
    x_2 = -x_1  # branch II
    x_s = x_2
    nu = U + 2j * c_u
    gamma = 1. + 1j * c_d
    chi = (-mu_2 / (2. * gamma)) ** 0.25    # chi: decay rate of global modes

    # Print parameters
    print('Parameters:')
    for var in [
        'nx', 'dt', 'U', 'c_u', 'c_d', 'mu_0', 'mu_2', 's', 'x_s', 'nu', 'gamma',
        'chi']:
        print('    %s = %s' % (var, str(eval(var))))

    # Collocation points in x are roughly [-85, 85], as in Ilak 2010
    x, Ds = hr.herdif(nx, 2, np.real(chi))

    # Inner product weights, trapezoidal rule
    weights = np.zeros(nx)
    weights[0] = 0.5 * (x[1] - x[0])
    weights[-1] = 0.5 * (x[-1] - x[-2])
    weights[1:-1] = 0.5 * (x[2:] - x[0:-2])
    M = np.diag(weights)
    inv_M = np.linalg.inv(M)
    M_sqrt = np.diag(weights ** 0.5)
    inv_M_sqrt = np.diag(weights ** -0.5)

    # LTI system arrays for direct and adjoint ("_adj") systems
    mu = (mu_0 - c_u ** 2) + mu_2 * x ** 2 / 2.
    A = -nu * Ds[0] + gamma * Ds[1] + np.diag(mu)


    # --> Resolvent analysis.
    Omega = np.linspace(-3, 3, 200)
    gain = np.zeros((4, Omega.size))

    from scipy.linalg import inv, svd

    for i, omega in enumerate(Omega):
        R = 1j*omega*np.eye(A.shape[0]) - A
        R = inv(R)
        u, s, vh = svd(M_sqrt.dot(R))
        gain[:, i] = s[:4]**2

    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_width/3))
    ax = axes[0]

    ax.semilogy(Omega, gain.T)

    ax.set_xlim(-3, 3)
    ax.set_xlabel(r'$\omega$')

    ax.set_ylim(1e-1, 1e3)
    ax.set_ylabel(r'$\mathcal{R}(\omega)$')


    idx = gain.argmax()
    omega = Omega[idx]
    R = 1j*omega*np.eye(A.shape[0]) - A
    R = inv(R)
    u, s, vh = svd(M_sqrt.dot(R))

    ax = axes[1]

    ax.plot(x, inv_M_sqrt.dot(vh.T[:, 0]).real, label=r'$\hat{\mathbf{f}}$')
    ax.plot(x, inv_M_sqrt.dot(u[:, 0]).real, ls='--', label=r'$\hat{\mathbf{u}}$')

    ax.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.35))

    ax.set_xlim(-85, 85)
    ax.set_xlabel(r'$x$')

    ax.set_ylabel(r'$\hat{\mathbf{f}}(\omega)$, $\hat{\mathbf{u}}(\omega)$')

    plt.tight_layout()

    plt.savefig('../imgs/S2_resolvent_analysis_bis.pdf', bbox_inches='tight', dpi=300)
    plt.show()
