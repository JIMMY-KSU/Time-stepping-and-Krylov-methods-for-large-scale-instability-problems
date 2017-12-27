import scipy.linalg
import scipy.optimize
import scipy.interpolate

from numpy import flipud, sum, where, dot, real, arange, abs, diag, zeros, argsort, sin, \
    isfinite, tile, delete, linspace, fliplr, pi, eye, zeros_like, ones, cos, newaxis, concatenate, exp, \
    mod, argmax, exp, sqrt, ones_like, imag, array, isfinite, matrix, inf, loadtxt, unravel_index, logspace, meshgrid

from matplotlib import pyplot as plt
import pystab
from pystab.utils import chebdif, chebdif_weights

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
from matplotlib import ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable

params = {'text.usetex' : True,
          'font.size' : 8,
          'font.family' : 'lmodern',
          'text.latex.unicode' : True}
plt.rcParams['text.latex.preamble']=[r'\usepackage{lmodern}']
plt.rcParams.update(params)

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

def diff_matrices(problem_setup):

    n = problem_setup['Number of points']
    y, DM_c = chebdif(n, 2)

    return y, DM_c

def generalized_eigenproblem(problem_setup, baseflow, diff_matrices):

    # -----     Miscellaneous     -----

    re = problem_setup['Reynolds number']
    alpha = problem_setup['Streamwise wavenumber']
    beta = problem_setup['Spanwise wavenumber']

    d = diff_matrices[0, :, :]
    d2 = diff_matrices[1, :, :]

    u = baseflow['Baseflow profile']
    du = baseflow['Baseflow shear']

    # ----- Construct the A matrix -----

    n = d.shape[0]
    k2 = alpha ** 2. + beta ** 2.

    I = eye(n).astype(d2.dtype)

    laplacian = d2 - k2 * I

    Lu = -1j * alpha * diag(u) + (1./ re) * laplacian
    Cu = -diag(du)
    Zu = zeros_like(Lu)

    Lv = -1j * alpha * diag(u) + (1./ re) * laplacian
    Zv = zeros_like(Lv)

    Lw = -1j * alpha * diag(u) + (1./ re) * laplacian
    Zw = zeros_like(Lw)

    if beta == 0:
        Au = concatenate((Lu, Cu), axis=1)
        Av = concatenate((Zv, Lv), axis=1)

        A = concatenate((Au, Av), axis=0)

        # ----- Construct the pressure projector -----

        I = eye(n)

        dpdx = -1j * alpha * I
        dpdy = -d

        gradient = concatenate((dpdx, dpdy), axis=0)
        divergence = concatenate((dpdx, dpdy), axis=1)
        A = concatenate((A, gradient), axis=1)
        tmp = concatenate((divergence, zeros((n, n))), axis=1)
        A = concatenate((A, tmp), axis=0)

        bc = I
        bc[0,0] = 0.
        bc[-1,-1] = 0.

        B = scipy.linalg.block_diag(bc, bc, zeros_like(bc))

        A[0, :] = 0
        A[0, 0] = 1.

        A[n-1, :] = 0
        A[n-1, n-1] = 1.

        A[n, :] = 0
        A[n, n] = 1

        A[2*n-1, :] = 0
        A[2*n-1, 2*n-1] = 1.

    else:
        Au = concatenate((Lu, Cu, Zu), axis=1)
        Av = concatenate((Zv, Lv, Zv), axis=1)
        Aw = concatenate((Zw, Zw, Lw), axis=1)

        A = concatenate((Au, Av, Aw), axis=0)

        # ----- Construct the pressure projector -----

        I = eye(n)

        dpdx = -1j * alpha * I
        dpdy = -d
        dpdz = -1j * beta * I

        gradient = concatenate((dpdx, dpdy, dpdz), axis=0)
        divergence = concatenate((dpdx, dpdy, dpdz), axis=1)

        A = concatenate((A, gradient), axis=1)
        tmp = concatenate((divergence, zeros((n, n))), axis=1)
        A = concatenate((A, tmp), axis=0)

        bc = I
        bc[0,0] = 0.
        bc[-1,-1] = 0.

        B = scipy.linalg.block_diag(bc, bc, bc, zeros_like(bc))

        A[0, :] = 0.
        A[0, 0] = 1.

        A[n-1, :] = 0.
        A[n-1, n-1] = 1.

        A[n, :] = 0.
        A[n, n] = 1.

        A[2*n-1, :] = 0.
        A[2*n-1, 2*n-1] = 1.

        A[2*n, :] = 0.
        A[2*n, 2*n] = 1.

        A[3*n-1, :] = 0.
        A[3*n-1, 3*n-1] = 1.

    eigenspectrum, eigenmodes = scipy.linalg.eig(A, B)

    idx = argsort(-eigenspectrum.real)
    eigenspectrum = eigenspectrum[idx]
    eigenmodes = eigenmodes[:, idx]


    idx = np.argwhere(abs(eigenspectrum)<1000).flatten()
    eigenspectrum = eigenspectrum[idx]
    eigenmodes = eigenmodes[:, idx]

    return eigenspectrum, eigenmodes

def schur_eigenproblem(problem_setup, baseflow, diff_matrices):

    A, B = generalized_eigenproblem(problem_setup, baseflow, diff_matrices)
    C = scipy.linalg.inv(A)
    C = C.dot(B)

    def sorting(val):
        if abs(val)>1e-8:
            return True
        else:
            return False

    T, Z, sdim = scipy.linalg.schur(C, sort=sorting)
    Q1 = Z[:, :sdim]
    G = T[:sdim, :sdim]
    Ginv = scipy.linalg.inv(G)

    eigenspectrum, y = scipy.linalg.eig(Ginv)
    eigenmodes = Q1.dot(y)

    idx = argsort(-eigenspectrum.real)
    eigenspectrum = eigenspectrum[idx]
    eigenmodes = eigenmodes[:, idx]

    return eigenspectrum, eigenmodes


def energy_norm(problem_setup, grid):

    weights = chebdif_weights(n ,boxsize=2.)
    w = weights
    if problem_setup['Spanwise wavenumber'] == 0.:
        Q = scipy.linalg.block_diag(w, w, zeros_like(w))
    else:
    	Q = scipy.linalg.block_diag(w, w, w, zeros_like(w))
    return Q

if __name__ == '__main__':

    # --> Import various things.
    import numpy as np
    import pystab

    #--> Set the parameters.
    re = 300    # Reynolds number
    n = 64      # Number of grid points.

    beta = np.array([1., 2., 3., 4.])
    gain = np.zeros_like(beta)

    # --> Setup figure.
    fig = plt.figure(figsize=(fig_width/2, fig_width/3))
    ax = fig.gca()

    for j in xrange(beta.size):

        problem_setup = {'Reynolds number': re,
                         'Number of points': n,
                         'Streamwise wavenumber': 0,
                         'Spanwise wavenumber': beta[j]
                         }

        ls = pystab.StabilityProblem(problem_setup=problem_setup,
                                     fixed_point='couette',
                                     diff_matrices=diff_matrices)

        eigenspectrum, eigenmodes = generalized_eigenproblem(ls.problem_setup,
                                                              ls.fixed_point,
                                                              ls.diff_matrices)

        # --> Energy norm.
        time, energy = ls.optimal_perturbation(T=250, nt=2500, nsave=1,
                                               inner_product=energy_norm,
                                               eigenpairs=[eigenspectrum, eigenmodes]
                                               )

        # --> Gain.
        gain[j] = energy.max()

        # --> Plot figure.
        ax.plot(time, energy, label=r'$\beta = %i$' %beta[j])


    # --> Decorators for the figure.
    ax.set_xlim(0, 250)
    ax.set_xlabel(r'$T$')

    ax.set_ylim(0, 105)
    ax.set_ylabel(r'$\mathcal{G}(T)$')

    ax.legend(loc=0)

    plt.savefig('../imgs/S2_optimal_perturbation_couette_flow_gain.pdf', bbox_inches='tight', dpi=1200)




    # --> Perform analysis to plot the optimal perturbation and optimal response.

    beta = 2.
    problem_setup = {'Reynolds number': re,
                     'Number of points': n,
                     'Streamwise wavenumber': 0,
                     'Spanwise wavenumber': beta
                     }

    ls = pystab.StabilityProblem(problem_setup=problem_setup,
                                 fixed_point='couette',
                                 diff_matrices=diff_matrices)

    eigenspectrum, eigenmodes = generalized_eigenproblem(ls.problem_setup,
                                                          ls.fixed_point,
                                                          ls.diff_matrices)

    # --> Energy norm.
    time, energy, opt_pert, opt_resp = ls.optimal_perturbation(T=250, nt=2500, nsave=1,
                                                               inner_product=energy_norm,
                                                               eigenpairs=[eigenspectrum, eigenmodes],
                                                               return_modes=True
                                                               )
    idx = energy.argmax()

    opt_pert = opt_pert[:, 0, idx].ravel()
    u, v, w = opt_pert[:n], opt_pert[n:2*n], opt_pert[2*n:3*n]
    y = ls.grid
    z = np.linspace(-2., 2., 100)

    y, z = np.meshgrid(y, z)
    v = v.real * np.cos(2*beta*z) - v.imag * np.sin(2*beta*z)
    w = w.real * np.cos(2*beta*z) - w.imag * np.sin(2*beta*z)

    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_width/3))


    ax = axes[0]
    ax.quiver(z[::2, ::2], y[::2, ::2], w[::2, ::2], v[::2, ::2])

    ax.set_aspect('equal')
    ax.set_xlim(-2, 2)
    ax.set_xlabel(r'$z$')

    ax.set_ylim(-1, 1)
    ax.set_ylabel(r'$y$')

    opt_resp = opt_resp[:, 0, idx].ravel()
    u, v, w = opt_resp[:n], opt_resp[n:2*n], opt_resp[2*n:3*n]

    u = u.real * np.cos(2*beta*z) - u.imag * np.sin(2*beta*z)


    ax = axes[1]
    ax.contour(z, y, u, cmap=plt.cm.RdBu)

    ax.set_aspect('equal')
    ax.set_xlim(-2, 2)
    ax.set_xlabel(r'$z$')

    ax.set_ylim(-1, 1)
    ax.set_yticklabels([])

    plt.savefig('../imgs/S2_optimal_perturbation_couette_flow.pdf', bbox_inches='tight', dpi=300)
    plt.show()
