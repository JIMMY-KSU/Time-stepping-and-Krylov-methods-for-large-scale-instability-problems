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

    omega = np.linspace(0, 2, 500)
    H = 1./(1. + 1j*omega)

    fig = plt.figure(figsize=(fig_width/3, fig_width/3))
    ax = fig.gca()

    ax.plot(omega, H.real, color='royalblue')
    ax.plot(omega, np.heaviside(-(omega-1), 1), color='gray', ls='--')

    ax.set_xlim(0, 2)
    ax.set_ylim(-0.05, 1.05)

    ax.set_xlabel(r'$\omega$')
    ax.set_ylabel(r'$\Re(\hat{\mathcal{H}})$')

    plt.savefig('../imgs/S3_SFD_transfer_function.pdf', bbox_inches='tight', dpi=1200)
    plt.show()
