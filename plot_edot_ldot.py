#!/usr/bin/env python3

import argparse
import numpy as np
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import figures as figs
import scipy.signal as sps




plt.rc('font', family='serif')
mpl.rcParams['text.usetex'] = True

red     = [237/255, 102/255,  93/255]
blue    = [114/255, 158/255, 206/255]
purp    = [123/255, 102/255, 210/255]
green   = [105/255, 183/255, 100/255]
orange  = [255/255, 187/255, 120/255]
babyblu = [174/255, 199/255, 232/255]
babyred = [255/255, 152/255, 150/255]

t_nu = 1e3
rbin = 32. / 1000




def E(e, M=1.0, a=1.0, mu=0.25):
    return -mu * M / 2. / a




def L(e, M=1.0, a=1.0, mu=0.25): 
    return mu * np.sqrt(M * a * (1 - e * e))




def midpoint(x):
    return (x[:-1] + x[1:]) / 2




def smooth(A):
    A = np.array(A)
    return np.array([A[0]] + list((A[2:] + A[1:-1] + A[:-2]) / 3) + [A[-1]])




def smooth_savgol(A, window_size=5, polyorder=3):
    return sps.savgol_filter(A, window_length=window_size, polyorder=polyorder)




def plot_cumsum(ax, x, y, c=None, **kwargs):
    ax.fill_between(x, np.cumsum(y), facecolor='purple', alpha=0.1)
    return ax.plot(x, np.cumsum(y), c=c, **kwargs)




def config_ax1(ax, e):
    ax.set_xlim([ 0.0 , 5.0])
    ax.set_ylim([-4e-6, 4e-6])
    ax.set_yticks([])
    ax.set_ylabel(r'$d \dot{e} / dr$' + ' [Arb.]', fontsize=12)
    ax.axhline(0.0, color='grey', lw=0.5)




def config_ax2(ax, e):
    ax.set_ylim([-1e-6, 1e-6])
    ax.set_yticks([])
    ax.set_ylabel(r'$\dot e (r)$ [Arb.]', fontsize=12)

    box = dict(boxstyle='round', facecolor='white', alpha=0.2)
    ax.text(0.015, 0.85, r'e = ' + str(e), transform=ax.transAxes, bbox=box)




def plot_edot_ldot_single(axa, axb, e, r, s, Edot, Ldot, edot):
    print(Edot.shape)
    rel_Edot = Edot / E(e) / 2.
    rel_Ldot = Ldot / L(e)     
    lab1 = 'Relative power'
    lab2 = 'Relative torque'

    config_ax1(axa, e)
    axa.plot(midpoint(r), smooth_savgol(-rel_Edot, window_size=7) / rbin, lw=2, color=red , alpha=0.8, label=lab1)
    axa.plot(midpoint(r), smooth_savgol(+rel_Ldot, window_size=7) / rbin, lw=2, color=blue, alpha=0.8, label=lab2)
    
    config_ax2(axb, e)
    axb.axhline(edot * e / (1 - e**2), color='green', ls='--', lw=0.8, alpha=0.7, label=r'Integrated $\dot e$')
    plot_cumsum(axb, midpoint(r), -rel_Edot - rel_Ldot, ls='--', lw=0.75, color='purple', label=r'Integrated difference')




def config_axes(axs, key, ecc):
    if key == 'u':
        axs[0].text(0.02, 1.02, r'Mean anomaly = $\pi$ / 2', transform=axs[0].transAxes)
    if key == 'd':
        axs[0].text(0.02, 1.02, r'Mean anomaly = $3\pi$ / 2', transform=axs[0].transAxes)
    if key == 'f':
        axs[0].text(0.02, 1.02, r'200 Orbit Average (1200-1400)', transform=axs[0].transAxes)
    axs[-1].set_xlabel(r'$r \, / \, a$', fontsize=14)




def load_radial_data(fname):
    h5f = h5py.File(fname, 'r')
    r   = h5f['radial_bins'][...]
    s   = h5f['sigma'      ][...]
    P   = h5f['work_on'    ][...]
    T   = h5f['torque_on'  ][...]
    return r, s, P, T




def get_radial_series(key):
    ecc  = [0.025, 0.1, 0.3, 0.4, 0.5, 0.6, 0.75]
    e025 = load_radial_data('./Data/time_avg_reductions_e025.h5')
    e100 = load_radial_data('./Data/time_avg_reductions_e100.h5')
    e300 = load_radial_data('./Data/time_avg_reductions_e300.h5')
    e400 = load_radial_data('./Data/time_avg_reductions_e400.h5')
    e500 = load_radial_data('./Data/time_avg_reductions_e500.h5')
    e600 = load_radial_data('./Data/time_avg_reductions_e600.h5')
    e750 = load_radial_data('./Data/time_avg_reductions_e750.h5')
    data = [e025, e100, e300, e400, e500, e600, e750]
    return ecc, data




def plot_edot_ldot(ecc, data, key):
    e_de_edot = np.load('./Data/e_de_dedt.npy')
    e_ts = e_de_edot[:, 0]
    edot = e_de_edot[:, 2]
    # plt.plot(e_ts, edot * t_nu)
    # plt.grid()
    # plt.show()

    fig, axs = plt.subplots(len(ecc), 1, figsize=[8, 12], sharex=True)
    for e, ax, dat in zip(ecc, axs, data):
        ax2 = ax.twinx()
        plot_edot_ldot_single(ax, ax2, e, dat[0], dat[1], dat[2], dat[3], edot[e_ts==e])

        if e==0.025:
            ax2.legend(loc="upper right", fontsize=11)

    config_axes(axs, key, ecc)
    axs[0].legend(loc="upper center", fontsize=11)

    plt.subplots_adjust(hspace=0.0)
    plt.savefig('d_dots_dr_test.pdf', dpi=800, pad_inches=0.1, bbox_inches='tight')
    plt.show()




if __name__ == '__main__':
    key = 'f'
    ecc, data = get_radial_series(key)
    plot_edot_ldot(ecc, data, key)
