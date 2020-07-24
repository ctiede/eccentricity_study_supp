#!/usr/bin/env python3

import argparse
import numpy as np
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import figures as figs




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
rbin = 10. / 1000




def E(e, M=1.0, a=1.0, mu=0.25):
    return mu * M / 2. / a




def L(e, M=1.0, a=1.0, mu=0.25): 
    return mu * np.sqrt(M * a * (1 - e * e))




def moving_average(a, window_size=10):
    n = window_size
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n




def plot_moving_average_mod(ax, x, y, window_size=50, avg_only=True, c=None, **kwargs):
    if not avg_only:
        ax.plot(x, y, c=c, lw=1.0, alpha=0.5)
    return ax.plot(moving_average(x, window_size), moving_average(y, window_size) * window_size, c=c, **kwargs)




def plot_cumsum(ax, x, y, c=None, **kwargs):
    ax.fill_between(x, np.cumsum(y), facecolor=c, alpha=0.2)
    return ax.plot(x, np.cumsum(y), c=c, **kwargs)




def plot_edot_ldot_single(ax, M, ecc, r, s, dE, dL):
    mask = r < 8    
    print(dE.shape)

    Edot = np.mean(dE, axis=0)[mask] / E(ecc) / 2. * t_nu / rbin
    Ldot = np.mean(dL, axis=0)[mask] / L(ecc)      * t_nu / rbin
    lab1 = 'Relative power, ' + r'$-(F_i \cdot v_i) \cdot t_\nu / 2E$'
    lab2 = 'Relative torque, ' + r'$\,(r_i \times F_i)_z \cdot t_\nu / L$'

    # plot_moving_average_mod(ax, r[mask], Edot, lw=2, color=red , label=lab1)
    # plot_moving_average_mod(ax, r[mask], Ldot, lw=2, color=blue, label=lab2)
    ax.plot(r[mask], Edot, lw=2, color=red , label=lab1)
    ax.plot(r[mask], Ldot, lw=2, color=blue, label=lab2)
    plot_cumsum(ax, r[mask], Edot - Ldot, ls='--', lw=0.75, color='violet', label='Cumulative difference, ' + r'$\dot e$')




def config_axes(axs, key, ecc):
    for ax, e in zip(axs, ecc):        
        ax.axhline(0.0, ls=':', color='grey', lw=0.75, alpha=0.7)
        ax.text(1.01, 0.5, r'e = ' + str(e), transform=ax.transAxes, rotation=90)
        ax.set_xlim([ 0.0 , 5.0])
        ax.set_ylim([-0.25, 0.25])
        ax.set_yticks([-0.15, 0.0, 0.15])

    if key == 'u':
        axs[0].text(0.02, 1.02, r'Mean anomaly = $\pi$ / 2', transform=axs[0].transAxes)
    if key == 'd':
        axs[0].text(0.02, 1.02, r'Mean anomaly = $3\pi$ / 2', transform=axs[0].transAxes)
    if key == 'f':
        axs[0].text(0.02, 1.02, r'200 Orbit Average (1200-1400)', transform=axs[0].transAxes)
    axs[-1].set_xlabel(r'$r \, / \, a$', fontsize=14)




def load_radial_timeseries(fname):
    h5f = h5py.File(fname, 'r')
    r   = h5f['radius']     [...]
    s   = h5f['sigma']      [...]
    W   = h5f['work_on']    [...]
    T   = h5f['torque_on']  [...]
    ev  = h5f['vr_moment']  [...]
    es  = h5f['mass_moment'][...]
    return r, s, W, T, ev, es




def get_radial_series(key):
    if key == 'u':
        M    = np.pi / 2.
        ecc  = [0.025, 0.100, 0.750]
        e025 = load_radial_timeseries('radial_timeseries_e025_u.h5')
        e100 = load_radial_timeseries('radial_timeseries_e100_u.h5')
        e750 = load_radial_timeseries('radial_timeseries_e750_u.h5')
        data = [e025, e100, e750]
    if key == 'd':
        M    = 3 * np.pi / 2.
        ecc  = [0.025, 0.100, 0.750]
        e025 = load_radial_timeseries('radial_timeseries_e025_d.h5')
        e100 = load_radial_timeseries('radial_timeseries_e100_d.h5')
        e750 = load_radial_timeseries('radial_timeseries_e750_d.h5')
        data = [e025, e100, e750]
    if key == 'f':
        M    = None
        ecc  = [0.025, 0.1, 0.3, 0.4, 0.5, 0.6, 0.75]
        e025 = load_radial_timeseries('./e025_full/radial_timeseries_e250_f.h5')
        e100 = load_radial_timeseries('./e100_full/radial_timeseries_e100_f.h5')
        e300 = load_radial_timeseries('./e300_full/radial_timeseries_e300_f.h5')
        e400 = load_radial_timeseries('./e400_full/radial_timeseries_e400_f.h5')
        e500 = load_radial_timeseries('./e500_full/radial_timeseries_e500_f.h5')
        e600 = load_radial_timeseries('./e600_full/radial_timeseries_e600_f.h5')
        e750 = load_radial_timeseries('./e750_full/radial_timeseries_e750_f.h5')
        data = [e025, e100, e300, e400, e500, e600, e750]
    return M, ecc, data




def plot_edot_ldot(M, ecc, data, key):
    fig, axs = plt.subplots(len(ecc), 1, figsize=[8, 12], sharex=True)

    for e, ax, dat in zip(ecc, axs, data):
        plot_edot_ldot_single(ax, M, e, dat[0], dat[1], dat[2], dat[3])

    config_axes(axs, key, ecc)
    axs[0].legend(loc="upper right", fontsize=12)

    plt.subplots_adjust(hspace=0.0)
    plt.savefig('Edot_Ldot_edot.pdf', dpi=800, pad_inches=0.1, bbox_inches='tight')
    plt.show()




if __name__ == '__main__':
    key = 'f'
    M, ecc, data = get_radial_series(key)
    plot_edot_ldot(M, ecc, data, key)
