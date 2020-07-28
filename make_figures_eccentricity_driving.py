#!/usr/bin/env python3

import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt

column_width = 3.32388
text_width   = 6.97385




def configure_matplotlib(hardcopy=False):
    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)
    plt.rc('axes', labelsize=8)
    plt.rc('legend', fontsize=8)
    plt.rc('font', family='Times New Roman' if hardcopy else 'DejaVu Sans', size=8)
    plt.rc('text', usetex=hardcopy)




def edot_times_r3_from_unit_point_mass_at_position(eccentricity, true_anomaly, x, y, rs=0.0, return_positions=False):
    G, M, a, W, q, e = 1.0, 1.0, 1.0, 1.0, 1.0, eccentricity
    M1   = M * (1.0 - q / (1.0 + q))
    M2   = M - M1
    E    = -0.5 * G * M1 * M2 / a
    L    = M1 * M2 / M * (G * M * a * (1 - e**2))**0.5
    f    = true_anomaly
    g    = (1 - e**2)**0.5
    r    = (x**2 + y**2)**0.5
    sinE = (g * np.sin(f)) / (1 + e * np.cos(f))
    cosE = (e + np.cos(f)) / (1 + e * np.cos(f))
    x1   = -a * (e - cosE) * q / (1 + q)
    x2   = +a * (e - cosE) * 1 / (1 + q)
    y1   = +a * (g * sinE) * q / (1 + q)
    y2   = -a * (g * sinE) * 1 / (1 + q)
    vx1  = -W * a * sinE / (1 - e * cosE) * q / (1 + q)
    vx2  = +W * a * sinE / (1 - e * cosE) * 1 / (1 + q)
    vy1  = +W * a * cosE / (1 - e * cosE) * q / (1 + q) * g
    vy2  = -W * a * cosE / (1 - e * cosE) * 1 / (1 + q) * g
    r1   = ((x - x1)**2 + (y - y1)**2)**0.5
    r2   = ((x - x2)**2 + (y - y2)**2)**0.5
    fx1  = G * M1 * (x - x1) / (r1 + rs)**3
    fy1  = G * M1 * (y - y1) / (r1 + rs)**3
    fx2  = G * M2 * (x - x2) / (r2 + rs)**3
    fy2  = G * M2 * (y - y2) / (r2 + rs)**3
    T1   = x1 * fy1 - y1 * fx1
    T2   = x2 * fy2 - y2 * fx2
    P1   = vx1 * fx1 + vy1 * fy1
    P2   = vx2 * fx2 + vy2 * fy2
    edot_r3 = r**3 * g**2 / e * (-0.5 * (P1 + P2) / E - (T1 + T2) / L)
    if return_positions:
        return edot_r3, (x1, y1), (x2, y2)
    else:
        return edot_r3




def edot_times_r3_limit_in_small_e_and_large_r(f, p):
    return -0.25 * (-2 * np.sin(f) + 9 * np.sin(f - 2 * p) + 3 * np.sin(3 * f - 2 * p))




def edot_times_r3_second_order_in_e_at_large_r(f, p):
    return -0.25 * (1 * np.sin(2 * f) - 3 * np.sin(2 * f - 2 * p) - 3 * np.sin(4 * f - 2 * p) + 6 * np.sin(2 * p))




def edot_times_r3_orbit_averaged_corotating(delta=0.0, eccentricity=1e-3, radius=10, order='exact'):
    def edot_exact(f):
        x = radius * np.cos(f + delta)
        y = radius * np.sin(f + delta)
        return edot_times_r3_from_unit_point_mass_at_position(eccentricity, f, x, y)
    def edot_first_order(f):
        return edot_times_r3_limit_in_small_e_and_large_r(f, f + delta)
    def edot_second_order(f):
        return edot_times_r3_second_order_in_e_at_large_r(f, f + delta)
    def integrate(g):
        return g(np.linspace(0.0, 2 * np.pi, 10000)).mean()
    if order == 'exact':
        return integrate(edot_exact) / 2 / np.pi
    if order == 1:
        return integrate(edot_first_order) / 2 / np.pi
    if order == 2:
        return integrate(edot_second_order) / 2 / np.pi




def plot_edot_map_at_true_anomaly(ax, true_anomaly, eccentricity=1e-3, show_orbiting_masses=True, component_size=150, radius=2.5):
    R = radius
    xs = np.linspace(-R, R, 400)
    ys = np.linspace(-R, R, 400)
    X, Y = np.meshgrid(xs, ys)
    edot, (x1, y1), (x2, y2) = edot_times_r3_from_unit_point_mass_at_position(eccentricity, true_anomaly, X, Y, rs=0.01, return_positions=True)
    extent = [-R, R, -R, R]
    levels = np.linspace(-3.5, 3.5, 9)
    ax.set_aspect('equal')
    ax.contourf(edot, extent=extent, levels=levels, cmap='Spectral_r', zorder=0)
    ax.contour (edot, extent=extent, levels=levels, colors='gray', linewidths=0.5, zorder=1)
    ax.scatter([x1, x2], [y1, y2], c='k', s=component_size, zorder=2)

    if show_orbiting_masses:
        r = 2**(2/3) # 1.587, radius where gas orbits at 1/2 Omega_bin
        phi = true_anomaly * 0.5 - np.array([3/8, 7/8]) * 2 * np.pi
        ax.scatter(r * np.cos(phi), r * np.sin(phi), marker='+', c='purple', s=200)




def plot_edot_map_at_true_anomaly_closeup(ax, true_anomaly, eccentricity=1e-3, component_size=150, radius=0.5):
    R = radius
    edot, (x1, y1), (x2, y2) = edot_times_r3_from_unit_point_mass_at_position(eccentricity, true_anomaly, 0.0, 0.0, return_positions=True)
    # extent = [x1 - R, x1 + R, y1 - R, y1 + R]
    extent = [-R, R, -R, R]
    xs = np.linspace(extent[0], extent[1], 400)
    ys = np.linspace(extent[2], extent[3], 400)
    X, Y = np.meshgrid(xs, ys)
    edot = edot_times_r3_from_unit_point_mass_at_position(eccentricity, true_anomaly, X, Y, rs=0.01)

    levels = np.linspace(-3, 3, 47)
    edot[edot < levels[ 0]] = levels[ 0]
    edot[edot > levels[-1]] = levels[-1]

    ax.set_aspect('equal')
    ax.contourf(edot, extent=extent, levels=levels, cmap='Spectral_r', zorder=0)
    ax.contour (edot, extent=extent, levels=levels, colors='gray', linewidths=0.5, zorder=1)
    ax.scatter([x1, x2], [y1, y2], c='k', s=component_size, zorder=2)

    # ax.arrow(0, 0, 0.5, 0.5, head_width=0.05, head_length=0.1, fc='k', ec='k')

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])




def make_figure_edot_orbit_averaged_corotating():
    fig, ax1 = plt.subplots(figsize=[6.5, 6.5])

    show_single_delta_versus_e = False

    if show_single_delta_versus_e:
        es = np.logspace(-2.5, -0.01, 60)
        edots = np.array([edot_times_r3_orbit_averaged_corotating(delta=0.125 * 2 * np.pi, eccentricity=e, radius=10) for e in es])
        ax1.plot(es, edots / es, '-o', mfc='none')
        ax1.set_xscale('log')
    else:
        e = 0.01
        deltas = np.linspace(0.0, 2 * np.pi, 100)

        edots = np.array([edot_times_r3_orbit_averaged_corotating(delta=d, eccentricity=e, radius=1000, order=2) for d in deltas])
        ax1.plot(deltas / 2 / np.pi, edots, '-o', mfc='none')

        edots = np.array([edot_times_r3_orbit_averaged_corotating(delta=d, eccentricity=e, radius=1000, order='exact') for d in deltas])
        ax1.plot(deltas / 2 / np.pi, edots / e, '-o', mfc='none')

        ax1.set_ylim(-0.25, 0.25)
        ax1.set_ylabel(r'$\dot e (a/r)^3$')
        ax1.set_xlabel(r'$\delta / 2\pi$')
    return fig




def make_figure_exact_edot_maps_circular():
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=[column_width, column_width])
    fig.subplots_adjust(left=0.002, right=0.998, top=0.998, bottom=0.002, wspace=0.02, hspace=0.02)

    labels = [
        r'Periapse',
        r'$f = \pi / 4$',
        r'$f = \pi / 2$',
        r'$f = 3 \pi / 4$',
    ]
    for n, ax in enumerate(axes.flatten()):
        plot_edot_map_at_true_anomaly(ax, n * np.pi / 4, show_orbiting_masses=False, component_size=25)
        ax.text(-2.8, 2.8, labels[n], ha='left', va='top', bbox=dict(facecolor='white', alpha=0.66, boxstyle='round'))
        ax.set_xticks([])
        ax.set_yticks([])
    return fig




def make_figure_edot_maps_eccentric():
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=[column_width, column_width * 2])
    fig.subplots_adjust(left=0.002, right=0.998, top=0.998, bottom=0.002, wspace=0.02, hspace=0.02)

    labels = [
        r'Periapse',
        r'$f = \pi / 4$',
        r'$f = \pi / 2$',
        r'$f = 3 \pi / 4$',
        r'Apoapse',
        r'$f = 5 \pi / 4$',
        r'$f = 3 \pi / 2$',
        r'$f = 7 \pi / 4$',
    ]
    for n, ax in enumerate(axes.flatten()):
        plot_edot_map_at_true_anomaly(ax, n * np.pi / 4, eccentricity=0.15, show_orbiting_masses=False, component_size=25, radius=2.5)
        ax.text(-2.3, 2.3, labels[n], ha='left', va='top', bbox=dict(facecolor='white', alpha=0.66, boxstyle='round'))
        ax.set_xticks([])
        ax.set_yticks([])
    return fig




def make_figure_edot_maps_closeup():
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[text_width, text_width])
    n = 7
    plot_edot_map_at_true_anomaly_closeup(ax, n * np.pi / 4, eccentricity=0.6, component_size=25, radius=0.2)
    return fig




def make_figure_first_and_second_order_in_e_terms():
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=[6.5, 6.5])
    fs = np.linspace(0.0, 2 * np.pi, 1000)
    e = 0.05
    r = 160.0
    phi = 0.12 * 2 * np.pi
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    edots = [edot_times_r3_from_unit_point_mass_at_position(e, f, x, y) for f in fs]
    edots_linear = np.array([edot_times_r3_limit_in_small_e_and_large_r(f, phi) for f in fs])
    edots_quadra = np.array([edot_times_r3_second_order_in_e_at_large_r(f, phi) for f in fs])
    ax1.plot(fs, edots, label='exact')
    ax1.plot(fs, edots_linear, label='linear term')
    ax2.plot(fs, edots - edots_linear, label='residual after taking out linear term')
    ax2.plot(fs, edots_quadra * e, label='quadratic term')
    ax1.legend()
    ax2.legend()
    ax1.set_xlabel('Orbital phase')
    return fig




def make_movie(fig, plot_fn, param_list, output='output.mp4'):
    from matplotlib.animation import FFMpegWriter
    writer = FFMpegWriter(fps=60)
    with writer.saving(fig, output, dpi=200):
        for n, param in enumerate(param_list):
            print(f'frame {n} / {len(param_list)}')
            plot_fn(fig, param)
            writer.grab_frame()
            fig.clf()
    print(f'writing {output}')




def make_movie_edot_maps(filename, eccentricity=1e-3):
    def plot_fn(fig, true_anomaly):
        ax1 = fig.add_subplot(1, 1, 1)
        plot_edot_map_at_true_anomaly(ax1, true_anomaly, eccentricity=eccentricity)
    fig = plt.figure(figsize=[10, 10])
    fs = np.linspace(0.0, 4 * np.pi, 960)
    make_movie(fig, plot_fn, fs, output=filename)




def make_movie_edot_maps_closeup(filename, eccentricity=1e-3):
    def plot_fn(fig, true_anomaly):
        ax1 = fig.add_subplot(1, 1, 1)
        plot_edot_map_at_true_anomaly_closeup(ax1, true_anomaly, eccentricity=eccentricity, component_size=25, radius=0.75)
    fig = plt.figure(figsize=[10, 10])
    fs = np.linspace(0.0, 2 * np.pi, 480)
    make_movie(fig, plot_fn, fs, output=filename)




# make_movie_edot_maps('edot_small_e.mp4', eccentricity=1e-3)
# make_movie_edot_maps('edot_e_50.mp4', eccentricity=0.5)
# make_movie_edot_maps_closeup('edot_closeup_e_50.mp4', eccentricity=0.5)




if __name__ == "__main__":
    from argparse import ArgumentParser, RawTextHelpFormatter
    figure_funcs = dict([(k[12:], v)  for k, v in locals().items() if k.startswith('make_figure')])
    parser = ArgumentParser(formatter_class=RawTextHelpFormatter, epilog='\n'.join(['all'] + list(figure_funcs.keys())))
    parser.add_argument('figures', nargs='+', metavar='figure', choices=['all'] + list(figure_funcs.keys()))
    parser.add_argument('--two-column', action='store_true')
    parser.add_argument('--hardcopy', action='store_true')
    parser.add_argument('--open', action='store_true')
    args = parser.parse_args()
    configure_matplotlib(args.hardcopy)
    figure_list = figure_funcs.keys() if args.figures == ['all'] else args.figures
    for figure in figure_list:
        fig = figure_funcs[figure]()
        if args.hardcopy:
            pdf_name = f'figures/{figure}.pdf'
            print(f'saving {pdf_name}')
            fig.savefig(f'{pdf_name}')
            if args.open:
                os.system(f'open {pdf_name}')
    if not args.hardcopy:
        plt.show()
