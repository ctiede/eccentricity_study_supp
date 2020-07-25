#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt




def edot_times_r3_from_unit_point_mass_at_position(eccentricity, true_anomaly, x, y, rs=0.0):
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
    return r**3 * g**2  / e * (-0.5 * (P1 + P2) / E - (T1 + T2) / L)




def edot_times_r3_limit_in_small_e_and_large_r(f, phi):
    return 0.25 * (2 * np.sin(f) - 9 * np.sin(f - 2 * phi) - 3 * np.sin(3 * f - 2 * phi))




def plot_edot_map_at_true_anomaly(ax, true_anomaly):
    xs = np.linspace(-3, 3, 400)
    ys = np.linspace(-3, 3, 400)
    X, Y = np.meshgrid(xs, ys)
    levels = np.linspace(-3, 3, 13)
    edot = edot_times_r3_from_unit_point_mass_at_position(0.001, true_anomaly, X, Y, rs=0.1)
    print(edot.max(), edot.min())
    ax.set_aspect('equal')
    ax.contourf(edot, extent=[-3, 3, -3, 3], levels=levels, cmap='Spectral')
    ax.contour(edot, extent=[-3, 3, -3, 3], levels=levels, colors='k', linewidths=0.5)




def make_figure_single_edot_map():
    fig = plt.figure(figsize=[10, 10])
    ax1 = fig.add_subplot(1, 1, 1)
    plot_edot_map_at_true_anomaly(ax1, 5 * np.pi / 8)
    return fig




def make_figure_many_edot_map():
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=[6.5, 6.5])

    fig.subplots_adjust(left=0.002, right=0.998, top=0.998, bottom=0.002, wspace=0.02, hspace=0.02)
    for n, ax in enumerate(axes.flatten()):
        plot_edot_map_at_true_anomaly(ax, n * np.pi / 4)
        ax.set_xticks([])
        ax.set_yticks([])

    return fig




fig = make_figure_many_edot_map()
plt.show()




# angles = np.linspace(0.0, 2 * np.pi, 1000)
# fs = angles

# r = 5.0
# phi = 0.25
# x = r * np.cos(phi)
# y = r * np.sin(phi)
# edots = [edot_times_r3_from_unit_point_mass_at_position(0.001, f, x, y) for f in fs]
# edots_linear = [edot_times_r3_limit_in_small_e_and_large_r(f, phi) for f in fs]
# ax1.plot(fs, edots)
# ax1.plot(fs, edots_linear)
