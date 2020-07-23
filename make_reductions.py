from argparse import ArgumentParser
import os
import numpy as np
import h5py
from eccentricity_study import *




def energy(e, M=1.0, a=1.0, mu=0.25):
    return - mu * M / 2. / a




def ang_mom(e, M=1.0, a=1.0, mu=0.25): 
    return mu * np.sqrt(M * a * (1 - e * e))




def r_cross_F(xh1, xh2, loc, mass, rsoft):
    y1 = loc - xh1[:, None, None, None]
    y2 = loc - xh2[:, None, None, None]
    r1 = np.sqrt(y1[0] * y1[0] + y1[1] * y1[1])
    r2 = np.sqrt(y2[0] * y2[0] + y2[1] * y2[1])
    fg1 = (mass / pow(r1**2 + rsoft**2, 3. / 2.)) * y1
    fg2 = (mass / pow(r2**2 + rsoft**2, 3. / 2.)) * y2
    tg1 = np.cross(xh1, fg1, axis=0)
    tg2 = np.cross(xh2, fg2, axis=0)
    return tg1 + tg2




def F_dot_v(xh1, xh2, vh1, vh2, loc, mass, rsoft):
    y1 = loc - xh1[:, None, None, None]
    y2 = loc - xh2[:, None, None, None]
    r1 = np.sqrt(y1[0] * y1[0] + y1[1] * y1[1])
    r2 = np.sqrt(y2[0] * y2[0] + y2[1] * y2[1])
    fg1 = (mass / pow(r1**2 + rsoft**2, 3. / 2.)) * y1
    fg2 = (mass / pow(r2**2 + rsoft**2, 3. / 2.)) * y2
    wg1 = np.tensordot(vh1, fg1, axes=1)
    wg2 = np.tensordot(vh2, fg2, axes=1)
    return wg1 + wg2




def get_radial_distributions(fname, nbins):
    X   = get_dataset(fname, 'x')
    Y   = get_dataset(fname, 'y')
    sig = get_dataset(fname, 'sigma')
    dA  = get_dataset(fname, 'cell_area')
    dM  = get_dataset(fname, 'cell_mass')

    rd = domain_radius(fname)
    e  = eccentricity_value(fname)
    E  = get_eccentric_anomaly(fname)
    rs = softening_radius(fname)
    xh1, xh2, vh1, vh2 = two_body_position_and_velocity(E, e=e)

    R = np.sqrt(X * X + Y * Y)
    T = r_cross_F(xh1, xh2, np.array([X, Y]), dM, rs)
    W = F_dot_v  (xh1, xh2, vh1, vh2, np.array([X, Y]), dM, rs)

    bins = np.linspace(0.0, rd, nbins + 1)
    A , _ = np.histogram(R, bins=bins, weights=dA)
    M , _ = np.histogram(R, bins=bins, weights=dM)
    dE, _ = np.histogram(R, bins=bins, weights=W)
    dL, _ = np.histogram(R, bins=bins, weights=T)

    if (A == 0.0).any():
        raise RuntimeError('Warning: there were empty radial bins. Re-run with a smaller bin count.')

    return dict(radial_bins=bins, sigma=M / A, work_on=dE, torque_on=dL)




def get_2d_maps(fname, nbins):
    X   = get_dataset(f, 'x')
    Y   = get_dataset(f, 'y')
    sig = get_dataset(f, 'sigma')
    dA  = get_dataset(f, 'cell_area')
    dM  = get_dataset(f, 'cell_mass')

    e  = eccentricity_value(f)
    E  = get_eccentric_anomaly(f)
    rs = softening_radius(f)
    xh1, xh2, vh1, vh2 = two_body_position_and_velocity(E, e=e)

    t = r_cross_F(xh1, xh2, np.array([X, Y]), dM, rs)           / ang_mom(e)
    p = F_dot_v  (xh1, xh2, vh1, vh2, np.array([X, Y]), dM, rs) / energy(e) / 2.

    nu = get_true_anomaly(f)
    Xr = X * np.cos(-nu) - Y * np.sin(-nu)
    Yr = X * np.sin(-nu) + Y * np.cos(-nu)

    rlim = 5
    bins = np.linspace(-rlim, rlim, nbins + 1)
    A, xb, yb = np.histogram2d(Xr.flatten(), Yr.flatten(), weights=dA.flatten(), bins=[bins, bins])
    M, xb, yb = np.histogram2d(Xr.flatten(), Yr.flatten(), weights=dM.flatten(), bins=[bins, bins])
    T, xb, yb = np.histogram2d(Xr.flatten(), Yr.flatten(), weights=t.flatten() , bins=[bins, bins])
    P, xb, yb = np.histogram2d(Xr.flatten(), Yr.flatten(), weights=p.flatten() , bins=[bins, bins])
    A[A==0.0] = 1.0

    return dict(remapped_sigma=M / A, remapped_Ldot=T / A, remapped_Edot=P / A)




def get_moments(fname, rin, rout):    
    phi = get_dataset(fname, 'phi')
    vr  = get_dataset(fname, 'radial_velocity')
    vp  = get_dataset(fname, 'phi_velocity')
    dA  = get_dataset(fname, 'cell_area')
    dM  = get_dataset(fname, 'cell_mass')

    sigma_moment = np.sum(     dM * np.exp(1.j * phi)) / np.sum(dA)
    vr_moment    = np.sum(vr * dM * np.exp(1.j * phi)) / np.sum(vp * dM)

    return dict(sigma_moment=sigma_moment, vr_moment=vr_moment)




if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("filenames", nargs='+')
    parser.add_argument("--bins-radial", type=int, default=250)
    parser.add_argument("--bins-2dmap"    , type=int, default=250)
    parser.add_argument("--moment-cut-in" , '-in' , type=float, default=1.5)
    parser.add_argument("--moment-cut-out", '-out', type=float, default=5.0)

    args = parser.parse_args()

    for f in args.filenames:
        print(f)

        print('\t radial dists')
        dsets = get_radial_distributions(f, args.bins_radial)

        print('\t moments')
        dsets.update(get_moments(f, args.moment_cut_in, args.moment_cut_out))

        print('\t 2D_maps')
        dsets.update(get_2d_maps(f, args.bins_2dmap))

        #----------------------------------------------------------------------
        output_filename = f.replace('diagnostics', 'reductions')
        h5f = h5py.File(output_filename, 'w')

        for key, value in dsets.items():
            h5f[key] = value






