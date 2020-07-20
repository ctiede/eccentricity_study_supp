
from argparse import ArgumentParser
import os
import pickle
import math
import numpy as np
import h5py
import scipy.optimize
import matplotlib as mpl
import matplotlib.pyplot as plt
from eccentricity_study import *



#------------------------------------------------------------------------------
def r_cross_F(xh1, xh2, loc, mass, rsoft):
    
    y1 = loc - xh1[:, None, None, None]
    y2 = loc - xh2[:, None, None, None]
    r1 = np.sqrt(y1[0] * y1[0] + y1[1] * y1[1])
    r2 = np.sqrt(y2[0] * y2[0] + y2[1] * y2[1])

    # fg1 = (mass / r1**3) * y1
    # fg2 = (mass / r2**3) * y2
    fg1 = (mass / pow(r1**2 + rsoft**2, 3. / 2.)) * y1
    fg2 = (mass / pow(r2**2 + rsoft**2, 3. / 2.)) * y2
    tg1 = np.cross(xh1, fg1, axis=0)
    tg2 = np.cross(xh2, fg2, axis=0)

    return (tg1 + tg2)


def F_dot_v(xh1, xh2, vh1, vh2, loc, mass, rsoft):
    
    y1 = loc - xh1[:, None, None, None]
    y2 = loc - xh2[:, None, None, None]
    r1 = np.sqrt(y1[0] * y1[0] + y1[1] * y1[1])
    r2 = np.sqrt(y2[0] * y2[0] + y2[1] * y2[1])

    # fg1 = (mass / r1**3) * y1
    # fg2 = (mass / r2**3) * y2
    fg1 = (mass / pow(r1**2 + rsoft**2, 3. / 2.)) * y1
    fg2 = (mass / pow(r2**2 + rsoft**2, 3. / 2.)) * y2
    wg1 = np.tensordot(vh1, fg1, axes=1)
    wg2 = np.tensordot(vh2, fg2, axes=1)

    return (wg1 + wg2)


def get_moments(q, dM, phi, m):
    return q * np.exp(1.j * m * phi) * dM


#------------------------------------------------------------------------------
def write_radial_timeseries(fname, r, s, dE, dL, ev, es):
    h5f = h5py.File(fname, 'w')
    h5f.create_dataset('radius'     , data=r)
    h5f.create_dataset('sigma'      , data=s)
    h5f.create_dataset('work_on'    , data=dE)
    h5f.create_dataset('torque_on'  , data=dL)
    h5f.create_dataset('vr_moment'  , data=ev)
    h5f.create_dataset('mass_moment', data=es)
    h5f.close()


if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument("filenames", nargs='+')
    parser.add_argument("--radius", '-r', type=float, default=25.0)
    parser.add_argument("--bins", '-n', type=int, default=200)

    args = parser.parse_args()
    rcut = args.radius
    bins = args.bins

    sigma  = []
    Edot   = []
    Ldot   = []
    e_sig  = []
    e_vr   = []

    for f in args.filenames:

        # if not math.isclose(get_mean_anomaly(f), (np.pi / 2.), abs_tol=1e-2):
        #     continue
        print(f)

        #----------------------------------------------------------------------
        X   = get_dataset(f, 'x')
        Y   = get_dataset(f, 'y')
        phi = get_dataset(f, 'phi')
        sig = get_dataset(f, 'sigma')
        dA  = get_dataset(f, 'cell_area')
        vr  = get_dataset(f, 'radial_velocity')
        vp  = get_dataset(f, 'phi_velocity')

        e  = eccentricity_value(f)
        E  = get_eccentric_anomaly(f)
        rs = softening_radius(f)
        print(rs)
        xh1, xh2, vh1, vh2 = two_body_position_and_velocity(E, e=e)

        R = np.sqrt(X * X + Y * Y)
        T = r_cross_F(xh1, xh2, np.array([X, Y]), sig * dA, rs)
        W = F_dot_v  (xh1, xh2, vh1, vh2, np.array([X, Y]), sig * dA, rs)

        Mtot = np.sum(sig * dA)
        dm_v = get_moments(vr / vp, sig * dA, phi, 1) / Mtot
        dm_s = get_moments(1.0    , sig * dA, phi, 1) / Mtot

        #----------------------------------------------------------------------
        mask = R < rcut
        N , r = np.histogram(R[mask].flat, bins=bins)
        A , r = np.histogram(R[mask].flat, bins=bins, weights=dA  [mask].flat)
        M , r = np.histogram(R[mask].flat, bins=bins, weights=sig [mask].flat)
        dE, r = np.histogram(R[mask].flat, bins=bins, weights=W   [mask].flat)
        dL, r = np.histogram(R[mask].flat, bins=bins, weights=T   [mask].flat)
        ev, r = np.histogram(R[mask].flat, bins=bins, weights=dm_v[mask].flat)
        es, r = np.histogram(R[mask].flat, bins=bins, weights=dm_s[mask].flat)
        
        A[A==0.0] = 1.0
        r_annulus = 0.5 * (r[1:] + r[:-1])  

        #----------------------------------------------------------------------
        sigma.append(M / A)
        Edot.append (dE)
        Ldot.append (dL)     
        e_vr.append (ev)
        e_sig.append(es)  

        #----------------------------------------------------------------------
        mod = 'f'
        w   = 1e3 
        if e < 0.1: 
            w = 1e4
        filename = 'radial_timeseries_e{}_{}.h5'.format(int(e * w), mod)
        write_radial_timeseries(filename, 
                                r_annulus,
                                np.row_stack(sigma),
                                np.row_stack(Edot), 
                                np.row_stack(Ldot),
                                np.row_stack(e_vr),
                                np.row_stack(e_sig))



