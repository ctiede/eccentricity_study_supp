
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
def energy(e, M=1.0, a=1.0, mu=0.25):
    return mu * M / 2. / a


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

    return (tg1 + tg2)


def F_dot_v(xh1, xh2, vh1, vh2, loc, mass, rsoft):
    
    y1 = loc - xh1[:, None, None, None]
    y2 = loc - xh2[:, None, None, None]
    r1 = np.sqrt(y1[0] * y1[0] + y1[1] * y1[1])
    r2 = np.sqrt(y2[0] * y2[0] + y2[1] * y2[1])

    fg1 = (mass / pow(r1**2 + rsoft**2, 3. / 2.)) * y1
    fg2 = (mass / pow(r2**2 + rsoft**2, 3. / 2.)) * y2
    wg1 = np.tensordot(vh1, fg1, axes=1)
    wg2 = np.tensordot(vh2, fg2, axes=1)

    return (wg1 + wg2)


#------------------------------------------------------------------------------
def write_maps(fname, e, xy, sig, edot):
    h5f = h5py.File(fname, 'w')
    h5f.create_dataset('eccentricity', data=e)
    h5f.create_dataset('vertices'    , data=xy)
    h5f.create_dataset('sigma'       , data=sig)
    h5f.create_dataset('edot'        , data=edot)
    h5f.close()


if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument("filenames", nargs='+')
    parser.add_argument("--bins", '-n', type=int, default=200)

    args = parser.parse_args()
    bins = args.bins


    sig_avg = None
    de_avg  = None
    for n, f in enumerate(args.filenames):

        # if not math.isclose(get_mean_anomaly(f), (np.pi / 2.), abs_tol=1e-2):
        #     continue
        print(f)

        #----------------------------------------------------------------------
        X   = get_dataset(f, 'x')
        Y   = get_dataset(f, 'y')
        sig = get_dataset(f, 'sigma')
        dA  = get_dataset(f, 'cell_area')

        #----------------------------------------------------------------------
        e  = eccentricity_value(f)
        E  = get_eccentric_anomaly(f)
        rs = softening_radius(f)
        xh1, xh2, vh1, vh2 = two_body_position_and_velocity(E, e=e)

        #----------------------------------------------------------------------
        torq = r_cross_F(xh1, xh2, np.array([X, Y]), sig * dA, rs)           / ang_mom(e)
        work = F_dot_v  (xh1, xh2, vh1, vh2, np.array([X, Y]), sig * dA, rs) / energy(e) / 2.
        edot = work - torq

        #----------------------------------------------------------------------
        nu = get_true_anomaly(f)
        Xr = X * np.cos(-nu) - Y * np.sin(-nu)
        Yr = X * np.sin(-nu) + Y * np.cos(-nu)

        A, xb, yb = np.histogram2d(Xr.flatten(), Yr.flatten(), weights=       (dA).flatten(), bins=bins)
        M, xb, yb = np.histogram2d(Xr.flatten(), Yr.flatten(), weights= (dA * sig).flatten(), bins=bins)
        Q, xb, yb = np.histogram2d(Xr.flatten(), Yr.flatten(), weights=(dA * edot).flatten(), bins=bins)
        A[A==0.0] = 1.0

        #----------------------------------------------------------------------
        s  = M / A 
        de = Q / A
        if n == 0:
            sig_avg = s
            de_avg  = de
        else:
            sig_avg = (sig_avg * n + s ) / (n + 1)
            de_avg  = (de_avg  * n + de) / (n + 1)


        #----------------------------------------------------------------------
        mod = 'f'
        if e < 0.1:
            filename = 'time_maps_e{}_{}.h5'.format(str(e).split('.')[1], mod)
        else:
            filename = 'time_maps_e{}_{}.h5'.format(int(e * 1e3), mod)
        if n%10 is 0:
            write_maps(filename, e, np.column_stack([xb, yb]), sig_avg, de_avg)
        
    #--------------------------------------------------------------------------
    write_maps(filename, e, np.column_stack([xb, yb]), sig_avg, de_avg)




