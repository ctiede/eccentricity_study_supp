
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


def write_avg_sigma(fname, e, xy, sig):
    h5f = h5py.File(fname, 'w')
    h5f.create_dataset('eccentricity', data=e)
    h5f.create_dataset('vertices'    , data=xy)
    h5f.create_dataset('sigma'       , data=sig)
    h5f.close()


if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument("filenames", nargs='+')
    parser.add_argument("--bins", '-n', type=int, default=200)

    args = parser.parse_args()
    bins = args.bins


    sig_avg = None
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
        nu = get_true_anomaly(f)
        Xr = X * np.cos(-nu) - Y * np.sin(-nu)
        Yr = X * np.sin(-nu) + Y * np.cos(-nu)

        M, xb, yb = np.histogram2d(Xr.flatten(), Yr.flatten(), weights=(dA * sig).flatten(), bins=bins)
        A, xb, yb = np.histogram2d(Xr.flatten(), Yr.flatten(), weights=(dA      ).flatten(), bins=bins)
        A[A==0.0] = 1.0

        #----------------------------------------------------------------------
        s = M / A       
        if n == 0:
            sig_avg = s
        else:
            sig_avg = (sig_avg * n + s) / (n + 1)


        #----------------------------------------------------------------------
        mod = 'f'
        w   = 1e3 
        if e < 0.1: 
            w = 1e4
        filename = 'time_avg_sigma_e{}_{}.h5'.format(int(e * w), mod)
        write_avg_sigma(filename, e, np.column_stack([xb, yb]), sig_avg)
        
        # np.save('resample_vertices.npy', np.column_stack([xb, yb]))
        # np.save(filename, sig_avg)




