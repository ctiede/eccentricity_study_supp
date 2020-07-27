#!/usr/bin/env python3

from argparse import ArgumentParser
import os
import h5py
import glob
import numpy as np
from eccentricity_study import *




parser = ArgumentParser()
parser.add_argument("filenames", nargs='+', help="set of stacked_reduction.h5 files to time average")
args = parser.parse_args()

for f in args.filenames:
    print(f)
    h5f = h5py.File(f, 'r') 

    # Scalar
    e = h5f['eccentricity'][...]
    n = h5f['radial_bins' ][...]
    # m = h5f['bins_2d'][...]
    
    # 1D arrays
    M = h5f['mean_anomaly'     ][...]
    E = h5f['eccentric_anomaly'][...]
    f = h5f['true_anomaly'     ][...]
    em = h5f['sigma_moment'][...]
    ev = h5f['vr_moment'   ][...]

    # 2D arrays
    s = h5f['sigma'      ][...]
    p = h5f['work_on'    ][...]
    t = h5f['torque_on'  ][...]

    # 3D maps
    S = h5f['remapped_sigma'][...]
    T = h5f['remapped_Ldot' ][...]
    P = h5f['remapped_Edot' ][...]

    avg_sigma_moment = np.mean(em)
    avg_vr_moment    = np.mean(ev)

    avg_sigma_profile = np.mean(s, axis=0)
    avg_work_profile  = np.mean(p, axis=0)
    avg_torq_profile  = np.mean(t, axis=0)

    avg_sigma_map = np.mean(S, axis=2)
    avg_Ldot_map  = np.mean(T, axis=2)
    avg_Edot_map  = np.mean(P, axis=2)

    if e < 0.1:
        ecc = str(e).split('.')[-1]
    else:
        ecc = str(int(e * 1e3))
    output_fname = 'time_avg_reductions_{}.h5'.format(ecc)

    print('   writing', output_fname)
    h5w = h5py.File(output_fname, 'w')
    h5w['eccentricity']   = e
    h5w['radial_bins']    = n
    h5w['sigma_moment']   = avg_sigma_moment
    h5w['vr_moment']      = avg_vr_moment
    h5w['sigma']          = avg_sigma_profile
    h5w['work_on']        = avg_work_profile
    h5w['torque_on']      = avg_torq_profile
    h5w['remapped_sigma'] = avg_sigma_map
    h5w['remapped_Ldot']  = avg_Ldot_map
    h5w['remapped_Edot']  = avg_Edot_map
    h5w.close()



