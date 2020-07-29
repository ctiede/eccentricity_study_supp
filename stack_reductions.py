#!/usr/bin/env python3

from argparse import ArgumentParser
import os
import h5py
import glob
import numpy as np
from eccentricity_study import *


parser = ArgumentParser()
parser.add_argument("directory", default='./')
args  = parser.parse_args()
files = sorted(glob.glob(args.directory + 'reductions.*.h5', recursive=False))

mean_anomaly      = []
eccentric_anomaly = []
true_anomaly      = []
sigma             = []
work_on           = []
torque_on         = []
remapped_sigma    = []
remapped_Ldot     = []
remapped_Edot     = []
sigma_moment      = []
vr_moment         = []

for fname in files:
    try:
        h5f = h5py.File(fname, 'r')
    except:
        print("\t Failed load:", fname)
        continue
    print(fname)

    e = h5f['eccentricity'     ][...]
    M = h5f['mean_anomaly'     ][...]
    E = h5f['eccentric_anomaly'][...]
    f = h5f['true_anomaly'     ][...]

    n = h5f['radial_bins'][...]
    s = h5f['sigma'      ][...]
    p = h5f['work_on'    ][...]
    t = h5f['torque_on'  ][...]

    m = h5f['bins_2d'][...]
    S = h5f['remapped_sigma'][...]
    T = h5f['remapped_Ldot' ][...]
    P = h5f['remapped_Edot' ][...]

    em = h5f['sigma_moment'][...]
    ev = h5f['vr_moment'   ][...]

    mean_anomaly.append     (M)
    eccentric_anomaly.append(E)
    true_anomaly.append     (f)
    sigma.append            (s)
    work_on.append          (p)
    torque_on.append        (t)
    remapped_sigma.append   (S)
    remapped_Ldot.append    (T)
    remapped_Edot.append    (P)
    sigma_moment.append     (em)
    vr_moment.append        (ev)

if e < 0.1:
    ecc = str(e).split('.')[-1]
else:
    ecc = str(int(e * 1e3))
output_fname = 'stacked_reductions_e{}.h5'.format(ecc)

print('   Writing output')
h5w = h5py.File(output_fname, 'w')
h5w['eccentricity']      = e
h5w['radial_bins']       = n
h5w['mean_anomaly']      = np.array(mean_anomaly) 
h5w['eccentric_anomaly'] = np.array(eccentric_anomaly)
h5w['true_anomaly']      = np.array(true_anomaly)
h5w['sigma_moment']      = np.array(sigma_moment)
h5w['vr_moment']         = np.array(vr_moment)
h5w['sigma']             = np.row_stack(sigma)
h5w['work_on']           = np.row_stack(work_on)
h5w['torque_on']         = np.row_stack(torque_on)
h5w['remapped_sigma']    = np.dstack(remapped_sigma)
h5w['remapped_Ldot']     = np.dstack(remapped_Ldot)
h5w['remapped_Edot']     = np.dstack(remapped_Edot)
