#!/usr/bin/env python3

from argparse import ArgumentParser
import os
import numpy as np
import h5py
import glob
from eccentricity_study import *




def get_moments(fname, rin, rout, corot=False): 
    R   = get_dataset(fname, 'radius')   
    cut = np.where((R > rin)&(R < rout))

    phi = get_dataset(fname, 'phi')[cut]
    vr  = get_dataset(fname, 'radial_velocity')[cut]
    vp  = get_dataset(fname, 'phi_velocity')[cut]
    dA  = get_dataset(fname, 'cell_area')[cut]
    dM  = get_dataset(fname, 'cell_mass')[cut]

    theta = phi
    if corot is True:
        nu    = get_true_anomaly(fname)
        theta = phi - nu

    sigma_moment    = np.sum(     dM * np.exp(1.j * theta)) / np.sum(dA)
    sigma_moment_m2 = np.sum(     dM * np.exp(2.j * theta)) / np.sum(dA)
    vr_moment       = np.sum(vr * dM * np.exp(1.j * theta)) / np.sum(vp * dM)
    vr_moment_m2    = np.sum(vr * dM * np.exp(2.j * theta)) / np.sum(vp * dM)

    return dict(sigma_moment=sigma_moment, vr_moment=vr_moment, sigma_moment_m2=sigma_moment_m2, vr_moment_m2=vr_moment_m2)




def meta_data(fname):
    e = eccentricity_value   (fname)
    M = get_mean_anomaly     (fname)
    E = get_eccentric_anomaly(fname)
    f = get_true_anomaly     (fname)
    return dict(eccentricity=e, mean_anomaly=M, eccentric_anomaly=E, true_anomaly=f)




def make_moment_files(args):
    print('Making moment files')
    for fname in args.filenames:
        print(fname)
        corot = dict()
        inert = dict()

        corot.update(get_moments(fname, args.moment_cut_in, args.moment_cut_out, corot=True))
        inert.update(get_moments(fname, args.moment_cut_in, args.moment_cut_out, corot=False))

        if len(fname.split('/')) == 1:
            output_filename = fname.replace('diagnostics', '__moments__')
        else:
            output_filename = (fname.split('/')[-1]).replace('diagnostics', '__moments__')

        print(f'     writing', output_filename)
        h5f = h5py.File(output_filename, 'w')
        gco = h5f.create_group('corot')
        gin = h5f.create_group('inert')
        
        for key, value in corot.items():
            gco[key] = value
        for key, value in inert.items():
            gin[key] = value

        extras = meta_data(fname)
        for key, value in extras.items():
            h5f[key] = value




def stack_moment_files():
    print('Stacking moment files')
    files = sorted(glob.glob('./' + '__moments__.*.h5', recursive=False))

    mean_anomaly      = []
    eccentric_anomaly = []
    true_anomaly      = []
    s_moment_corot    = []
    v_moment_corot    = []
    s_moment_inert    = []
    v_moment_inert    = []
    s_moment_m2_corot = []
    v_moment_m2_corot = []
    s_moment_m2_inert = []
    v_moment_m2_inert = []

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

        em_c  = h5f['corot']['sigma_moment']   [...]
        ev_c  = h5f['corot']['vr_moment'   ]   [...]
        em2_c = h5f['corot']['sigma_moment_m2'][...]
        ev2_c = h5f['corot']['vr_moment_m2']   [...]

        em_i  = h5f['inert']['sigma_moment']   [...]
        ev_i  = h5f['inert']['vr_moment'   ]   [...]
        em2_i = h5f['inert']['sigma_moment_m2'][...]
        ev2_i = h5f['inert']['vr_moment_m2']   [...]

        s_moment_corot.append(em_c)
        v_moment_corot.append(ev_c)
        s_moment_inert.append(em_i)
        v_moment_inert.append(ev_i)
        s_moment_m2_corot.append(em2_c)
        v_moment_m2_corot.append(ev2_c)
        s_moment_m2_inert.append(em2_i)
        v_moment_m2_inert.append(ev2_i)
        os.remove(fname)

    if e < 0.1:
        ecc = str(e).split('.')[-1]
    else:
        ecc = str(int(e * 1e3))
    output_fname = 'moments_e{}.h5'.format(ecc)

    print('   Writing output')
    h5w = h5py.File(output_fname, 'w')
    gco = h5w.create_group('corot')
    gin = h5w.create_group('inert')
    h5w['eccentricity']      = e
    h5w['mean_anomaly']      = np.array(mean_anomaly) 
    h5w['eccentric_anomaly'] = np.array(eccentric_anomaly)
    h5w['true_anomaly']      = np.array(true_anomaly)
    gco['sigma_moment']      = np.array(s_moment_corot)
    gco['vr_moment']         = np.array(v_moment_corot)
    gco['sigma_moment_m2']   = np.array(s_moment_m2_corot)
    gco['vr_moment_m2']      = np.array(v_moment_m2_corot)
    gin['sigma_moment']      = np.array(s_moment_inert)
    gin['vr_moment']         = np.array(v_moment_inert)
    gin['sigma_moment_m2']   = np.array(s_moment_m2_inert)
    gin['vr_moment_m2']      = np.array(v_moment_m2_inert)




if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("filenames", nargs='+')
    parser.add_argument("--moment-cut-in" , type=float, default=1.0)
    parser.add_argument("--moment-cut-out", type=float, default=2.5)
    args = parser.parse_args()

    make_moment_files (args)
    stack_moment_files()
