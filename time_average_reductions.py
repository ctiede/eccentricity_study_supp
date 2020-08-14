#!/usr/bin/env python3

from argparse import ArgumentParser
import os
import h5py
import glob
import numpy as np
from eccentricity_study import *




def write_moments_to_group(group, moments):
    group['avg_modulus' ] = np.mean(np.absolute(moments))
    group['phase_series'] = np.angle(moments)




def write_strobed_to_group(group, data, dim):
    if dim == 1:
        group['periapse'] = np.mean(data[7::8])
        group['pi_4'    ] = np.mean(data[0::8])
        group['pi_2'    ] = np.mean(data[1::8])
        group['3pi_4'   ] = np.mean(data[2::8])
        group['apoapse' ] = np.mean(data[3::8])
        group['5pi_4'   ] = np.mean(data[4::8])
        group['3pi_2'   ] = np.mean(data[5::8])
        group['7pi_4'   ] = np.mean(data[6::8])
    elif dim == 2:
        group['periapse'] = np.mean(data[7::8], axis=0) 
        group['pi_4'    ] = np.mean(data[0::8], axis=0) #The first output from relevant data is at pi / 4
        group['pi_2'    ] = np.mean(data[1::8], axis=0)
        group['3pi_4'   ] = np.mean(data[2::8], axis=0)
        group['apoapse' ] = np.mean(data[3::8], axis=0)
        group['5pi_4'   ] = np.mean(data[4::8], axis=0)
        group['3pi_2'   ] = np.mean(data[5::8], axis=0)
        group['7pi_4'   ] = np.mean(data[6::8], axis=0)
    elif dim == 3:
        group['periapse'] = np.mean(data[:, :, 7::8], axis=2) 
        group['pi_4'    ] = np.mean(data[:, :, 0::8], axis=2)
        group['pi_2'    ] = np.mean(data[:, :, 1::8], axis=2)
        group['3pi_4'   ] = np.mean(data[:, :, 2::8], axis=2)
        group['apoapse' ] = np.mean(data[:, :, 3::8], axis=2)
        group['5pi_4'   ] = np.mean(data[:, :, 4::8], axis=2)
        group['3pi_2'   ] = np.mean(data[:, :, 5::8], axis=2)
        group['7pi_4'   ] = np.mean(data[:, :, 6::8], axis=2)
    else:
        print("Invalid (dim)")




if __name__ == '__main__':
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
        M   = h5f['mean_anomaly'     ][...]
        E   = h5f['eccentric_anomaly'][...]
        f   = h5f['true_anomaly'     ][...]
        em  = h5f['sigma_moment']     [...]
        ev  = h5f['vr_moment'   ]     [...]
        em2 = h5f['sigma_moment_m2']  [...]
        ev2 = h5f['vr_moment_m2'   ]  [...]

        # 2D arrays
        s = h5f['sigma'      ][...]
        p = h5f['work_on'    ][...]
        t = h5f['torque_on'  ][...]

        # 3D maps
        S = h5f['remapped_sigma'][...]
        T = h5f['remapped_Ldot' ][...]
        P = h5f['remapped_Edot' ][...]


        if e < 0.1:
            ecc = str(e).split('.')[-1]
        else:
            ecc = str(int(e * 1e3))
        output_fname = 'time_avg_reductions_e{}.h5'.format(ecc)

        print('   writing', output_fname)
        h5w = h5py.File(output_fname, 'w')
        h5w.create_dataset('eccentricity', data=e)
        h5w.create_dataset('radial_bins' , data=n)
        h52.create_dataset('bins_2d'     , data=m)
        write_moments_to_group(h5w.create_group('sigma_moment')   , em)
        write_moments_to_group(h5w.create_group('vr_moment')      , ev)
        write_moments_to_group(h5w.create_group('sigma_moment_m2'), em2)
        write_moments_to_group(h5w.create_group('vr_moment_m2')   , ev2)
        write_strobed_to_group(h5w.create_group('mean_anomaly')  , M , 1)
        write_strobed_to_group(h5w.create_group('true_anomaly')  , f , 1)
        write_strobed_to_group(h5w.create_group('sigma')         , s , 2)
        write_strobed_to_group(h5w.create_group('work_on')       , p , 2)
        write_strobed_to_group(h5w.create_group('torque_on')     , t , 2)
        write_strobed_to_group(h5w.create_group('remapped_sigma'), S , 3)
        write_strobed_to_group(h5w.create_group('remapped_Ldot') , T , 3)
        write_strobed_to_group(h5w.create_group('remapped_Edot') , P , 3)
        h5w.close()
