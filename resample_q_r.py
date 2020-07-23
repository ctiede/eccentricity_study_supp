from argparse import ArgumentParser
import os
import numpy as np
import h5py
from eccentricity_study import *




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




def get_moments(q, dM, phi, m=1):
    return q * np.exp(1.j * m * phi) * dM




if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("filenames", nargs='+')
    parser.add_argument("--bins", '-n', type=int, default=200)

    args = parser.parse_args()

    for f in args.filenames:
        print(f)

        #----------------------------------------------------------------------
        X   = get_dataset(f, 'x')
        Y   = get_dataset(f, 'y')
        phi = get_dataset(f, 'phi')
        sig = get_dataset(f, 'sigma')
        dA  = get_dataset(f, 'cell_area')
        dM  = get_dataset(f, 'cell_mass')
        vr  = get_dataset(f, 'radial_velocity')
        vp  = get_dataset(f, 'phi_velocity')

        rd = domain_radius(f)
        e  = eccentricity_value(f)
        E  = get_eccentric_anomaly(f)
        rs = softening_radius(f)
        xh1, xh2, vh1, vh2 = two_body_position_and_velocity(E, e=e)

        R = np.sqrt(X * X + Y * Y)
        T = r_cross_F(xh1, xh2, np.array([X, Y]), dM, rs)
        W = F_dot_v  (xh1, xh2, vh1, vh2, np.array([X, Y]), dM, rs)

        Mtot = np.sum(dM)
        dm_v = get_moments(vr / vp, dM, phi, m=1) / Mtot
        dm_s = get_moments(1.0    , dM, phi, m=1) / Mtot

        #----------------------------------------------------------------------
        bins = np.linspace(0.0, rd, args.bins)
        A , _ = np.histogram(R, bins=bins, weights=dA)
        M , _ = np.histogram(R, bins=bins, weights=dM)
        dE, _ = np.histogram(R, bins=bins, weights=W)
        dL, _ = np.histogram(R, bins=bins, weights=T)
        # ev, _ = np.histogram(R, bins=bins, weights=dm_v)
        # es, _ = np.histogram(R, bins=bins, weights=dm_s)

        if (A == 0.0).any():
            print('Warning: there were empty radial bins. Re-run with a smaller bin count. Exiting!')
            exit()

        #----------------------------------------------------------------------
        output_filename = f.replace('diagnostics', 'radial_profiles')
        h5f = h5py.File(output_filename, 'w')
        h5f['radial_bins'] = bins
        h5f['sigma'      ] = M / A
        h5f['work_on'    ] = dE
        h5f['torque_on'  ] = dL
        # h5f['vr_moment'  ] = ev
        # h5f['mass_moment'] = es
