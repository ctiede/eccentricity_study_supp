
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


class Signal:
    def __init__(self, fname, saturation_orbit=450, completion_orbit=800):
        h5f = h5py.File(fname, 'r')
        self.e = h5f['run_config']['eccentricity'][()]
        self.orbits = h5f['time_series']['time'][:] / 2 / np.pi
        self.a_acc  = h5f['time_series']['orbital_elements_acc' ]['elements']['separation'][:]
        self.e_acc  = h5f['time_series']['orbital_elements_acc' ]['elements']['eccentricity'][:]
        self.M_acc  = h5f['time_series']['orbital_elements_acc' ]['elements']['total_mass'][:]
        self.a_grv  = h5f['time_series']['orbital_elements_grav']['elements']['separation'][:]
        self.e_grv  = h5f['time_series']['orbital_elements_grav']['elements']['eccentricity'][:]
        self.M_grv  = h5f['time_series']['orbital_elements_grav']['elements']['total_mass'][:]
        self.L_grv  = h5f['time_series']['integrated_torque_on'][:,0] + h5f['time_series']['integrated_torque_on'][:,1]
        self.p_acc  = h5f['time_series']['orbital_elements_acc' ]['pomega'][:]
        self.p_grv  = h5f['time_series']['orbital_elements_grav']['pomega'][:]
        isat = np.where(self.orbits > saturation_orbit)[0][0]
        ifin = np.where(self.orbits > completion_orbit)[0][0]
        dm = self.M_acc[ifin] - self.M_acc[isat]
        self.isat = isat
        self.ifin = ifin
        self.dm = dm
        self.dt = (self.orbits[ifin] - self.orbits[isat]) * 2 * np.pi
        self.da_acc = (self.a_acc[ifin] - self.a_acc[isat]) / dm
        self.da_grv = (self.a_grv[ifin] - self.a_grv[isat]) / dm
        self.de_acc = (self.e_acc[ifin] - self.e_acc[isat]) / dm
        self.de_grv = (self.e_grv[ifin] - self.e_grv[isat]) / dm
        self.dp_acc = (self.p_acc[ifin] - self.p_acc[isat]) / dm
        self.dp_grv = (self.p_grv[ifin] - self.p_grv[isat]) / dm
        self.dl_grv = (self.L_grv[ifin] - self.L_grv[isat]) / dm
        self.fname = fname
    @property
    def time(self):
        return self.orbits * 2 * np.pi
    @property
    def e_tot(self):
        return self.e_acc + self.e_grv
    @property
    def a_tot(self):
        return self.a_acc + self.a_grv
    @property
    def da_acc_iso(self):
        return figures.predicted_isotropic_da_dm(self.fname, saturation_orbit=450, final_orbit=800)
    @property
    def disk_mass(self):
        return h5py.File(self.fname, 'r')['time_series']['disk_mass']
    @property
    def pomega_grv(self):
        return h5py.File(self.fname, 'r')['time_series']['orbital_elements_grav']['pomega'][:]
    @property
    def pomega_acc(self):
        return h5py.File(self.fname, 'r')['time_series']['orbital_elements_acc' ]['pomega'][:]



if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument("filenames", nargs='+')
    args = parser.parse_args()
    
    ecc  = []
    decc = []
    Mdot = []
    for f in args.filenames:
        print(f)
        s  = Signal(f, saturation_orbit=1200, completion_orbit=1400)
        e  = s.e
        de = s.de_acc + s.de_grv
        dM = s.dm
        dt = s.dt


        ecc.append ( e)
        decc.append(de)
        Mdot.append(dM / dt)
        np.save('e_de_dM.npy', np.column_stack([ecc, decc, Mdot]))



        







