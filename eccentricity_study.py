import os
import pickle
import math
import numpy as np
import h5py
import scipy.optimize
import matplotlib as mpl
import matplotlib.pyplot as plt




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





class HelperFunctions(object):
    """
    Contains functions to convert datasets in an HDF5 diagnostics file into more
    useful data.
    """
    def cell_center_x_array(self, vertices):
        x = vertices[:,:,0]
        xc = 0.25 * (x[:-1,:-1] + x[1:,:-1] + x[:-1,1:] + x[1:,1:])
        return xc

    def cell_center_y_array(self, vertices):
        y = vertices[:,:,1]
        yc = 0.25 * (y[:-1,:-1] + y[1:,:-1] + y[:-1,1:] + y[1:,1:])
        return yc

    def area_array(self, vertices):
        x = vertices[:,:,0]
        y = vertices[:,:,1]
        dx = np.diff(x, axis=0)
        dy = np.diff(y, axis=1)
        dx_m = 0.5 * (dx[:,:-1] + dx[:,1:])
        dy_m = 0.5 * (dy[:-1,:] + dy[1:,:])
        return dx_m * dy_m

    def radius_array(self, vertices):
        xc = self.cell_center_x_array(vertices)
        yc = self.cell_center_y_array(vertices)
        return (xc**2 + yc**2)**0.5

    def phi_array(self, vertices):
        xc = self.cell_center_x_array(vertices)
        yc = self.cell_center_y_array(vertices)
        return np.arctan2(yc, xc)




def get_dataset(fname, key):
    """
    Generates a dataset from an HDF5 diagnostics file, either by loading it
    directly, or performing a simple operation on the raw data.
    
    :type       fname:     str
    :param      fname:     The name of the HDF5 file to load (or generate) from
    :type       key:       str
    :param      key:       The name of the dataset to load (or generate). Can be
                           one of [sigma, radial_velocity, phi_velocity, radius,
                           phi, cell_area, mass].

    :returns:   The dataset
    :rtype:     a 3D numpy array with shape (num_blocks, block_size, block_size)
    
    :raises     KeyError:  If the key was not known
    """
    h5f = h5py.File(fname, 'r')
    helpers = HelperFunctions()

    if key == 'x':
        return np.array([helpers.cell_center_x_array(g[...]) for g in h5f['vertices'].values()])

    if key == 'y':
        return np.array([helpers.cell_center_y_array(g[...]) for g in h5f['vertices'].values()])

    if key in ['sigma', 'radial_velocity', 'phi_velocity']:
        return np.array([g[...] for g in h5f[key].values()])

    if key == 'radius':
        return np.array([helpers.radius_array(g[...]) for g in h5f['vertices'].values()])

    if key == 'phi':
        return np.array([helpers.phi_array(g[...]) for g in h5f['vertices'].values()])

    if key == 'cell_area':
        return np.array([helpers.area_array(g[...]) for g in h5f['vertices'].values()])

    if key == 'cell_mass':
        return get_dataset(fname, 'sigma') * get_dataset(fname, 'cell_area')

    if key == 'x_velocity':
        vr = get_dataset(fname, 'radial_velocity')
        vp = get_dataset(fname, 'phi_velocity')
        phi = get_dataset(fname,'phi')
        return vr * np.cos(phi) - vp * np.sin(phi)

    if key == 'y_velocity':
        vr = get_dataset(fname, 'radial_velocity')
        vp = get_dataset(fname, 'phi_velocity')
        phi = get_dataset(fname,'phi')
        return vr * np.sin(phi) + vp * np.cos(phi)

    raise KeyError('unknown dataset: ' + key)




def softening_radius(fname):
    h5f = h5py.File(fname, 'r')
    return h5f['run_config']['softening_radius'][()]




def domain_radius(fname):
    h5f = h5py.File(fname, 'r')
    return h5f['run_config']['domain_radius'][()]



def eccentricity_value(fname):
    h5f = h5py.File(fname, 'r')
    return h5f['run_config']['eccentricity'][()]




def two_body_position_and_velocity(E, M=1.0, a=1.0, q=1.0, e=0.0):
    omega = (M / a**3)**0.5
    mu = q / (1 + q)
    M1 = M * (1 - mu)
    M2 = M * mu
    x1 = -a * mu * (e - np.cos(E))
    y1 = +a * mu * (0 + np.sin(E)) * (1 - e**2)**0.5
    x2 = -x1 / q
    y2 = -y1 / q
    vx1 = -a * mu * omega / (1 - e * np.cos(E)) * np.sin(E)
    vy1 = +a * mu * omega / (1 - e * np.cos(E)) * np.cos(E) * (1 - e**2)**0.5
    vx2 = -vx1 / q
    vy2 = -vy1 / q
    return np.array([x1, y1]), np.array([x2, y2]), np.array([vx1, vy1]), np.array([vx2, vy2])




def E_from_M(M, e=1.0):
    f = lambda E: E - e * np.sin(E) - M
    E = scipy.optimize.newton(f, x0=M)
    return E




def get_eccentric_anomaly(fname):
    h5f = h5py.File(fname, 'r')
    t = h5f['time'][...]
    M = t
    e = eccentricity_value(fname)
    E = E_from_M(M, e=e)
    return E




def get_mean_anomaly(fname):
    h5f = h5py.File(fname, 'r')
    t = h5f['time'][...]
    return t % (2 * np.pi)




def get_true_anomaly(fname):
    e = eccentricity_value   (fname)
    E = get_eccentric_anomaly(fname)
    g = np.sqrt(1 - e * e)
    sinf = (np.sin(E) * g) / (1 - e * np.cos(E))
    cosf = (np.cos(E) - e) / (1 - e * np.cos(E))
    return math.atan2(sinf, cosf)
