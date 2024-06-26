
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('../src')
import numpy as np
import h5py
import time
import os
from os.path import exists
import multiprocessing as mp
from progressbar import progressbar

from a09_PRTCLE_STOKS import particle_stokes

"""
Created on Fri Mar 18 16:17:20 2022
@author: Julio Urizarna Carasa

This file generates the trajectory and velocity data for Stokes particles
moving in a velocity field by using 

Once generated, the data is saved in some HDF5 files for a later processing.
"""



######################################
# Import Parameters from python file #
######################################

from a00_PMTERS_INPUT import (Case_v, Case_elem,
                              taxis,   dt,        tini,
                              x0,      y0,        vel,
                              u0,      v0,        parallel_flag,
                              rho_p,   rho_f,
                              rad_p,   nu_f,      t_scale,
                              save_output_to,     number_cores)



#########################################################
# Check for errors and define total number of particles #
#########################################################

if type(x0) == int or type(x0) == float:
    particle_number = 1
    parallel_flag   = False
else:
    assert len(x0) == len(y0) and \
            len(u0) == len(v0) and \
             len(x0) == len(u0), "Different dimentions of the Initial conditions."
    
    limit           = x0.shape[0]
    particle_number = x0.size



##############################
# Delete files with OLD data #
##############################

St = round((rad_p**2)/(3*t_scale*nu_f), 2)
R  = round((1. + 2.*(rho_p / rho_f)) / 3., 2)

if exists(save_output_to + "b02_STOKS_TIMEVC-R=" + str(R) + "-St=" + str(St) + ".hdf5"):
    os.remove(save_output_to + "b02_STOKS_TIMEVC-R=" + str(R) + "-St=" + str(St) + ".hdf5")

if exists(save_output_to + "b02_STOKS_TRAJCT-R=" + str(R) + "-St=" + str(St) + ".hdf5"):
    os.remove(save_output_to + "b02_STOKS_TRAJCT-R=" + str(R) + "-St=" + str(St) + ".hdf5")

if exists(save_output_to + "b02_STOKS_VELOCT-R=" + str(R) + "-St=" + str(St) + ".hdf5"):
    os.remove(save_output_to + "b02_STOKS_VELOCT-R=" + str(R) + "-St=" + str(St) + ".hdf5")



###################################################
# Define functions needed for the Parallelization #
###################################################

def save_hdf5(vector):
    with h5py.File(save_output_to + 'b02_STOKS_TRAJCT-R=' + str(R) + '-St=' + str(St) + '.hdf5', 'a') as f:
        f.create_dataset(str(vector[0]), data=vector[1])
        
    with h5py.File(save_output_to + 'b02_STOKS_VELOCT-R=' + str(R) + '-St=' + str(St) + '.hdf5', 'a') as f:
        f.create_dataset(str(vector[0]), data=vector[2])

def compute_particle(particle, taxis):
    
    tag     = particle.tag
    pos_vec = np.array([particle.pos_vec])
    vel_vec = np.array([particle.pos_vec])
    
    for tt in range(1, len(taxis)):
        h        = taxis[tt] - taxis[tt-1]
        particle.update(h)
        
        pos_vec = np.vstack((pos_vec, particle.pos_vec[tt]))
        vel_vec = np.vstack((vel_vec, particle.vel_vec[tt]))
    
    return (tag, pos_vec, vel_vec)



###################################
# Create dictionary to store data #
###################################

parameter_dic = {"Velocity field": Case_v[Case_elem],
       "Initial time": tini,
       "Final time": taxis[-1],
       "Time step": dt,
       "Particle density": rho_p,
       "Fluid density": rho_f,
       "Particle radius": rad_p,
       "Kinematic viscosity": nu_f,
       "Time scale": t_scale,
       "Stokes number": (rad_p**2)/(3*t_scale*nu_f),
       "R constant": (1 + 2*(rho_p/rho_f))/3,
       "Number of particles": particle_number,
       "Calculated in Parallel? ": parallel_flag}



####################################################
# Create files and include dataset with Parameters #
####################################################

with h5py.File(save_output_to + 'b02_STOKS_TRAJCT-R=' + str(R) + '-St=' + str(St) + '.hdf5', 'a') as f:
    f.create_dataset("Parameters", data=str(parameter_dic))
        
with h5py.File(save_output_to + 'b02_STOKS_VELOCT-R=' + str(R) + '-St=' + str(St) + '.hdf5', 'a') as f:
    f.create_dataset("Parameters", data=str(parameter_dic))



##############################
# Calculate Stokes solutions #
##############################

waiting_time = 0.3

print('Calculating Stokes solutions: \n + Creating Particles:')

time.sleep(waiting_time)


t0 = time.time()

Stokes_particle_set = set()
for elem in progressbar(range(0, particle_number)):
    
    if type(x0) == int or type(x0) == float:
        x_ini, y_ini = x0, y0
        u_ini, v_ini = u0, v0
    else:
        xelem        = elem  % limit
        yelem        = elem // limit
        
        x_ini, y_ini = x0[xelem, yelem], y0[xelem, yelem]
        u_ini, v_ini = u0[xelem, yelem], v0[xelem, yelem]
        
    Stokes_particle_set.add(particle_stokes(elem+1,
                                 np.array([x_ini, y_ini]),
                                 np.array([u_ini, v_ini]),
                                 tini, vel,
                                 particle_density    = rho_p,
                                 fluid_density       = rho_f,
                                 particle_radius     = rad_p,
                                 kinematic_viscosity = nu_f,
                                 time_scale          = t_scale))



##########################
# Calculate trajectories #
##########################
    
print(' + Calculating Trajectories:')

time.sleep(waiting_time)

if parallel_flag == True:
    if __name__ == '__main__':
        p = mp.Pool(number_cores)
        result = [p.apply_async(compute_particle,
                                args=(particle, taxis),
                                callback = save_hdf5) \
                  for particle in progressbar(Stokes_particle_set)]
        p.close()
        p.join()
else:
    for particle in progressbar(Stokes_particle_set):
        result = compute_particle(particle, taxis)


tf = time.time()

time.sleep(waiting_time)

print("Computing time for Stokes & Tracer: " + str(tf - t0) + ' seconds.\n')



############################
# Save data into hdf5 file #
############################

# Save times
with h5py.File(save_output_to + 'b02_STOKS_TIMEVC-R=' + str(R) + '-St=' + str(St) + '.hdf5', 'w') as f:
    f.create_dataset("Parameters", data=str(parameter_dic))
    f.create_dataset("default", data=taxis)
    
# print("\007")


if parallel_flag == False:
    with h5py.File(save_output_to + 'b02_STOKS_TRAJCT-R=' + str(R) + '-St=' + str(St) + '.hdf5', 'w') as f:
        f.create_dataset("Parameters", data=str(parameter_dic))
        for particle in Stokes_particle_set:
            f.create_dataset(str(particle.tag), data=particle.pos_vec)
        
    with h5py.File(save_output_to + 'b02_STOKS_VELOCT-R=' + str(R) + '-St=' + str(St) + '.hdf5', 'w') as f:
        f.create_dataset("Parameters", data=str(parameter_dic))
        for particle in Stokes_particle_set:
            f.create_dataset(str(particle.tag), data=particle.vel_vec)

print('\007')
