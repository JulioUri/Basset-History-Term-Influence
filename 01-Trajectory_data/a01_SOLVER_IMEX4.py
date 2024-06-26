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

from a09_PRTCLE_IMEX4 import maxey_riley_imex

"""
Created on Fri Mar 18 16:17:20 2022
@author: Leon Schlegel and Julio Urizarna Carasa

This file uses 1st, 2nd, 3rd, 4th order ImEx methods specified below + 2nd
order centered difference method to generate the data of the trajectories
and relative velocities of particles moving in a velocity field.

    - 1st Order: IMEX Euler
    - 2nd Order: Ascher et al. 1997; Implicit Explicit Midpoint (1,2,2)
    - 3rd Order: Ascher et al. 1997; A third Order Combination (2,3,3)
    - 4th Order: Kennedy and Carpenter 2003; ARK4(3)6L[2]SA–ERK for the explicit part and ARK4(3)6L[2]SA–ESDIRK for the implicit part

The second order centered difference method used for the spatial discretization
is taken from M. Koleva (2006), which is a FD approach for the heat equation
for unbounded spatial domains.

The 4th order method should be primarily used. The other methods are just for comparison.
Once generated, the data is saved in some HDF5 files for a later processing.

This file generates three HDF5 files that contain the following data:

  - b06_IMEX4_RELVEL.hdf5: Particle's RELATIVE velocity at time vector points,
  - b06_IMEX4_TIMEVC.hdf5: Time vector,
  - b06_IMEX4_TRAJCT.hdf5: Particle's positions at time vector points.
"""



######################################
# Import Parameters from python file #
######################################

from a00_PMTERS_INPUT import (Case_v,  Case_elem, vel,
                              x0,      y0,        parallel_flag,
                              u0,      v0,        c,
                              rho_p,   rho_f,     x_fd_v,
                              rad_p,   nu_f,      t_scale,
                              save_output_to,     number_cores,
                              order_IMEX, order_FD, taxis,   dt,
                              tini)


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

if exists(save_output_to + "b06_IMEX" + str(order_IMEX) + \
                         "_TIMEVC-R=" + str(R) + "-St=" + str(St) + ".hdf5"):
    os.remove(save_output_to + "b06_IMEX" + str(order_IMEX) + \
                         "_TIMEVC-R=" + str(R) + "-St=" + str(St) + ".hdf5")

if exists(save_output_to + "b06_IMEX" + str(order_IMEX) + \
                         "_TRAJCT-R=" + str(R) + "-St=" + str(St) + ".hdf5"):
    os.remove(save_output_to + "b06_IMEX" + str(order_IMEX) + \
                         "_TRAJCT-R=" + str(R) + "-St=" + str(St) + ".hdf5")

if exists(save_output_to + "b06_IMEX" + str(order_IMEX) + \
                         "_RELVEL-R=" + str(R) + "-St=" + str(St) + ".hdf5"):
    os.remove(save_output_to + "b06_IMEX" + str(order_IMEX) + \
                         "_RELVEL-R=" + str(R) + "-St=" + str(St) + ".hdf5")



###################################################
# Define functions needed for the Parallelization #
###################################################

def save_hdf5(vector):
    with h5py.File(save_output_to + 'b06_IMEX' + str(order_IMEX) + \
                   '_TRAJCT-R=' + str(R) + '-St=' + str(St) + '.hdf5', 'a') as f:
        f.create_dataset(str(vector[0]), data=vector[1])
        
    with h5py.File(save_output_to + 'b06_IMEX' + str(order_IMEX) + \
                   '_RELVEL-R=' + str(R) + '-St=' + str(St) + '.hdf5', 'a') as f:
        f.create_dataset(str(vector[0]), data=vector[2])


def compute_particle(particle, taxis):
    tag     = particle.tag
    pos_vec = np.array([particle.pos_vec])
    q0_vec = np.array([particle.q0_vec])
    for tt in range(1, len(taxis)):
        particle.update()
        
        pos_vec = np.vstack((pos_vec, particle.pos_vec[tt]))
        q0_vec = np.vstack((q0_vec, particle.q0_vec[tt]))
        
        
    return (tag, pos_vec, q0_vec)



#########################################
# Create dictionary to store parameters #
#########################################

# Create dictionary that stores information of the calculation. This will
# be stored in the HDF5 so the information of the parameters with which the
# particle trajectories were calculated are always together with the data.
parameter_dic = {"Velocity field": Case_v[Case_elem],
       "Number of particles": particle_number,
       "Particle density": rho_p,
       "Fluid density": rho_f,
       "R constant": (1 + 2*(rho_p/rho_f))/3,
       "Particle radius": rad_p,
       "Kinematic viscosity": nu_f,
       "Time scale": t_scale,
       "Stokes number": (rad_p**2)/(3*t_scale*nu_f),
       "Initial time": tini,
       "Final time": taxis[-1],
       "Time step": dt,
       "Grid constant": c,
       "Number of pseudospacial nodes": len(x_fd_v),
       "Calculated in Parallel? ": parallel_flag}



####################################################
# Create files and include dataset with Parameters #
####################################################

with h5py.File(save_output_to + 'b06_IMEX' + str(order_IMEX) + \
               '_TRAJCT-R=' + str(R) + '-St=' + str(St) + '.hdf5', 'w') as f:
    f.create_dataset("Parameters", data=str(parameter_dic))
        
with h5py.File(save_output_to + 'b06_IMEX' + str(order_IMEX) + \
               '_RELVEL-R=' + str(R) + '-St=' + str(St) + '.hdf5', 'w') as f:
    f.create_dataset("Parameters", data=str(parameter_dic))



#######################
# Calculate particles #
#######################

# Create instances of IMEX_particle_set

waiting_time = 0.3

print('Calculating IMEX ' + str(order_IMEX) + ' order solutions: \n + Creating Particles:')

time.sleep(waiting_time)


t0 = time.time()

imex_particle_set = set()
for elem in progressbar(range(0, particle_number)):
    
    if type(x0) == int or type(x0) == float:
        x_ini, y_ini = x0, y0
        u_ini, v_ini = u0, v0
    else:
        # Reshape initial conditions in parallel case
        xelem        = elem  % limit
        yelem        = elem // limit
        
        x_ini, y_ini = x0[xelem, yelem], y0[xelem, yelem]
        u_ini, v_ini = u0[xelem, yelem], v0[xelem, yelem]
    
    imex_particle_set.add(maxey_riley_imex(elem+1,
                                       np.array([x_ini, y_ini]),
                                       np.array([u_ini, v_ini]), vel,
                                       x_fd_v, c, dt, tini,
                                       particle_density    = rho_p,
                                       fluid_density       = rho_f,
                                       particle_radius     = rad_p,
                                       kinematic_viscosity = nu_f,
                                       time_scale          = t_scale,
                                       IMEXOrder           = order_IMEX,
                                       FDOrder             = order_FD,
                                       parallel_flag       = parallel_flag))



##########################
# Calculate trajectories #
##########################
    
print(' + Calculating Trajectories:')

time.sleep(waiting_time)


time.sleep(waiting_time)

# For many particles, use parallel programming with maximum power of the computer.
# For one particle, we use serial.
if parallel_flag == True:
    if __name__ == '__main__':
        t0 = time.time()
        p = mp.Pool(number_cores)
        result  = [p.apply_async(compute_particle, args=(particle, taxis),
                                 callback=save_hdf5,
                                 error_callback=lambda x: print("Error")) \
                   for particle in progressbar(imex_particle_set)]
        
        p.close()
        p.join()
        tf = time.time()
        print("Calculation took " + str(tf - t0) + " seconds.")
        
else:
    
    for particle in progressbar(imex_particle_set):
        compute_particle(particle, taxis)


tf = time.time()

time.sleep(waiting_time)

print("Computing time for the calculation of the trajectories with " + str(order_IMEX) + " order IMEX method: " + str(tf - t0) + ' seconds.\n')



############################
# Save data into hdf5 file #
############################

# Save times
with h5py.File(save_output_to + 'b06_IMEX' + str(order_IMEX) + \
               '_TIMEVC-R=' + str(R) + '-St=' + str(St) + '.hdf5', 'w') as f:
    f.create_dataset("Parameters", data=str(parameter_dic))
    f.create_dataset("default", data=taxis)


# Save data in case of serial programming
if parallel_flag == False:
    with h5py.File(save_output_to + 'b06_IMEX' + str(order_IMEX) + \
                   '_TRAJCT-R=' + str(R) + '-St=' + str(St) + '.hdf5', 'a') as f:
        for particle in imex_particle_set:
            dset = f.create_dataset(str(particle.tag), data=particle.pos_vec)
        
    with h5py.File(save_output_to + 'b06_IMEX' + str(order_IMEX) + \
                   '_RELVEL-R=' + str(R) + '-St=' + str(St) + '.hdf5', 'a') as f:
        for particle in imex_particle_set:
            dset = f.create_dataset(str(particle.tag), data=particle.q0_vec)
            
print('\007')
