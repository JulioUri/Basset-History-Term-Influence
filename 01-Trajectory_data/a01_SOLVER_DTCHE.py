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

from a09_PRTCLE_DTCHE import maxey_riley_Daitche

"""
Created on Fri Mar 18 16:17:20 2022
@author: Julio Urizarna Carasa

This file uses 1st, 2nd and 3rd order methods provided in A. Daitche (2013)
to generate the data of the trajectories and relative velocities of particles
moving in a velocity field.

Daitche's methods are based on the combination of a quadrature scheme for
the integral term of the Basset History Term and three different
Adams-Bashforth multi-step methods. More information can be found in
Daitche (2013).

This file generates three HDF5 files that contain the following data:

  - b05_DTCHE_VELOCT.hdf5: Particle's ABSOLUTE velocity at time vector points,
  - b05_DTCHE_TIMEVC.hdf5: Time vector,
  - b05_DTCHE_TRAJCT.hdf5: Particle's positions at time vector points.
  
"""


######################################
# Import Parameters from python file #
######################################


from a00_PMTERS_INPUT import ( Case_v,  Case_elem,  t_scale,  vel,
                               x0,      y0,         parallel_flag,
                               u0,      v0,
                               rho_p,   rho_f,
                               rad_p,   nu_f,
                               order_Daitche,   number_cores,
                               save_output_to, taxis)


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


if exists(save_output_to + "b05_DTCHE_TIMEVC-R=" + str(R) + "-St=" + str(St)  +  ".hdf5"):
    os.remove(save_output_to + "b05_DTCHE_TIMEVC-R=" + str(R) + "-St=" + str(St) + ".hdf5")

if exists(save_output_to + "b05_DTCHE_TRAJCT-R=" + str(R) + "-St=" + str(St) + ".hdf5"):
    os.remove(save_output_to + "b05_DTCHE_TRAJCT-R=" + str(R) + "-St=" + str(St) + ".hdf5")

if exists(save_output_to + "b05_DTCHE_VELOCT-R=" + str(R) + "-St=" + str(St) + ".hdf5"):
    os.remove(save_output_to + "b05_DTCHE_VELOCT-R=" + str(R) + "-St=" + str(St) + ".hdf5")



###################################################
# Define functions needed for the Parallelization #
###################################################

def save_hdf5(vector):
    with h5py.File(save_output_to + 'b05_DTCHE_TRAJCT-R=' + str(R) + '-St=' + str(St) + '.hdf5', 'a') as f:
        f.create_dataset(str(vector[0]), data=vector[1])
        
    with h5py.File(save_output_to + 'b05_DTCHE_VELOCT-R=' + str(R) + '-St=' + str(St) + '.hdf5', 'a') as f:
        f.create_dataset(str(vector[0]), data=vector[2])


def compute_particle(particle, taxis, conv_order):
    
    tag     = particle.tag
    
    if conv_order == 1:
        particle.Euler(taxis, flag=True)  # First order method
    elif conv_order == 2:
        particle.AdamBashf2(taxis, flag=True) # Second order method
    elif conv_order == 3:
        particle.AdamBashf3(taxis, flag=True) # Third order method
    
    return (tag, particle.pos_vec, particle.q0_vec)




###################################
# Create dictionary to store data #
###################################

# Create dictionary that stores information of the calculation. This will
# be stored in the HDF5 so the information of the parameters with which the
# particle trajectories were calculated are always together with the data.
parameter_dic = {"Velocity field": Case_v[Case_elem],
       "Number of particles": particle_number,
       "Particle density": rho_p,
       "Fluid density": rho_f,
       "Particle radius": rad_p,
       "Kinematic viscosity": nu_f,
       "Time scale": t_scale,
       "Stokes number": round((rad_p**2)/(3*t_scale*nu_f), 2),
       "R constant": round((1 + 2*(rho_p/rho_f))/3, 2),
       "Initial time": taxis[0],
       "Final time": taxis[-1],
       "Time step": taxis[1] - taxis[0],
       "Calculated in Parallel? ": parallel_flag}



####################################################
# Create files and include dataset with Parameters #
####################################################

with h5py.File(save_output_to + 'b05_DTCHE_TRAJCT-R=' + str(R) + '-St=' + str(St) + '.hdf5', 'w') as f:
    f.create_dataset("Parameters", data=str(parameter_dic))
        
with h5py.File(save_output_to + 'b05_DTCHE_VELOCT-R=' + str(R) + '-St=' + str(St) + '.hdf5', 'w') as f:
    f.create_dataset("Parameters", data=str(parameter_dic))
    


####################
# Create Particles #
####################

# Create instances of maxey_riley_Daitche class

waiting_time = 0.3

print('Calculating Direct solution: ')
print(' + Creating Particles: ')

time.sleep(waiting_time)


t0 = time.time()

Daitche_particle_set = set()
for elem in progressbar(range(0, particle_number)):
    
    if type(x0) == int or type(x0) == float:
        x_ini, y_ini = x0, y0
        u_ini, v_ini = u0, v0
    else:
        xelem        = elem  % limit
        yelem        = elem // limit
        
        x_ini, y_ini = x0[xelem, yelem], y0[xelem, yelem]
        u_ini, v_ini = u0[xelem, yelem], v0[xelem, yelem]
    
    
    Daitche_particle_set.add(maxey_riley_Daitche(elem+1,
                                    np.array([x_ini, y_ini]),
                                    np.array([u_ini, v_ini]),
                                    vel, len(taxis),
				    order_Daitche,
                                    particle_density    = rho_p,
                                    fluid_density       = rho_f,
                                    particle_radius     = rad_p,
                                    kinematic_viscosity = nu_f,
                                    time_scale          = t_scale))



#######################
# Obtain trajectories #
#######################
    
print(' + Calculating Trajectories:')

time.sleep(waiting_time)

if parallel_flag == True:
    if __name__ == '__main__':
        p = mp.Pool(number_cores)
        result = [p.apply_async(compute_particle,
                                args=(particle, taxis, order_Daitche),
                                callback = save_hdf5) \
                                for particle in progressbar(Daitche_particle_set)]
        p.close()
        p.join()
else:
    for particle in progressbar(Daitche_particle_set):
        compute_particle(particle, taxis, order_Daitche)


tf = time.time()

time.sleep(waiting_time)

print("Computing time for the calculation of the trajectory with Direct Integration: " + str(tf - t0) + ' seconds.\n')



############################
# Save data into hdf5 file #
############################

# Save times
with h5py.File(save_output_to + 'b05_DTCHE_TIMEVC-R=' + str(R) + '-St=' + str(St) + '.hdf5', 'w') as f:
    f.create_dataset("Parameters", data=str(parameter_dic))
    f.create_dataset("default", data=taxis)


if parallel_flag == False: # If run in Serial
    # Save all trajectories:
    with h5py.File(save_output_to + 'b05_DTCHE_TRAJCT-R=' + str(R) + '-St=' + str(St) + '.hdf5', 'a') as f:
        for particle in Daitche_particle_set:
            f.create_dataset(str(particle.tag), data=particle.pos_vec)


    # Save all relative velocities, i.e. q(0,t):
    with h5py.File(save_output_to + 'b05_DTCHE_VELOCT-R=' + str(R) + '-St=' + str(St) + '.hdf5', 'a') as f:
        for particle in Daitche_particle_set:
            f.create_dataset(str(particle.tag), data=particle.q0_vec)

print('\007')
