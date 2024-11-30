#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy
import multiprocessing as mp
# import sys

from a03_FIELD0_BICKL import velocity_field_Bickley
from a03_FIELD0_DATA1 import velocity_field_Faraday1
from a03_FIELD0_DGYRE import velocity_field_DoubleGyre

"""
Created on Thu Mar 24 15:47:47 2022
@author: Julio Urizarna Carasa

This file contains some of the parameters needed to run the SOLVER files.
"""


################################
# Decide velocity field to use #
################################

Case_v    = np.array(["Double Gyre",      # 0
                      "Bickley jet",      # 1
                      "Faraday flow"])    # 2

Case_elem = 1



###########################################
# Define particle's and fluid's densities #
###########################################

# Densities:
# - Particle's density
rho_p   = 2.0 / 3.0

# - Fluid's density
rho_f   = 1.0 # Water's density



##############################################################################
# Define repository where data must be saved ## Import chosen velocity field #
##############################################################################

R_value = str(round((1.+2.*(rho_p/rho_f))/3., 2))

if Case_elem == 0:
    '''Double Gyre background flow.'''
    save_output_to = './R=' + R_value + '/01-Double_Gyre/'
    vel     = velocity_field_DoubleGyre(periodic=False)
elif Case_elem == 1:
    '''Bickley Jet velocity field'''
    save_output_to = './R=' + R_value + '/02-Bickley_Jet/'
    vel     = velocity_field_Bickley()
elif Case_elem == 2:
    '''Faraday Velocity Field'''
    save_output_to = './R=' + R_value + '/03-Faraday_Flow/'
    vel     = velocity_field_Faraday1(field_boundary=True)



####################
# Define time grid #
####################

if Case_elem == 0:
    
    # Time Scale of the flow
    t_scale = np.copy(vel.T)
    
    # Initial time
    tini  = (  0. * t_scale ) / t_scale
    
    # Final time, for plotting reasons we want it to be 10
    tend  = ( 10. * t_scale ) / t_scale  # Final time, remember that this is nondimensional time
    
elif Case_elem == 1:
    
    # Time Scale of the flow
    t_scale = np.copy(vel.T)
    
    # Initial time
    tini  = ( 10. * t_scale ) / t_scale
    
    # Final time, for plotting reasons we want it to be 10
    tend  = ( 30. * t_scale ) / t_scale  # Final time, remember that this is nondimensional time

elif Case_elem == 2:
        
    # Time Scale of the flow
    t_scale = np.copy(vel.T)
    
    # Initial time
    tini  = 0. / t_scale
    
    # Final time
    tend  = 10. / t_scale  # Final time, remember that this is nondimensional time
    
nt    = 1001  # Time nodes


# For experimental flows, the final time cannot go beyond a threshold.
if Case_elem == 2:
    assert tend < vel.t_lim, "Time domain for the Faraday flow is restricted to the available data."


# Create time axis
taxis  = np.linspace(tini, tend, nt)
dt     = taxis[1] - taxis[0]



####################################################
# Define parameters to calculate the Stokes number #
####################################################https://start.zorin.com/

# Kinematic viscosity
nu_f    = 8.917e-7 # Kinematic viscosity of water at 25ºC

# Particle's radius
# In an actual experiment one would need to set rad_p = a number, in this case
# I am interesting in obtaining an specific value of S, so I obtain it
# inversely with the other parameters I have.
rad_p   = np.sqrt(0.1) * np.sqrt(3. * nu_f * t_scale)



###############################################################
# Definition of the pseudo-space grid for FD as Koleva (2005) #
###############################################################

# Define Uniform grid [0,1]:
xi0_fd      = 0.0
xif_fd      = 1.0
N_fd        = 101  # Number of nodes
xi_fd_v     = np.linspace(xi0_fd, xif_fd, int(N_fd))[:-1]

# Control constant (Koleva 2005)
c           = 20

# Logarithm map to obtain QUM
x_fd_v      = -c * np.log(1.0 - xi_fd_v)



##############################################
# Decide whether to apply parallel computing #
##############################################

parallel_flag = True
number_cores  = int(mp.cpu_count()) - 1



########################################
# Define particle's initial conditions #
########################################

if Case_elem == 0:
	mat = scipy.io.loadmat('../IniCondDblGyre.mat')
elif Case_elem == 1:
	mat = scipy.io.loadmat('../IniCondBickley.mat')
elif Case_elem == 2:
	mat = scipy.io.loadmat('../IniCondFaraday.mat')

x0     = mat['X'] / vel.L  # Nondimensionalise initial positions
y0     = mat['Y'] / vel.L  # Nondimensionalise initial positions

# If initial velocity is taken equal to the background: 
u0, v0 = vel.get_velocity(x0, y0, tini)

# If initial velocity is preferred to be manually set:
# u0  = mat['vx']
# v0  = mat['vy']



############################################################
# Choose convergence order for Direct Integration and IMEX #
############################################################

# # For Daitche's method, values 1, 2 and 3 are available
order_Daitche = 2

# For IMEX implementation, define the convergence order of the finite
# differences methods. It should either be 2 or 4. Koleva's scheme
# corresponds to 2 and our own corresponds to 4.
order_FD   = 2

# For IMEX implementation, values 1, 2, 3, 4 are available
order_IMEX = 2
