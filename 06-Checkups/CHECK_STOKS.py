#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 14:43:01 2023

@author: cfg4065
"""
import numpy as np
import scipy
import h5py



###################################
# Define Variables to import data #
###################################

# Decide which value of R is used:
## R=7/9;   R=1;   R=11/9
R_folder = np.array(["/R=0.78", "/R=1.22"])

R_v      = np.array(["0.78", "1.22"])

# Decide which field flow is used:
## 1: Double Gyre;   3: Bickley Jet;   4: Faraday Flow
Flowfield_folder = np.array(["/01-Double_Gyre/", "/02-Bickley_Jet/", "/03-Faraday_Flow/"])

# Define vector of Stokes numbers that loop over all files.
St_v = np.array(["0.01", "0.1", "1.0", "10.0"])



############################################
# Load data from files and calculate FTLEs #
############################################

for k in range(0, len(R_folder)):
    if R_v[k] == "0.78":
        print("1.- Checking final trajectory points for R = 0.78.")
    elif R_v[k] == "1.22":
        print("3.- Checking final trajectory points for R = 1.22.")
    
    for i in range(0, len(Flowfield_folder)):
        if Flowfield_folder[i] == "/01-Double_Gyre/":
            # Import Initial Condition file
            mat = scipy.io.loadmat('../IniCondDblGyre.mat')
            print("      - In the Double Gyre.")
            
        elif Flowfield_folder[i] == "/02-Bickley_Jet/":
            # Import Initial Condition file
            mat = scipy.io.loadmat('../IniCondBickley.mat')
            print("      - Bickley Jet.")
            
        elif Flowfield_folder[i] == "/03-Faraday_Flow/":
            # Import Initial Condition file
            mat = scipy.io.loadmat('../IniCondFaraday.mat')
            print("      - Faraday Flow.")
        
        load_input_from = "../01-Trajectory-data" + R_folder[k] + Flowfield_folder[i]
        save_output_in  = "." + R_folder[k] + Flowfield_folder[i]
        
        # Import initial positions
        x0     = mat['X']
        y0     = mat['Y']
        
        N, L   = np.shape(x0)
        n      = int(N * L)
        
        average_v = np.array([])
        max_v     = np.array([])
        # Loop over Stokes numbers
        for j in range(0, len(St_v)):
            print("          - St = " + str(St_v[j]))
            
            # Loop over all particles
            for elem in range(0, n):
                # Take Leap-frog method data
                with h5py.File(load_input_from + 'b02_STOKS_TRAJCT-R=' +\
                           R_v[k] + '-St=' + St_v[j] + '.hdf5', "r") as f:
                
                    LeapFrog_traj = f[str(elem+1)][()][-1]
                
                # Obtain Solve_ivp (RK45) data
                with h5py.File(load_input_from + 'b02_STKS2_TRAJCT-R=' +\
                           R_v[k] + '-St=' + St_v[j] + '.hdf5', "r") as f:
                
                    SolveIvp_traj = f[str(elem+1)][()][-1]
                
                if elem == 0:
                    LeapFrog_v = LeapFrog_traj
                    SolveIvp_v = SolveIvp_traj
                else:
                    LeapFrog_v = np.vstack((LeapFrog_v, LeapFrog_traj))
                    SolveIvp_v = np.vstack((SolveIvp_v, SolveIvp_traj)) 
            
            # Calculate difference
            Diff_traj = np.linalg.norm(LeapFrog_v - SolveIvp_v, axis=1)
            
            Diff_ave  = np.average(Diff_traj)
            Diff_max  = np.linalg.norm(Diff_traj, ord=np.inf)
            
            average_v = np.append(average_v, Diff_ave)
            max_v     = np.append(max_v, Diff_max)
            
        # Save data
        with open(save_output_in + 'Stokes-Check-R=' + R_v[k] + '.txt', 'w') as file:
            file.write("Average and maximum difference of trajectories calculated with " +\
                       "Leap Frog method and Solve_Ivp method for the ")
            if Flowfield_folder[i] == "/01-Double_Gyre/":
                file.write( "Double Gyre ")
            elif Flowfield_folder[i] == "/02-Bickley_Jet/":
                file.write( "Bickley Jet ")
            elif Flowfield_folder[i] == "/03-Faraday_Flow/":
                file.write( "Faraday flow ")
            file.write( "with parameters, " )
            if R_v[k] == "0.78":
                file.write( "R = 7/9 " )
            elif R_v[k] == "1.0":
                file.write( "R = 1 " )
            elif R_v[k] == "1.22":
                file.write( "R = 11/9 " )
            file.write( "and\n")
            
            for j in range(0, len(St_v)):
                file.write("\n - S = " + St_v[j] + ":\n" )
                file.write( "    - Average = " + str(average_v[j]) + ",\n" )
                file.write( "    - Maximum difference = " + str(max_v[j]) + ".\n" )

