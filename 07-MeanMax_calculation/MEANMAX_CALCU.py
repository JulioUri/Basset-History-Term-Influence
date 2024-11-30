#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 14:43:01 2023

@author: cfg4065
"""
import numpy as np
import scipy
from random import randint
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
St_v = np.array(["0.1", "1.0", "10.0"])



############################################
# Load data from files and calculate FTLEs #
############################################

# Create variables for min and max FTLE values
zmin_Gyre, zmax_Gyre = 10., 1.
zmin_Bkly, zmax_Bkly = 10., 1.
zmin_Frdy, zmax_Frdy = 10., 1.

n_dic   = dict()
vel_dic = dict()
max_dic = dict()

# Loop over R values [0.78, 1.22]
for k in range(0, len(R_folder)):
    
    # Print text on console to know which R we are calculating
    if R_folder[k] == "/R=0.78":
        print("1.- Calculate FTLEs for R = 0.78.")
    elif R_folder[k] == "/R=1.22":
        print("3.- Calculate FTLEs for R = 1.22.")
    
    
    
    # Loop over flow fields [Double gyre, Bickley jet, Faraday flow]
    for i in range(0, len(Flowfield_folder)):
        
        # Load initial position data (dimensionfull data).
        # Define the lengthscale of the flow, in case we wanted to
        # nondimensionalise.
        # Print text on console to indicate which field we are calculating.
        if Flowfield_folder[i] == "/01-Double_Gyre/":
            # Import Initial Condition file
            mat = scipy.io.loadmat('../IniCondDblGyre.mat')
            L_scale = 1.0
            print("      - In the Double Gyre.")
            
        elif Flowfield_folder[i] == "/02-Bickley_Jet/":
            # Import Initial Condition file
            mat = scipy.io.loadmat('../IniCondBickley.mat')
            L_scale = 1.770
            print("      - Bickley Jet.")
            
        elif Flowfield_folder[i] == "/03-Faraday_Flow/":
            # Import Initial Condition file
            mat = scipy.io.loadmat('../IniCondFaraday.mat')
            L_scale = 0.052487
            print("      - Faraday Flow.")
        
        
        
        # Define address to directories to load and save data
        load_input_from = "../01-Trajectory_data" + R_folder[k] + Flowfield_folder[i]
        
        
        
        # Loop over Stokes numbers
        for j in range(0, len(St_v)):
            
            # Print on console the Stokes number we are calculating
            print("          - St = " + str(St_v[j]))
                
                
                
            # Import Trajectory data
            with h5py.File(load_input_from + 'b06_IMEX2_RELVEL-R=' +\
                           R_v[k] + '-St=' + St_v[j] + '.hdf5', "r") as f:
                Parameters = f["Parameters"][()]
                n        = len(f.keys()) - 1 # Parameters data-set does not count
                
                n1       = randint(1, n)
                relvel_1 = f.get(str(n1))[:]
                absvel_1 = np.linalg.norm(relvel_1, ord=2, axis=1)
                avevel_1 = np.mean(absvel_1)
                maxvel_1 = max(absvel_1)
                n_dic[Flowfield_folder[i] + St_v[j] + R_folder[k] + "/1"] = n1
                vel_dic[Flowfield_folder[i] + St_v[j] + R_folder[k] + "/1"] = avevel_1
                max_dic[Flowfield_folder[i] + St_v[j] + R_folder[k] + "/1"] = maxvel_1
                
                n2       = randint(1, n)
                relvel_2 = f.get(str(n2))[:]
                absvel_2 = np.linalg.norm(relvel_2, ord=2, axis=1)
                avevel_2 = np.mean(absvel_2)
                maxvel_2 = max(absvel_2)
                n_dic[Flowfield_folder[i] + St_v[j] + R_folder[k] + "/2"] = n2
                vel_dic[Flowfield_folder[i] + St_v[j] + R_folder[k] + "/2"] = avevel_2
                max_dic[Flowfield_folder[i] + St_v[j] + R_folder[k] + "/2"] = maxvel_2
                
                n3       = randint(1, n)
                relvel_3 = f.get(str(n3))[:]
                absvel_3 = np.linalg.norm(relvel_3, ord=2, axis=1)
                avevel_3 = np.mean(absvel_3)
                maxvel_3 = max(absvel_3)
                n_dic[Flowfield_folder[i] + St_v[j] + R_folder[k] + "/3"] = n3
                vel_dic[Flowfield_folder[i] + St_v[j] + R_folder[k] + "/3"] = avevel_3
                max_dic[Flowfield_folder[i] + St_v[j] + R_folder[k] + "/3"] = maxvel_3
            
with open('Mean-and-max-Relative-velocities.txt', 'w') as file:
    file.write("Mean and maximum relative velocities for particle " +\
               "trajectories calculated with history. \n\n")
    
    for i in range(0, len(Flowfield_folder)):
        if Flowfield_folder[i] == "/01-Double_Gyre/":
            file.write("- In the double gyre: \n")
            
        elif Flowfield_folder[i] == "/02-Bickley_Jet/":
            file.write("- In the Bickley jet: \n")
            
        elif Flowfield_folder[i] == "/03-Faraday_Flow/":
            file.write("- In the Faraday flow: \n")
        
        for j in range(0, len(St_v)):
            if St_v[j] == "0.1":
                file.write("   - With St=0.1:\n")
                
            elif St_v[j] == "1.0":
                file.write("   - With St=1.0:\n")
                
            elif St_v[j] == "10.0":
                file.write("   - With St=10.0:\n")
            
            for k in range(0, len(R_folder)):
                if R_folder[k] == "/R=0.78":
                    file.write("      - R=0.78: \n")
                    
                elif R_folder[k] == "/R=1.22":
                    file.write("      - R=1.22: \n")
                
                file.write("         - Particle number: " + str(n_dic[Flowfield_folder[i] + St_v[j] + R_folder[k] + "/1"]) + "\n")
                file.write("            - Mean relative velocity: " + str(vel_dic[Flowfield_folder[i] + St_v[j] + R_folder[k] + "/1"]) + "\n")
                file.write("            - Max. relative velocity: " + str(max_dic[Flowfield_folder[i] + St_v[j] + R_folder[k] + "/1"]) + "\n")
                file.write("        - Particle number: " + str(n_dic[Flowfield_folder[i] + St_v[j] + R_folder[k] + "/2"]) + "\n")
                file.write("            - Mean relative velocity: " + str(vel_dic[Flowfield_folder[i] + St_v[j] + R_folder[k] + "/2"]) + "\n")
                file.write("            - Max. relative velocity: " + str(max_dic[Flowfield_folder[i] + St_v[j] + R_folder[k] + "/2"]) + "\n")
                file.write("        - Particle number: " + str(n_dic[Flowfield_folder[i] + St_v[j] + R_folder[k] + "/3"]) + "\n")
                file.write("            - Mean relative velocity: " + str(vel_dic[Flowfield_folder[i] + St_v[j] + R_folder[k] + "/3"]) + "\n")
                file.write("            - Max. relative velocity: " + str(max_dic[Flowfield_folder[i] + St_v[j] + R_folder[k] + "/3"]) + "\n")
