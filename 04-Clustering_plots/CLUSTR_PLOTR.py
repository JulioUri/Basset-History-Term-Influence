#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 14:43:01 2023

@author: cfg4065
"""
import numpy as np
import scipy
import matplotlib.pyplot as plt
from subprocess import call
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

# Define vector of strings to chose among files with and without History Term.
Method_v = np.array(["b02_STKS2", "b06_IMEX2"])

# Font size in plots:
fs = 6



#################################################
# Load data from files and plot final positions #
#################################################

# Loop over R folders
for k in range(0, len(R_folder)):
    
    # Print on console the R for which we are plotting the final positions 
    if R_v[k] == "0.78":
        print("\n1.- Plotting final trajectory point for R = 0.78.\n")
        
    elif R_v[k] == "1.22":
        print("\n3.- Plotting final trajectory point for R = 1.22.\n")
    
    
    
    # Loop over Field folders
    for i in range(0, len(Flowfield_folder)):
        
        # Load initial consitions (dimensionfull data).
        # Define length scales of the flow in case of wanting to nondimensionalise
        # Print on console the field for which we are obtaining the plot
        if Flowfield_folder[i] == "/01-Double_Gyre/":
            # Load initial conditions
            mat = scipy.io.loadmat('../IniCondDblGyre.mat')
            L_scale = 1.0
            print("      - Double Gyre.")
            
        elif Flowfield_folder[i] == "/02-Bickley_Jet/":
            # Load initial conditions
            mat = scipy.io.loadmat('../IniCondBickley.mat')
            L_scale = 1.770 
            print("      - Bickley Jet.")
            
        elif Flowfield_folder[i] == "/03-Faraday_Flow/":
            # Load initial conditions
            mat = scipy.io.loadmat('../IniCondFaraday.mat')
            L_scale = 0.052487
            print("      - Faraday Flow.")
        
        
        
        # Set name of folders where data is loaded from and saved to
        load_input_from = "../01-Trajectory_data" + R_folder[k] + Flowfield_folder[i]
        save_plot_to    = "." + R_folder[k] + Flowfield_folder[i]
        
        
        
        # Import initial positions (Divide by L_scale for nondimensionalised plot)
        x0     = mat['X'] # / L_scale
        y0     = mat['Y'] # / L_scale
        
        # Obtain dimensions of the FTLE matrix
        N, L   = np.shape(x0)
        limit  = N   # x0.shape[0]
        
        
        
        # Loop over Stokes numbers folders
        for j in range(0, len(St_v)):
            print("          - St = " + str(St_v[j]) + "\n")
            
            # set x- and y- min and max values for the colourmap
            x_min  = x0.min()
            x_max  = x0.max()
            y_min  = y0.min()
            y_max  = y0.max()
            
            
            ###########################################################
            # Uncomment following part in case of wanting to see data #
            # outside the initial domain.                             #
            ###########################################################
            
            # print("            Defining Boundaries.\n")
            
            # Loop over Integration methods folders to obtain boundary values
            # for l in range(0, len(Method_v)):
                
            #     if Method_v[l] == "b02_STKS2":
            #         print("               - Solve_ivp (no History Term).\n")
                    
            #     elif Method_v[l] == "b06_IMEX2":
            #         print("               - IMEX 2nd order method (with History Term).\n")
                
            #     # Import number of sets in the data set and trajectories
            #     with h5py.File(load_input_from + Method_v[l] + '_TRAJCT-R=' + \
            #                     R_v[k] + '-St=' + St_v[j] + '.hdf5', "r") as f:
            #         # Parameters = f["Parameters"][()]
            #         n        = len(f.keys()) - 1 # Parameters data-set does not count
                
            #         # Obtain trajectories
            #         for elem in range(0,n):
            #             if elem == 0:
            #                 traj_f = f.get(str(elem + 1))[:][-1] * L_scale
            #             else:
            #                 # print(str(elem))
            #                 if elem != 20832: traj_f = np.vstack((traj_f, f.get(str(elem + 1))[:][-1])) * L_scale
                
            #     # Reset min and max values if particles leave the domain
            #     #if x_min > traj_f[:,0].min():
            #     #    x_min = traj_f[:,0].min()
                    
            #     #if x_max < traj_f[:,0].max():
            #     #    x_max = traj_f[:,0].max()
                    
            #     #if y_min > traj_f[:,1].min():
            #     #    y_min = traj_f[:,1].min()
                
            #     #if y_max < traj_f[:,1].max():
            #     #    y_max = traj_f[:,1].max()
            
            
            
            # Loop over Integration methods folders
            print("            Plotting.\n")
            for l in range(0, len(Method_v)):
                
                # Print on console the type of data used in the plotting:
                # with or without history term
                if Method_v[l] == "b02_STKS2":
                    print("               - Solve_ivp (no History Term).\n")
                    
                elif Method_v[l] == "b06_IMEX2":
                    print("               - IMEX 2nd order method (with History Term).\n")
                
                
                
                # Obtain final time positions
                with h5py.File(load_input_from + Method_v[l] + '_TRAJCT-R=' + \
                               R_v[k] + '-St=' + St_v[j] + '.hdf5', "r") as f:
                    # Parameters = f["Parameters"][()]
                    n        = len(f.keys()) - 1 # Parameters data-set does not count
                    
                    
                    
                    # Save final position values (careful, this data is
                    # nondimensional, since we use the nondimensional version
                    # of the MRE)
                    for elem in range(0,n):
                        if elem == 0:
                            traj_f = f.get(str(elem + 1))[:][-1]
                        else:
                            try:
                                	traj_f = np.vstack((traj_f, f.get(str(elem + 1))[:][-1]))
                            except:
                                	pass
                
                
                
                # Multiply by the lengthscale to obtain dimensional data.
                traj_f *= L_scale
                
                
                
                # Start plotting process
                if Flowfield_folder[i] == "/03-Faraday_Flow/":
                    fig = plt.figure(1, layout='tight', figsize=(2.5, 2.5))
                    
                else:
                    fig = plt.figure(1, layout='tight', figsize=(2.5, 1.4))
                    
                #colors = cm.rainbow(np.linspace(0, 1, int(N*L)))
                if Method_v[l] == "b02_STKS2":
                    plt.title("Particle clustering without History, S=" + \
                              str(St_v[j]), fontsize=fs)
                        
                elif Method_v[l] == "b06_IMEX2":
                    plt.title("Particle clustering with History, S=" + \
                              str(St_v[j]), fontsize=fs)
                        
                plt.scatter(traj_f[:,0], traj_f[:,1], s=0.15, color="grey", linewidth=0)
                plt.xlabel("x", fontsize=fs)
                plt.ylabel("y", fontsize=fs)
                plt.xlim([x_min, x_max])
                plt.ylim([y_min, y_max])
                
                plt.tick_params(axis='both', labelsize=fs)
                
                # EPS
                plt.savefig(save_plot_to + Method_v[l] + '_CLUST-R=' + R_v[k] +\
                            '-St=' + St_v[j] + '.eps', format='eps', dpi=200,
                            bbox_inches='tight')
                
                # PDF
                plt.savefig(save_plot_to + Method_v[l] + '_CLUST-R=' + R_v[k] +\
                            '-St=' + St_v[j] + '.pdf', format='pdf', dpi=200,
                            bbox_inches='tight')
                call(["pdfcrop", save_plot_to + Method_v[l] + '_CLUST-R=' + R_v[k] +\
                            '-St=' + St_v[j] + '.pdf',
                            save_plot_to + Method_v[l] + '_CLUST-R=' + R_v[k] +\
                                        '-St=' + St_v[j] + '.pdf'])
                
                # JPG
                plt.savefig(save_plot_to + Method_v[l] + '_CLUST-R=' + R_v[k] +\
                            '-St=' + St_v[j] + '.jpg', format='jpg', dpi=500,
                            bbox_inches='tight')
                    
                plt.close(fig)
