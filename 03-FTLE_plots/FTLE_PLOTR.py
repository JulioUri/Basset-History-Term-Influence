#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 14:43:01 2023

@author: cfg4065
"""
import numpy as np
import matplotlib.pyplot as plt
import h5py
from subprocess import call



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
St_v = np.array(["3.0"])

# Define vector of strings to chose among files with and without History Term.
Method_v = np.array(["b02_STKS2", "b06_IMEX2"])

# Font size in plots:
fs = 6



########################
# Load data from files #
########################

zmin_Gyre = np.load("../02-FTLE_data/zmin_Gyre.npy")
zmax_Gyre = np.load("../02-FTLE_data/zmax_Gyre.npy")

zmin_Bkly = np.load("../02-FTLE_data/zmin_Bkly.npy")
zmax_Bkly = np.load("../02-FTLE_data/zmax_Bkly.npy")

zmin_Frdy = np.load("../02-FTLE_data/zmin_Frdy.npy")
zmax_Frdy = np.load("../02-FTLE_data/zmax_Frdy.npy")



###############################
# Start FTLE plotting process #
###############################

count = 0
# Loop over folders with different R values
for k in range(0, len(R_folder)):
    
    # Print in console the values of R for which we are currently
    # plotting the data
    if R_folder[k] == "/R=0.78":
        print("1.- Plotting FTLEs for R = 0.78.")
        
    elif R_folder[k] == "/R=1.22":
        print("3.- Plotting FTLEs for R = 1.22.")
    
    
    
    # Loop over field folders
    for i in range(0, len(Flowfield_folder)):
        
        # Import shrink_scale for a more proportionate bar in the colormap.
        # Import min and max FTLE values so it is constant among FTLE plots
        # of the same field.
        # Print on the console the name of the field for which we are
        # currently calculating data.
        if Flowfield_folder[i] == "/01-Double_Gyre/":
            shrink_scale = 0.5
            zmin = zmin_Gyre
            zmax = zmax_Gyre
            print("      - In the Double Gyre.")
            
        elif Flowfield_folder[i] == "/02-Bickley_Jet/":
            shrink_scale = 0.45
            zmin = zmin_Bkly
            zmax = zmax_Bkly
            print("      - Bickley Jet.")
            
        elif Flowfield_folder[i] == "/03-Faraday_Flow/":
            shrink_scale = 0.5
            zmin = zmin_Frdy
            zmax = zmax_Frdy
            print("      - Faraday Flow.")
        
        
        
        # Set folder addresses
        load_input_from = "../02-FTLE_data" + R_folder[k] + Flowfield_folder[i]
        save_plot_to    = "." + R_folder[k] + Flowfield_folder[i]
        
        
        
        # Loop over Stokes numbers folders
        for j in range(0, len(St_v)):
            
            # Print on console the Stokes number for which we are currently
            # generating the plots
            print("\n          - St = " + str(St_v[j]) + ":\n")
            
            
            
            # Import file with differences
            with h5py.File(load_input_from + 'b00_DIFFR_iFTLE-R=' + R_v[k] + \
                           '-St=' + St_v[j] + '.hdf5', "r") as f:
                x0         = f["X_0"][()]
                y0         = f["Y_0"][()]
                FTLEdiff   = f["FTLE diff in %"][()]
            
            count += 1
            
            
            #
            #############
            # Plot FTLE #
            #############
            #
            fig, axs = plt.subplots(1, 2, sharey=True, figsize=(4.5, 2.5))
            
            
            
            # Loop over Integrating methods folders
            for l in range(0, len(Method_v)):
                
                # Print on console the kind of particle trajectories we
                # are using right now to generate the plot: with or without
                # history term
                if Method_v[l] == "b02_STKS2":
                    print("               - Without History Term.")
                    
                elif Method_v[l] == "b06_IMEX2":
                    print("               - With History Term.\n")
                
                
                
                # Import FTLE data
                with h5py.File(load_input_from + Method_v[l] + '_iFTLE-R=' + \
                               R_v[k] + '-St=' + St_v[j] + '.hdf5', "r") as f:
                    FTLEmap  = f["FTLE"][()]
                
                
                
                # Create colourmap
                im    = axs[l].imshow(FTLEmap, cmap='jet',
                                      interpolation='bilinear',
                                      vmin = zmin, vmax = zmax,
                                      extent = [x0.min(), x0.max(),
                                                y0.min(), y0.max()])
                
                
                
                # Set plotting parameters
                if Flowfield_folder[i] == "/02-Bickley_Jet/":
                    axs[l].set_xlabel("x / Mm", fontsize=fs)
                    if l == 0: axs[l].set_ylabel("y / Mm", fontsize=fs)
                else:
                    axs[l].set_xlabel("x / m", fontsize=fs)
                    if l == 0: axs[l].set_ylabel("y / m", fontsize=fs)
                
                if Method_v[l] == "b02_STKS2":
                    axs[l].set_title("FTLE without History, S=" + str(St_v[j]), fontsize=fs)
                    
                elif Method_v[l] == "b06_IMEX2":
                    axs[l].set_title("FTLE with History, S=" + str(St_v[j]), fontsize=fs)
            
            
            
            # Keep setting plotting parameters
            fig.subplots_adjust(right=0.85)
            cbar_ax = fig.add_axes([0.87, 0.27, 0.012, 0.43])
            clb = fig.colorbar(im, shrink=shrink_scale, cax=cbar_ax)
            clb.ax.tick_params(labelsize=fs)
            plt.rcParams.update({'font.size': fs})
            
            
            
            # Save plot
            # EPS
            plt.savefig(save_plot_to + 'b00_PLOTS_FTLEXP-R=' + R_v[k] + '-St=' + St_v[j] + '.eps',
                        format='eps', dpi=200, bbox_inches='tight')
            
            
            # PDF
            plt.savefig(save_plot_to + 'b00_PLOTS_FTLEXP-R=' + R_v[k] + '-St=' + St_v[j] + '.pdf',
                        format='pdf', dpi=200, bbox_inches='tight')            
            
            # Crop borders
            call(["pdfcrop", save_plot_to + \
                  'b00_PLOTS_FTLEXP-R=' + R_v[k] + '-St=' + St_v[j] + '.pdf',
                   save_plot_to + \
                  'b00_PLOTS_FTLEXP-R=' + R_v[k] + '-St=' + St_v[j] + '.pdf'])
            
            plt.close(fig)
            
            
            # Print on folder that we are generating the plot with
            # the differences
            print("\n               - Plot with differences.\n")
            
            
            
            # Plot differences on FTLE for Stokes and Full MRE #
            count += 1
            
            
            
            # Plot parameters for Difference plots
            fig = plt.figure(count, layout='tight', figsize=(2.5, 2.5))
            
            plt.title("Difference in % of the FTLE, S=" + str(St_v[j]), fontsize=fs)
            c    = plt.imshow(FTLEdiff, cmap='jet',
                              interpolation='bilinear',
                              vmin = -100, vmax = 100,
                              extent = [x0.min(), x0.max(),
                                        y0.min(), y0.max()])
            
            if Flowfield_folder[i] == "/02-Bickley_Jet/":
                plt.xlabel("x / Mm", fontsize=fs)
                plt.ylabel("y / Mm", fontsize=fs)
            else:
                plt.xlabel("x / m", fontsize=fs)
                plt.ylabel("y / m", fontsize=fs)

            plt.colorbar(c, shrink=shrink_scale)
            
            plt.tick_params(axis='both', labelsize=fs)
            
            # PDF
            plt.savefig(save_plot_to + 'b00_DIFFR_iFTLE-R=' + R_v[k] +\
                                       '-St=' + St_v[j] + '.pdf',
                        format='pdf', dpi=200, bbox_inches='tight')
            
            call(["pdfcrop", save_plot_to + \
                  'b00_DIFFR_iFTLE-R=' + R_v[k] + '-St=' + St_v[j] + '.pdf',
                   save_plot_to + \
                  'b00_DIFFR_iFTLE-R=' + R_v[k] + '-St=' + St_v[j] + '.pdf'])
            
            # EPS
            plt.savefig(save_plot_to + 'b00_DIFFR_iFTLE-R=' + R_v[k] +\
                                       '-St=' + St_v[j] + '.eps',
                        format='eps', dpi=200, bbox_inches='tight')
                
            plt.close(fig)
