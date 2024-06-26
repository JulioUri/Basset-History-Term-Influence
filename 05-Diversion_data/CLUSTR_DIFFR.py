#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 14:43:01 2023

@author: cfg4065
"""
import numpy as np
import matplotlib.pyplot as plt
from subprocess import call
import scipy
import h5py



###################################
# Define Variables to import data #
###################################

# Decide which value of R is used:
## R=7/9;   R=1;   R=11/9
R_folder = np.array(["/R=0.78", "/R=1.00", "/R=1.22"])

R_v      = np.array(["0.78", "1.0", "1.22"])

# Decide which field flow is used:
## 1: Double Gyre;   3: Bickley Jet;   4: Faraday Flow
Flowfield_folder = np.array(["/01-Double_Gyre/", "/02-Bickley_Jet/", "/03-Faraday_Flow/"])

# Define vector of Stokes numbers that loop over all files.
St_v = np.array([0.1, 1.0, 10.0])



############################################
# Load data from files and calculate FTLEs #
############################################

# Loop over field folders
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
    
    
    # Set dictionaries where arrays are saved to
    Aver_ndif_dic         = dict()
    Aver_adif_dic         = dict()
    Stdv_ndif_dic         = dict()
    max_dic               = dict()
    noHistory_count_dic   = dict()
    fullHistory_count_dic = dict()
    # Loop over R folders
    for k in range(0, len(R_folder)):
        
        if R_v[k] == "0.78":
            print("      - Checking final trajectory points for R = 0.78.\n")
            
        elif R_v[k] == "1.0":
            print("      - Checking final trajectory points for R = 1.00.\n")
            
        elif R_v[k] == "1.22":
            print("      - Checking final trajectory points for R = 1.22.\n")
            
        # Set folder addresses where data is collected and saved
        load_input_from = "../01-Trajectory_data" + R_folder[k] + Flowfield_folder[i]
        save_output_in  = "." + Flowfield_folder[i]
        
        # Import initial positions
        x0     = mat['X']
        y0     = mat['Y']
        
        N, L   = np.shape(x0)
        n      = int(N * L)
        
        Aver_ndif_v         = np.array([])
        Aver_adif_v         = np.array([])
        Stdv_ndif_v         = np.array([])
        max_v               = np.array([])
        noHistory_count_v   = np.array([])
        fullHistory_count_v = np.array([])
        # Loop over Stokes numbers
        for j in range(0, len(St_v)):
            print("          - St = " + str(St_v[j]) + "\n")
            
            noHistory_count     = 0
            fullHistory_count   = 0
            # Loop over all particles
            for elem in range(0, n):
                
                try:
                    # Import Stokes data
                    with h5py.File(load_input_from + 'b02_STKS2_TRAJCT-R=' +\
                           R_v[k] + '-St=' + str(St_v[j]) + '.hdf5', "r") as f:
                    
                        noHistory_fpos = f[str(elem+1)][()][-1]
                        
                        # Check boundary limits for the fields
                        if Flowfield_folder[i] == "/01-Double_Gyre/":
                            if noHistory_fpos[0] < x0.min() or noHistory_fpos[0] > x0.max() or\
                               noHistory_fpos[1] < y0.min() or noHistory_fpos[1] > y0.max():
                                   noHistory_count += 1
                                   
                        elif Flowfield_folder[i] == "/02-Bickley_Jet/":
                            if noHistory_fpos[1] < y0.min() or noHistory_fpos[1] > y0.max():
                                   noHistory_count += 1
                                   
                        elif Flowfield_folder[i] == "/03-Faraday_Flow/":
                            if noHistory_fpos[0] < x0.min() or noHistory_fpos[0] > x0.max() or\
                               noHistory_fpos[1] < y0.min() or noHistory_fpos[1] > y0.max():
                                   noHistory_count += 1
                    
                    # Import Full MRE data
                    with h5py.File(load_input_from + 'b06_IMEX2_TRAJCT-R=' +\
                           R_v[k] + '-St=' + str(St_v[j]) + '.hdf5', "r") as f:
                    
                        fullHistory_inic = f[str(elem+1)][()][0]
                        fullHistory_fpos = f[str(elem+1)][()][-1]
                        
                        # Check boundary limits for the fields
                        if Flowfield_folder[i] == "/01-Double_Gyre/":
                            if fullHistory_fpos[0] < x0.min() or fullHistory_fpos[0] > x0.max() or\
                               fullHistory_fpos[1] < y0.min() or fullHistory_fpos[1] > y0.max():
                                   fullHistory_count += 1
                                   
                        elif Flowfield_folder[i] == "/02-Bickley_Jet/":
                            if fullHistory_fpos[1] < y0.min() or fullHistory_fpos[1] > y0.max():
                                   fullHistory_count += 1
                                   
                        elif Flowfield_folder[i] == "/03-Faraday_Flow/":
                            if fullHistory_fpos[0] < x0.min() or fullHistory_fpos[0] > x0.max() or\
                               fullHistory_fpos[1] < y0.min() or fullHistory_fpos[1] > y0.max():
                                   fullHistory_count += 1
                    
                    # Set arrays where to save data
                    if elem == 0:
                        noHistory_fpos_v   = noHistory_fpos
                        fullHistory_fpos_v = fullHistory_fpos
                        
                        fullHistory_tdis_v = np.linalg.norm(fullHistory_fpos - fullHistory_inic)
                        
                    else:
                        noHistory_fpos_v   = np.vstack((noHistory_fpos_v, noHistory_fpos))
                        fullHistory_fpos_v = np.vstack((fullHistory_fpos_v, fullHistory_fpos))
                        
                        fullHistory_tdis_v = np.vstack((fullHistory_tdis_v,
                                                       np.linalg.norm(fullHistory_fpos - fullHistory_inic)))
                except: pass
            
            # Save data to plot
            noHistory_count_v   = np.append(noHistory_count_v, noHistory_count)
            fullHistory_count_v = np.append(fullHistory_count_v, fullHistory_count)
            
            Diff_fpos_v         = np.linalg.norm(fullHistory_fpos_v - noHistory_fpos_v, axis=1)
            Aver_tdis           = np.average(fullHistory_tdis_v)
            
            Aver_ndif           = np.average(Diff_fpos_v / Aver_tdis)
            Aver_ndif_v         = np.append(Aver_ndif_v, Aver_ndif)
            Aver_adif_v         = np.append(Aver_adif_v, np.average(Diff_fpos_v))
            
            Stdv_ndif           = np.std(Diff_fpos_v / Aver_tdis)
            Stdv_ndif_v         = np.append(Stdv_ndif_v, Stdv_ndif)
            
            Diff_max            = np.linalg.norm(Diff_fpos_v, ord=np.inf) 
            max_v               = np.append(max_v, Diff_max)
        
        # Save data into dictionaries
        Aver_ndif_dic[R_v[k]]         = Aver_ndif_v
        Aver_adif_dic[R_v[k]]         = Aver_adif_v
        Stdv_ndif_dic[R_v[k]]         = Stdv_ndif_v
        max_dic[R_v[k]]               = max_v
        noHistory_count_dic[R_v[k]]   = noHistory_count_v
        fullHistory_count_dic[R_v[k]] = fullHistory_count_v
        
    # Start plotting process
    lw = 1.4
    fs = 6
    ms = 2.2
    cs = 2
    
    # Plot average distances
    fig1 = plt.figure(1, layout='tight', figsize=(2.5, 2.5))
    plt.errorbar(St_v, Aver_ndif_dic[R_v[0]], yerr=Stdv_ndif_dic[R_v[0]],
                 solid_capstyle='projecting', capsize=cs, marker="X",
                 lw=lw, label="R=7/9", markersize=ms)
    # plt.errorbar(St_v, Aver_ndif_dic[R_v[1]], yerr=Stdv_ndif_dic[R_v[1]],
    #               solid_capstyle='projecting', capsize=cs, marker="X",
    #               lw=lw, label="R=1", markersize=ms)
    plt.errorbar(St_v, Aver_ndif_dic[R_v[2]], yerr=Stdv_ndif_dic[R_v[1]],
                 solid_capstyle='projecting', capsize=cs, marker="X",
                 lw=lw, label="R=11/9", markersize=ms)
    plt.xscale('log')
    # plt.yscale('log')
    plt.legend(fontsize=fs)
    plt.ylim(-0.3, 4.8)

    plt.grid()
    plt.xlabel("Stokes number", fontsize=fs)
    plt.ylabel("Average relative deviation at $t_f = 10$", fontsize=fs)
    
    plt.tick_params(axis='both', labelsize=fs)
    
    plt.savefig(save_output_in + 'b08_CLUSTR_DIFAV.pdf',
                format='pdf', dpi=200, bbox_inches='tight')
        
    call(["pdfcrop",
          save_output_in + 'b08_CLUSTR_DIFAV.pdf',
          save_output_in + 'b08_CLUSTR_DIFAV.pdf'])
        
    plt.close(fig1)
    
    # Plot maximum distance
    fig2 = plt.figure(2, layout='tight', figsize=(2.5, 2.5))
    plt.plot(St_v, max_dic[R_v[0]], marker="X", lw=lw, label="R=7/9", markersize=ms)
    # plt.plot(St_v, max_dic[R_v[1]], marker="X", lw=lw, label="R=1", markersize=ms)
    plt.plot(St_v, max_dic[R_v[2]], marker="X", lw=lw, label="R=11/9", markersize=ms)
    plt.xscale('log')
    # plt.yscale('log')
    plt.legend(fontsize=fs)
    plt.grid()
    plt.xlabel("Stokes number", fontsize=fs)
    plt.ylabel("Maximum relative dispersion at $t_f = 10$", fontsize=fs)
    
    plt.tick_params(axis='both', labelsize=fs)
    
    plt.savefig(save_output_in + 'b08_CLUSTR_DIFIN.pdf',
                format='pdf', dpi=200, bbox_inches='tight')
        
    call(["pdfcrop",
          save_output_in + 'b08_CLUSTR_DIFIN.pdf',
          save_output_in + 'b08_CLUSTR_DIFIN.pdf'])
        
    plt.close(fig2)
    
    # Save data into txt file
    with open(save_output_in + 'History_VS_Stokes.txt', 'w') as file:
        file.write("Average and maximum difference in final-time position" +\
                   "calculated with History and without History for the ")
        if Flowfield_folder[i] == "/01_DOUBLE_GYRE/":
            file.write( "Double Gyre ")
        elif Flowfield_folder[i] == "/03_BICKLEY_JET/":
            file.write( "Bickley Jet ")
        elif Flowfield_folder[i] == "/04_FARADAY_FLOW/":
            file.write( "Faraday flow ")
        file.write( "with" )
        for k in range(0, len(R_v)):
            file.write("\n - R = " + R_v[k] + ":\n")
        
            for j in range(0, len(St_v)):
                file.write("\n   - S = " + str(St_v[j]) + ":\n" )
                file.write("      - Relative average distance = " + \
                           str(Aver_ndif_dic[R_v[k]][j]) + ",\n" )
                file.write("      - Relative Standard deviation = " + \
                           str(Stdv_ndif_dic[R_v[k]][j]) + ",\n" )
                file.write("      - Average distance = " + \
                           str(Aver_adif_dic[R_v[k]][j]) + ",\n" )
                file.write("      - Maximum distance = " + \
                           str(max_dic[R_v[k]][j]) + ".\n" )
                file.write("      - Number of particles that left the domain: \n" +\
                           "        * With History: " + str(fullHistory_count_dic[R_v[k]][j]) + "\n" +\
                           "        * Without History: " + str(noHistory_count_dic[R_v[k]][j]) + "\n")

