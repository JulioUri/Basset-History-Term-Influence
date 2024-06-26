#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 14:43:01 2023

@author: cfg4065
"""
import numpy as np
from numpy import linalg as LA
import scipy
import multiprocessing as mp
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
St_v = np.array(["0.1", "1.0", "10.0"])

# Define vector of strings to chose among files with and without History Term.
Method_v = np.array(["b02_STKS2", "b06_IMEX2"])



##########################################################################
# Define functions that generate FLTE values and print error in parallel #
##########################################################################

def calculateFTLE(elem, limit):
    F  = np.zeros((2,2))
    if (elem % limit == 0 and elem // limit == 0):
        X0      = np.zeros((2,2))
        X0[1,0] = f.get(str(elem + 1))[:][0][0]
        X0[0,0] = f.get(str(elem + 2))[:][0][0]
        X0[1,1] = f.get(str(elem + 1 + limit))[:][0][0]
        
        Xf      = np.zeros((2,2))
        Xf[1,0] = f.get(str(elem + 1))[:][-1][0]
        Xf[0,0] = f.get(str(elem + 2))[:][-1][0]
        Xf[1,1] = f.get(str(elem + 1 + limit))[:][-1][0]
        
        Y0      = np.zeros((2,2))
        Y0[1,0] = f.get(str(elem + 1))[:][0][1]
        Y0[0,0] = f.get(str(elem + 2))[:][0][1]
        Y0[1,1] = f.get(str(elem + 1 + limit))[:][0][1]
        
        Yf      = np.zeros((2,2))
        Yf[1,0] = f.get(str(elem + 1))[:][-1][1]
        Yf[0,0] = f.get(str(elem + 2))[:][-1][1]
        Yf[1,1] = f.get(str(elem + 1 + limit))[:][-1][1]
        
        F[0,0]  = (Xf[1, 1] - Xf[1, 0]) / (X0[1, 1] - X0[1, 0])
        F[0,1]  = (Xf[0, 0] - Xf[1, 0]) / (Y0[0, 0] - Y0[1, 0])
        F[1,0]  = (Yf[1, 1] - Yf[1, 0]) / (X0[1, 1] - X0[1, 0])
        F[1,1]  = (Yf[0, 0] - Yf[1, 0]) / (Y0[0, 0] - Y0[1, 0])
    
    elif (elem % limit == (limit - 1) and elem // limit == 0):
        X0      = np.zeros((2,2))
        X0[0,0] = f.get(str(elem + 1))[:][0][0]
        X0[1,0] = f.get(str(elem))[:][0][0]
        X0[0,1] = f.get(str(elem + 1 + limit))[:][0][0]
        
        Xf      = np.zeros((2,2))
        Xf[0,0] = f.get(str(elem + 1))[:][-1][0]
        Xf[1,0] = f.get(str(elem))[:][-1][0]
        Xf[0,1] = f.get(str(elem + 1 + limit))[:][-1][0]
        
        Y0      = np.zeros((2,2))
        Y0[0,0] = f.get(str(elem + 1))[:][0][1]
        Y0[1,0] = f.get(str(elem))[:][0][1]
        Y0[0,1] = f.get(str(elem + 1 + limit))[:][0][1]
        
        Yf      = np.zeros((2,2))
        Yf[0,0] = f.get(str(elem + 1))[:][-1][1]
        Yf[1,0] = f.get(str(elem))[:][-1][1]
        Yf[0,1] = f.get(str(elem + 1 + limit))[:][-1][1]
        
        F[0,0]  = (Xf[0, 1] - Xf[0, 0]) / (X0[0, 1] - X0[0, 0])
        F[0,1]  = (Xf[1, 0] - Xf[0, 0]) / (Y0[1, 0] - Y0[0, 0])
        F[1,0]  = (Yf[0, 1] - Yf[0, 0]) / (X0[0, 1] - X0[0, 0])
        F[1,1]  = (Yf[1, 0] - Yf[0, 0]) / (Y0[1, 0] - Y0[0, 0])
    
    elif (elem // limit == 0):
        X0      = np.zeros((3,2))
        X0[0,0] = f.get(str(elem + 2))[:][0][0]
        X0[1,0] = f.get(str(elem + 1))[:][0][0]
        X0[1,1] = f.get(str(elem + 1 + limit))[:][0][0]
        X0[2,0] = f.get(str(elem))[:][0][0]
        
        Xf      = np.zeros((3,2))
        Xf[0,0] = f.get(str(elem + 2))[:][-1][0]
        Xf[1,0] = f.get(str(elem + 1))[:][-1][0]
        Xf[1,1] = f.get(str(elem + 1 + limit))[:][-1][0]
        Xf[2,0] = f.get(str(elem))[:][-1][0]
        
        Y0      = np.zeros((3,2))
        Y0[0,0] = f.get(str(elem + 2))[:][0][1]
        Y0[1,0] = f.get(str(elem + 1))[:][0][1]
        Y0[1,1] = f.get(str(elem + 1 + limit))[:][0][1]
        Y0[2,0] = f.get(str(elem))[:][0][1]
        
        Yf      = np.zeros((3,2))
        Yf[0,0] = f.get(str(elem + 2))[:][-1][1]
        Yf[1,0] = f.get(str(elem + 1))[:][-1][1]
        Yf[1,1] = f.get(str(elem + 1 + limit))[:][-1][1]
        Yf[2,0] = f.get(str(elem))[:][-1][1]
        
        F[0,0]  = (Xf[1, 1] - Xf[1, 0]) / (X0[1, 1] - X0[1, 0])
        F[0,1]  = (Xf[0, 0] - Xf[2, 0]) / (Y0[0, 0] - Y0[2, 0])
        F[1,0]  = (Yf[1, 1] - Yf[1, 0]) / (X0[1, 1] - X0[1, 0])
        F[1,1]  = (Yf[0, 0] - Yf[2, 0]) / (Y0[0, 0] - Y0[2, 0])
    
    elif (elem % limit == (limit - 1) and elem // limit == (L - 1)):
        X0      = np.zeros((2,2))
        X0[0,1] = f.get(str(elem + 1))[:][0][0]
        X0[1,1] = f.get(str(elem))[:][0][0]
        X0[0,0] = f.get(str(elem + 1 - limit))[:][0][0]
        
        Xf      = np.zeros((2,2))
        Xf[0,1] = f.get(str(elem + 1))[:][-1][0]
        Xf[1,1] = f.get(str(elem))[:][-1][0]
        Xf[0,0] = f.get(str(elem + 1 - limit))[:][-1][0]
        
        Y0      = np.zeros((2,2))
        Y0[0,1] = f.get(str(elem + 1))[:][0][1]
        Y0[1,1] = f.get(str(elem))[:][0][1]
        Y0[0,0] = f.get(str(elem + 1 - limit))[:][0][1]
        
        Yf      = np.zeros((2,2))
        Yf[0,1] = f.get(str(elem + 1))[:][-1][1]
        Yf[1,1] = f.get(str(elem))[:][-1][1]
        Yf[0,0] = f.get(str(elem + 1 - limit))[:][-1][1]
        
        F[0,0]  = (Xf[0, 0] - Xf[0, 1]) / (X0[0, 0] - X0[0, 1])
        F[0,1]  = (Xf[1, 1] - Xf[0, 1]) / (Y0[1, 1] - Y0[0, 1])
        F[1,0]  = (Yf[0, 0] - Yf[0, 1]) / (X0[0, 0] - X0[0, 1])
        F[1,1]  = (Yf[1, 1] - Yf[0, 1]) / (Y0[1, 1] - Y0[0, 1])
        
    elif (elem % limit == (limit - 1)):
        X0      = np.zeros((2,3))
        X0[0,0] = f.get(str(elem + 1 - limit))[:][0][0]
        X0[0,1] = f.get(str(elem + 1))[:][0][0]
        X0[0,2] = f.get(str(elem + 1 + limit))[:][0][0]
        X0[1,1] = f.get(str(elem))[:][0][0]
        
        Xf      = np.zeros((2,3))
        Xf[0,0] = f.get(str(elem + 1 - limit))[:][-1][0]
        Xf[0,1] = f.get(str(elem + 1))[:][-1][0]
        Xf[0,2] = f.get(str(elem + 1 + limit))[:][-1][0]
        Xf[1,1] = f.get(str(elem))[:][-1][0]
        
        Y0      = np.zeros((2,3))
        Y0[0,0] = f.get(str(elem + 1 - limit))[:][0][1]
        Y0[0,1] = f.get(str(elem + 1))[:][0][1]
        Y0[0,2] = f.get(str(elem + 1 + limit))[:][0][1]
        Y0[1,1] = f.get(str(elem))[:][0][1]
        
        Yf      = np.zeros((2,3))
        Yf[0,0] = f.get(str(elem + 1 - limit))[:][-1][1]
        Yf[0,1] = f.get(str(elem + 1))[:][-1][1]
        Yf[0,2] = f.get(str(elem + 1 + limit))[:][-1][1]
        Yf[1,1] = f.get(str(elem))[:][-1][1]
        
        F[0,0]  = (Xf[0, 2] - Xf[0, 0]) / (X0[0, 2] - X0[0, 0])
        F[0,1]  = (Xf[1, 1] - Xf[0, 0]) / (Y0[1, 1] - Y0[0, 0])
        F[1,0]  = (Yf[0, 2] - Yf[0, 0]) / (X0[0, 2] - X0[0, 0])
        F[1,1]  = (Yf[1, 1] - Yf[0, 0]) / (Y0[1, 1] - Y0[0, 0])
        
    elif (elem % limit == 0 and elem // limit == (L - 1)):
        X0      = np.zeros((2,2))
        X0[1,1] = f.get(str(elem + 1))[:][0][0]
        X0[0,1] = f.get(str(elem + 2))[:][0][0]
        X0[1,0] = f.get(str(elem + 1 - limit))[:][0][0]
        
        Xf      = np.zeros((2,2))
        Xf[1,1] = f.get(str(elem + 1))[:][-1][0]
        Xf[0,1] = f.get(str(elem + 2))[:][-1][0]
        Xf[1,0] = f.get(str(elem + 1 - limit))[:][-1][0]
        
        Y0      = np.zeros((2,2))
        Y0[1,1] = f.get(str(elem + 1))[:][0][1]
        Y0[0,1] = f.get(str(elem + 2))[:][0][1]
        Y0[1,0] = f.get(str(elem + 1 - limit))[:][0][1]
        
        Yf      = np.zeros((2,2))
        Yf[1,1] = f.get(str(elem + 1))[:][-1][1]
        Yf[0,1] = f.get(str(elem + 2))[:][-1][1]
        Yf[1,0] = f.get(str(elem + 1 - limit))[:][-1][1]
        
        F[0,0]  = (Xf[1, 0] - Xf[1, 1]) / (X0[1, 0] - X0[1, 1])
        F[0,1]  = (Xf[0, 1] - Xf[1, 1]) / (Y0[0, 1] - Y0[1, 1])
        F[1,0]  = (Yf[1, 0] - Yf[1, 1]) / (X0[1, 0] - X0[1, 1])
        F[1,1]  = (Yf[0, 1] - Yf[1, 1]) / (Y0[0, 1] - Y0[1, 1])
        
    elif (elem // limit == (L - 1)):
        X0      = np.zeros((3,2))
        X0[0,1] = f.get(str(elem + 2))[:][0][0]
        X0[1,0] = f.get(str(elem + 1 - limit))[:][0][0]
        X0[1,1] = f.get(str(elem + 1))[:][0][0]
        X0[2,1] = f.get(str(elem))[:][0][0]
        
        Xf      = np.zeros((3,2))
        Xf[0,1] = f.get(str(elem + 2))[:][-1][0]
        Xf[1,0] = f.get(str(elem + 1 - limit))[:][-1][0]
        Xf[1,1] = f.get(str(elem + 1))[:][-1][0]
        Xf[2,1] = f.get(str(elem))[:][-1][0]
        
        Y0      = np.zeros((3,2))
        Y0[0,1] = f.get(str(elem + 2))[:][0][1]
        Y0[1,0] = f.get(str(elem + 1 - limit))[:][0][1]
        Y0[1,1] = f.get(str(elem + 1))[:][0][1]
        Y0[2,1] = f.get(str(elem))[:][0][1]
        
        Yf      = np.zeros((3,2))
        Yf[0,1] = f.get(str(elem + 2))[:][-1][1]
        Yf[1,0] = f.get(str(elem + 1 - limit))[:][-1][1]
        Yf[1,1] = f.get(str(elem + 1))[:][-1][1]
        Yf[2,1] = f.get(str(elem))[:][-1][1]
        
        F[0,0]  = (Xf[1, 1] - Xf[1, 0]) / (X0[1, 1] - X0[1, 0])
        F[0,1]  = (Xf[0, 1] - Xf[2, 1]) / (Y0[0, 1] - Y0[2, 1])
        F[1,0]  = (Yf[1, 1] - Yf[1, 0]) / (X0[1, 1] - X0[1, 0])
        F[1,1]  = (Yf[0, 1] - Yf[2, 1]) / (Y0[0, 1] - Y0[2, 1])
        
    elif (elem % limit == 0):
        X0      = np.zeros((2,3))
        X0[0,1] = f.get(str(elem + 2))[:][0][0]
        X0[1,0] = f.get(str(elem + 1 - limit))[:][0][0]
        X0[1,1] = f.get(str(elem + 1))[:][0][0]
        X0[1,2] = f.get(str(elem + 1 + limit))[:][0][0]
        
        Xf      = np.zeros((2,3))
        Xf[0,1] = f.get(str(elem + 2))[:][-1][0]
        Xf[1,0] = f.get(str(elem + 1 - limit))[:][-1][0]
        Xf[1,1] = f.get(str(elem + 1))[:][-1][0]
        Xf[1,2] = f.get(str(elem + 1 + limit))[:][-1][0]
        
        Y0      = np.zeros((2,3))
        Y0[0,1] = f.get(str(elem + 2))[:][0][1]
        Y0[1,0] = f.get(str(elem + 1 - limit))[:][0][1]
        Y0[1,1] = f.get(str(elem + 1))[:][0][1]
        Y0[1,2] = f.get(str(elem + 1 + limit))[:][0][1]
        
        Yf      = np.zeros((2,3))
        Yf[0,1] = f.get(str(elem + 2))[:][-1][1]
        Yf[1,0] = f.get(str(elem + 1 - limit))[:][-1][1]
        Yf[1,1] = f.get(str(elem + 1))[:][-1][1]
        Yf[1,2] = f.get(str(elem + 1 + limit))[:][-1][1]
        
        F[0,0]  = (Xf[1, 2] - Xf[1, 0]) / (X0[1, 2] - X0[1, 0])
        F[0,1]  = (Xf[0, 1] - Xf[1, 1]) / (Y0[0, 1] - Y0[1, 1])
        F[1,0]  = (Yf[1, 2] - Yf[1, 0]) / (X0[1, 2] - X0[1, 0])
        F[1,1]  = (Yf[0, 1] - Yf[1, 1]) / (Y0[0, 1] - Y0[1, 1])
        
    else:
        X0      = np.zeros((3,3))
        X0[0,1] = f.get(str(elem + 2))[:][0][0]
        X0[1,0] = f.get(str(elem + 1 - limit))[:][0][0]
        X0[1,2] = f.get(str(elem + 1 + limit))[:][0][0]
        X0[2,1] = f.get(str(elem))[:][0][0]
        
        Xf      = np.zeros((3,3))
        Xf[0,1] = f.get(str(elem + 2))[:][-1][0]
        Xf[1,0] = f.get(str(elem + 1 - limit))[:][-1][0]
        Xf[1,2] = f.get(str(elem + 1 + limit))[:][-1][0]
        Xf[2,1] = f.get(str(elem))[:][-1][0]
        
        Y0      = np.zeros((3,3))
        Y0[0,1] = f.get(str(elem + 2))[:][0][1]
        Y0[1,0] = f.get(str(elem + 1 - limit))[:][0][1]
        Y0[1,2] = f.get(str(elem + 1 + limit))[:][0][1]
        Y0[2,1] = f.get(str(elem))[:][0][1]
        
        Yf      = np.zeros((3,3))
        Yf[0,1] = f.get(str(elem + 2))[:][-1][1]
        Yf[1,0] = f.get(str(elem + 1 - limit))[:][-1][1]
        Yf[1,2] = f.get(str(elem + 1 + limit))[:][-1][1]
        Yf[2,1] = f.get(str(elem))[:][-1][1]
        
        F[0,0]  = (Xf[1, 2] - Xf[1, 0]) / (X0[1, 2] - X0[1, 0])
        F[0,1]  = (Xf[0, 1] - Xf[2, 1]) / (Y0[0, 1] - Y0[2, 1])
        F[1,0]  = (Yf[1, 2] - Yf[1, 0]) / (X0[1, 2] - X0[1, 0])
        F[1,1]  = (Yf[0, 1] - Yf[2, 1]) / (Y0[0, 1] - Y0[2, 1])
    
    C           = np.transpose(F) @ F
    w, v        = LA.eig(C)
    mu1         = max(w)
    
    return np.log(np.sqrt(mu1))



def printfun(x):
    print(x)



############################################
# Load data from files and calculate FTLEs #
############################################

# Create variables for min and max FTLE values
zmin_Gyre, zmax_Gyre = 10., 1.
zmin_Bkly, zmax_Bkly = 10., 1.
zmin_Frdy, zmax_Frdy = 10., 1.

# Loop over R values
for k in range(0, len(R_folder)):
    
    if R_folder[k] == "/R=0.78":
        print("1.- Calculate FTLEs for R = 0.78.")
    elif R_folder[k] == "/R=1.00":
        print("2.- Calculate FTLEs for R = 1.00")
    elif R_folder[k] == "/R=1.22":
        print("3.- Calculate FTLEs for R = 1.22.")
    
    # Loop over flow fields
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
        
        # Define address to directories
        load_input_from = "../01-Trajectory_data" + R_folder[k] + Flowfield_folder[i]
        save_output_to  = "." + R_folder[k] + Flowfield_folder[i]
        
        # Import initial positions
        x0     = mat['X']
        y0     = mat['Y']
        
        N, L   = np.shape(x0)
        
        limit = N
        
        # Loop over Stokes numbers
        for j in range(0, len(St_v)):
            
            print("          - St = " + str(St_v[j]))
            
            # Loop over integrating methods
            for l in range(0, len(Method_v)):
                
                if Method_v[l] == "b02_STKS2":
                    print("               - Without History Term (solve_ivp solver).")
                elif Method_v[l] == "b06_IMEX2":
                    print("               - With History Term (FD2 + IMEX2).")
                
                # Create empty matrix to plot
                FTLEmap = np.zeros((N, L))
                
                # # Import time values for the FTLE calculation
                # with h5py.File(load_input_from + Method_v[l] + '_TIMEVC-St=' + St_v[j] + '.hdf5', "r") as f:
                #     taxis      = f['default'][()]
                # tau   = 1.0 # taxis[-1] - taxis[0]
                
                result = dict()
                traj_f = dict()
                # Import Trajectory data
                with h5py.File(load_input_from + Method_v[l] + '_TRAJCT-R=' +\
                               R_v[k] + '-St=' + St_v[j] + '.hdf5', "r") as f:
                    Parameters = f["Parameters"][()]
                    n        = len(f.keys()) - 1 # Parameters data-set does not count
                    
                    # Calculate FTLEs using the trajectory data
                    with mp.Pool(36) as pool:
                        for elem in range(0,n):
                            result[str(elem+1)] = pool.apply_async(calculateFTLE, args = (elem, limit), error_callback=printfun)
                            
                        pool.close()
                        pool.join()
                        
                # Fill in empty matrix
                for elem in range(0,n):
                    try: FTLEmap[N-(elem % limit)-1, elem // limit] = result[str(elem+1)].get()
                    except: pass
                
                # Save FTLE data
                with h5py.File(save_output_to + Method_v[l] + '_iFTLE-R=' +\
                               R_v[k] + '-St=' + St_v[j] + '.hdf5', 'w') as f:
                    f.create_dataset("Parameters", data=Parameters)
                    f.create_dataset("X", data=x0)
                    f.create_dataset("Y", data=y0)
                    f.create_dataset("FTLE", data=FTLEmap)
                
                # Save max and min FTLE into a variable to set colorbar
                if Flowfield_folder[i] == "/01-Double_Gyre/":
                    if FTLEmap.min() < zmin_Gyre:
                        zmin_Gyre = FTLEmap.min()
                    
                    if FTLEmap.max() > zmax_Gyre:
                        zmax_Gyre = FTLEmap.max()
                elif Flowfield_folder[i] == "/02-Bickley_Jet/":
                    if FTLEmap.min() < zmin_Bkly:
                        zmin_Bkly = FTLEmap.min()
                    
                    if FTLEmap.max() > zmax_Bkly:
                        zmax_Bkly = FTLEmap.max()
                elif Flowfield_folder[i] == "/03-Faraday_Flow/":
                    if FTLEmap.min() < zmin_Frdy:
                        zmin_Frdy = FTLEmap.min()
                    
                    if FTLEmap.max() > zmax_Frdy:
                        zmax_Frdy = FTLEmap.max()
                
                # Save matrix
                if Method_v[l] == "b02_STKS2":
                    FTLE_Stokes = np.copy(FTLEmap)
                elif Method_v[l] == "b06_IMEX2":
                    FTLE_Full   = np.copy(FTLEmap)
            
            # Calculate Relative difference between solutions
            print("               - Calculating differences in final position.")
            FTLEdiff         = 100. * (FTLE_Full - FTLE_Stokes) / abs(FTLE_Full).max()
            
            # Save Relative difference values
            with h5py.File(save_output_to + 'b00_DIFFR_iFTLE-R=' + R_v[k] +\
                           '-St=' + St_v[j] + '.hdf5', 'w') as f:
                f.create_dataset("Parameters", data=Parameters)
                f.create_dataset("X_0", data=x0)
                f.create_dataset("Y_0", data=y0)
                f.create_dataset("FTLE diff in %", data=FTLEdiff)

# Save max and min FTLE into a file to set colorbar
## For the Double Gyre
np.save("./zmin_Gyre.npy", zmin_Gyre)
np.save("./zmax_Gyre.npy", zmax_Gyre)

## For the Bickley Jet
np.save("./zmin_Bkly.npy", zmin_Bkly)
np.save("./zmax_Bkly.npy", zmax_Bkly)

## For the Faraday flow
np.save("./zmin_Frdy.npy", zmin_Frdy)
np.save("./zmax_Frdy.npy", zmax_Frdy)
