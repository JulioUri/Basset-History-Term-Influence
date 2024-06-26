import numpy as np

'''
Created on 2020
@author: Daniel Ruprecht

This file contains the formulas needed to calculate the parameters of the
Maxey-Riley equation that are needed to run the SOLVER files.
'''

class mr_parameter(object):

  def set_beta(self):
    self.beta = self.rho_p /self.rho_f
  
  def set_S(self):
    self.S = (1.0/3.0)*self.a**2/(self.nu * self.T)
  
  def set_R(self):
    self.R = (1.0 + 2.0*self.beta)/3.0
    
  def set_alpha(self):
    self.alpha = 1 / (self.R * self.S) 

  def set_gamma(self):
    self.gamma = (1 / self.R) * np.sqrt(3 / self.S)
  
  def __init__(self, particle_density, fluid_density, particle_radius,
                     kinematic_viscosity, time_scale):
    self.rho_p = particle_density
    self.rho_f = fluid_density
    self.a     = particle_radius
    self.nu    = kinematic_viscosity
    self.T     = time_scale

    self.set_beta()
    self.set_S()
    self.set_R()
    self.set_alpha()
    self.set_gamma()