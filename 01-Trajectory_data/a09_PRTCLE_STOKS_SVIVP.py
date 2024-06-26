from a00_PMTERS_CONST import mr_parameter
import numpy as np
from scipy import integrate

'''
There are two classes defined below.

These two classes calculate the trajectory and velocity of particles
either with Stokes drag or as a Perfect Tracer only (Material derivative only)
by using the Leapfrog method (further details in PDF document developed by Daniel).

Function "__init__" defines all the variables associated to the particle.

Function "update" initiates the update in the particle's velocity and trajectory
for a given time step.

Functions "update_f_tile", "update_f" and "leapfrog" calculate all the necessary
terms in the Leapfrog method. Further details are provided in the PDF document
developed by Daniel.
'''

'''
###############################################################################
#########################PARTICLE WITH STOKES DRAG#############################
###############################################################################
'''
class particle_stokes(object):

  def __init__(self, tag, x, v, tini, velocity_field, particle_density=1, fluid_density=1, \
               particle_radius=1, kinematic_viscosity=1, time_scale=1):
      self.tag     = tag    # particle name/tag
      self.x       = np.copy(x)        # particle position
      self.v       = np.copy(v)        # particle velocity
      self.p       = mr_parameter(particle_density, fluid_density,
                                  particle_radius, kinematic_viscosity,
                                  time_scale)
      self.vel     = velocity_field
      self.time    = tini
      self.pos_vec = np.copy(x)
      self.vel_vec = np.copy(v)
      
      if self.vel.limits == True:
          if (self.x[0] > self.vel.x_right or self.x[0] < self.vel.x_left or self.x[1] > self.vel.y_up or self.x[1] < self.vel.y_down):
              raise Exception("Particle's initial position is outside the spatial domain")

  def function(self, t, x):

      if self.vel.periodic == True:

          if x[0] > self.vel.x_right:
              x[0] -= self.vel.dx
          elif x[0] < self.vel.x_left:
              x[0] += self.vel.dx

          if x[1] > self.vel.y_up:
              x[1] -= self.vel.dy
          elif x[1] < self.vel.y_down:
              x[1] += self.vel.dy

      u, v           = self.vel.get_velocity(x[0], x[1], t)
      ux, uy, vx, vy = self.vel.get_gradient(x[0], x[1], t)
      u_gradu        = np.array([ u*ux + v*uy, u*vx + v*vy])
      du_dt          = self.vel.get_dudt(x[0], x[1], t)
      St_drag        = (np.array([x[2], x[3]]) - np.array([u, v])) / (self.p.R * self.p.S) 
      f              = (1.0/self.p.R)*( du_dt + u_gradu) - St_drag
      
      output    = np.zeros(len(x))
      
      output[0] = x[2]
      output[1] = x[3]
      output[2] = f[0]
      output[3] = f[1]
      
      return output

  def update(self, dt):
      tarray = np.array([self.time, self.time + dt])
      if self.time == 0:
          x0 = np.copy(self.pos_vec)
          v0 = np.copy(self.vel_vec)
          y0 = np.append(x0, v0)
      else:
          x0 = np.copy(self.pos_vec[-1])
          v0 = np.copy(self.vel_vec[-1])
          y0 = np.append(x0, v0)
      vec = integrate.solve_ivp(self.function, tarray, y0, rtol=1e-8, atol=1e-8)
      
      x = np.array([vec.y[0][-1], vec.y[1][-1]])
      
      if self.vel.periodic == True:
          if x[0] > self.vel.x_right:
              x[0] -= self.vel.dx
          elif x[0] < self.vel.x_left:
              x[0] += self.vel.dx
        		
          if x[1] > self.vel.y_up:
              x[1] -= self.vel.dy
          elif x[1] < self.vel.y_down:
       	      x[1] += self.vel.dy

      v = np.array([vec.y[2][-1], vec.y[3][-1]])
      
      self.x = np.copy(x)
      self.v = np.copy(v)
      
      self.pos_vec = np.vstack([self.pos_vec, self.x])
      self.vel_vec = np.vstack([self.vel_vec, self.v])
      
      self.time += dt
      
      return self.x, self.v
