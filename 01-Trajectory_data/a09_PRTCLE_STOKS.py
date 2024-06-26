from a00_PMTERS_CONST import mr_parameter
import numpy as np

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

  def update(self,dt):
      '''
      The Leapfrog integrator requires a value for the velocity at time -dt. To compute it,
      the velocity field is set to the value at t=0, which sets the partial derivative with respect to time
      of the velocity field to zero in the first step.
      '''
      self.u         = self.vel.get_velocity(self.x[0], self.x[1], self.time)
      #if self.u_old is None:
      #    self.u_old = self.vel.get_velocity(self.x[0], self.x[1], self.time)
      #    # Note: because u = u_old, the value of dt has no impact and is set to 1.0
      self.update_f_tilde(dt)
      self.update_f(dt)
      self.leapfrog(dt)
      
      return self.x, self.v
    
  '''
  Update forces to value at current particle position
  '''
  def update_f_tilde(self, dt):
      u, v           = self.vel.get_velocity(self.x[0], self.x[1], self.time)
      ux, uy, vx, vy = self.vel.get_gradient(self.x[0], self.x[1], self.time)
      u_gradu        = np.array([ u*ux + v*uy, u*vx + v*vy])
      du_dt          = self.vel.get_dudt(self.x[0], self.x[1], self.time)
      self.f_tilde   = (1.0/self.p.R)*( du_dt + u_gradu) + \
                          (1.0/(self.p.S*self.p.R))*np.array([u,v])
      #self.u_old     = np.array([u,v])
  
  def update_f(self, dt):
      self.f         = self.f_tilde - 1.0/(self.p.S*self.p.R)*self.v
    
  '''
  Computes an update of particle position and velocity using a Leapfrog scheme.
  '''
  def leapfrog(self, dt):
      #First half-step to compute intermediate velocity
      v_half         = self.v + 0.5*dt*self.f
      
      #Position update step
      x              = self.x + dt*v_half
      
      if self.vel.periodic == True:
          
          if x[0] > self.vel.x_right:
              x[0] -= self.vel.dx
          elif x[0] < self.vel.x_left:
              x[0] += self.vel.dx

          if x[1] > self.vel.y_up:
              x[1] -= self.vel.dy
          elif x[1] < self.vel.y_down:
              x[1] += self.vel.dy

      self.x         = x
      self.pos_vec   = np.vstack([self.pos_vec,self.x])

      # Update forces acting on particle
      self.time     += dt
      self.update_f_tilde(dt)
  
      # Solve for v_n+1
      self.v         = (1.0/(1.0+dt/(2*self.p.S*self.p.R)))*\
                          (v_half + 0.5*dt*self.f_tilde)
      self.vel_vec   = np.vstack([self.vel_vec,self.v])
      
      #Check we are still within the bounds for which we have information
      if self.vel.limits == True:
          if (self.x[0] > self.vel.x_right or self.x[0] < self.vel.x_left or self.x[1] > self.vel.y_up or self.x[1] < self.vel.y_down):
              raise Exception("Particle's position exits the spatial domain")
