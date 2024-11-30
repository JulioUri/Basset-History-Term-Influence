import numpy as np
import scipy.sparse as sp
from a00_PMTERS_CONST import mr_parameter

'''
There are four classes defined below.

The first two classes calculate the trajectory and velocity of particles whose
dynamics is governed by the FULL MAXEY-RILEY by using either an approach by a
direct Numerical Integration given in the paper A. Daitche (2013). 
'''

'''
###############################################################################
####################FULL MAXEY RILEY (Direct Integration)######################
###############################################################################
'''
class maxey_riley_Daitche(object):
    
  def __init__(self, tag, x, v, velocity_field, Nt, order, particle_density,
               fluid_density, particle_radius, kinematic_viscosity,
               time_scale):
      
      self.tag     = tag              # particle name/tag
      self.v       = np.copy(v)       # particle velocity
      self.x       = np.copy(x)       # particle position
      
      self.p       = mr_parameter(particle_density, fluid_density,
                            particle_radius, kinematic_viscosity, time_scale)
      
      self.vel     = velocity_field
      
      if self.vel.limits == True:
          if (x[0] > self.vel.x_right or x[0] < self.vel.x_left or x[1] > self.vel.y_up or x[1] < self.vel.y_down):
              raise Exception("Particle's initial position is outside the spatial domain")
    
      if order == 1:
          self.calc_alpha_mat(Nt)
      elif order == 2:
          self.euler_nodes  = 21 # This number could be increased to increase accuracy
          self.calc_alpha_mat(self.euler_nodes)
          self.calc_beta_mat(Nt)
      elif order == 3:
          self.euler_nodes  = 21 # This number could be increased to increase accuracy
          self.calc_alpha_mat(self.euler_nodes)
          self.calc_beta_mat(self.euler_nodes)
          self.calc_gamma_mat(Nt)
      else:
          raise("Requested order for Daitche's method not available.")



  # Define f(q(0,t))    # This will differ for different boundary problems
  def calculate_f(self, q, p, x, y, t):
      
      coeff          = (1.0 / self.p.R) - 1.0
      
      u, v           = self.vel.get_velocity(x, y, t)
      ux, uy, vx, vy = self.vel.get_gradient(x, y, t)
      
      ut, vt         = self.vel.get_dudt(x, y, t)
      
      f              = coeff * ut + (coeff * u - q) * ux + \
                                    (coeff * v - p) * uy
      g              = coeff * vt + (coeff * u - q) * vx + \
                                    (coeff * v - p) * vy  
      return f, g
    
    

  def alpha_jn(self, j, n):
      if n == 0:
          alpha   = 0.0
      elif j == 0:
          alpha   = 4.0 / 3.0
      elif j != n:
          alpha   = ((j - 1.0)**1.5 + (j + 1.0)**1.5 - 2.0 * j**1.5) * 4.0 / 3.0
      else:
          alpha   = ((n - 1.0)**1.5 - n**1.5 + np.sqrt(n)*3.0/2.0) * 4.0/3.0
      return alpha

  
    
  def alpha_v(self, N):
      
      AlphaSubst_v = np.array([])
      for j in range(0, N+1):
          AlphaSubst_v = np.append(AlphaSubst_v,
                            self.alpha_jn(j+1, N+1) -\
                                self.alpha_jn(j, N))
      
      return AlphaSubst_v
  
    
  def calc_alpha_mat(self, N):
      for nn in range(0, N-1):
          if nn == 0:
              alpha_mat     = sp.csr_matrix(self.alpha_v(nn),
                                            shape=(1, N))
          else:
              alpha_v       = sp.csr_matrix(self.alpha_v(nn),
                                            shape=(1, N))
              alpha_mat     = sp.vstack([alpha_mat, alpha_v])
      self.alpha_mat = alpha_mat
      
    
  def Euler(self, t_v, flag=False):
      assert len(t_v) >= 2, "This time grid cannot be used for time stepping."
      
      h     = t_v[1] - t_v[0]
      
      x_v   = np.array([self.x[0]])
      y_v   = np.array([self.x[1]])
      
      u0, v0 = self.vel.get_velocity(self.x[0], self.x[1], t_v[0])
      q_v   = np.array([self.v[0] - u0])
      p_v   = np.array([self.v[1] - v0])
      
      xi    = self.p.gamma * np.sqrt(h / np.pi)
      
      for nn in range(0, len(t_v)-1):
          
          Sum_q     = np.dot(self.alpha_mat.toarray()[nn, :nn+1], q_v)
          Sum_p     = np.dot(self.alpha_mat.toarray()[nn, :nn+1], p_v)
          
          if nn == 0:
              x_np1     = x_v[nn] + h * (q_v[0] + u0)
              y_np1     = y_v[nn] + h * (p_v[0] + v0)
          else:
              u_n, v_n  = self.vel.get_velocity(x_v[nn], y_v[nn], t_v[nn])
          
              x_np1     = x_v[nn] + h * (q_v[0] + u_n)
              y_np1     = y_v[nn] + h * (p_v[0] + v_n)
      
          f, g      = self.calculate_f(q_v[0], p_v[0],
                                      x_v[nn], y_v[nn],
                                      t_v[nn])
          
          Gq_n      = f - self.p.alpha * q_v[0]
          Gp_n      = g - self.p.alpha * p_v[0]
          
          q_np1     = ( q_v[0] + h * Gq_n - xi * Sum_q ) / \
                          (1.0 + xi * self.alpha_jn(0, nn+1))
          p_np1     = ( p_v[0] + h * Gp_n - xi * Sum_p ) / \
                          (1.0 + xi * self.alpha_jn(0, nn+1))
          
          if self.vel.periodic == True:
          
              try:
                  if x_np1 >= self.vel.x_right:
                      x_np1 -= self.vel.dx
                  elif x_np1 < self.vel.x_left:
                      x_np1 += self.vel.dx
              except:
                  pass
    
              try:
                  if y_np1 >= self.vel.y_up:
                       y_np1 -= self.vel.dy
                  elif y_np1 < self.vel.y_down:
                       y_np1 += self.vel.dy  
              except:
                  pass

          x_v       = np.append(x_v, x_np1)
          y_v       = np.append(y_v, y_np1)
          q_v       = np.append(q_np1, q_v)
          p_v       = np.append(p_np1, p_v)
          
      pos_vec      = np.transpose(np.array([x_v, y_v]))
      q0_vec       = np.transpose(np.array([np.flip(q_v), np.flip(p_v)]))
      
      if flag == True:
          self.pos_vec = np.copy(pos_vec)
          self.q0_vec  = np.copy(q0_vec)
      
      return pos_vec, q0_vec
  
    
    
  def beta_jn(self, j, n): 
      
      if n == 0:
          beta = 0.0
          
      elif n == 1:
          beta = self.alpha_jn(j, n)
          
      elif n == 2:
          if j == 0:
              beta = 12.0 * np.sqrt(2.0) / 15.0
          elif j == 1:
              beta = 16.0 * np.sqrt(2.0) / 15.0
          elif j == 2:
              beta = 2.0 * np.sqrt(2.0) / 15.0
          
      elif n == 3:
          if j == 0:
              beta = 4.0 * np.sqrt(2.0) / 5.0
          elif j == 1:
              beta = (14.0 * np.sqrt(3.0) - 12.0 * np.sqrt(2.0)) / 5.0
          elif j == 2:
              beta = (12.0 * np.sqrt(2) - 8.0 * np.sqrt(3.0)) / 5.0
          elif j == 3:
              beta = (np.sqrt(3.0) - np.sqrt(2.0)) * (4.0 / 5.0)
            
      else:
          if j == 0:
              beta = 4.0 * np.sqrt(2.0) / 5.0
          elif j == 1:
              beta = (14.0 * np.sqrt(3.0) - 12.0 * np.sqrt(2.0)) / 5.0
          elif j == 2:
              beta = (176.0/15.0) + ( 12.0 * np.sqrt(2.0) - 42.0 * np.sqrt(3.0)) / 5.0
          elif j == n-1:
              beta = (8.0/15.0) * (-2.0 * n**(5.0/2.0) + \
                            3.0 * (n - 1.0)**(5.0/2.0) - \
                                  (n - 2.0)**(5.0/2.0)) + \
                       (2.0/3.0) * (4.0 * n**(3.0/2.0) - \
                            3.0 * (n - 1.0)**(3.0/2.0) + \
                                  (n - 2.0)**(3.0/2.0))
          elif j == n:
              beta = (8.0/15.0) * (       n**(5.0/2.0) - \
                                  (n - 1.0)**(5.0/2.0)) + \
                      (2.0/3.0) * (-3.0 * n**(3.0/2.0) + \
                                  (n - 1.0)**(3.0/2.0)) + \
                                  2.0 * np.sqrt(n)
          else:
              beta = (8.0/15.0) * ((j + 2.0)**(5.0/2.0) - \
                             3.0 * (j + 1.0)**(5.0/2.0) + \
                                     3.0 * j**(5.0/2.0) - \
                                   (j - 1.0)**(5.0/2.0)) + \
                     (2.0/3.0) * (-(j + 2.0)**(3.0/2.0) + \
                             3.0 * (j + 1.0)**(3.0/2.0) - \
                                     3.0 * j**(3.0/2.0) + \
                                   (j - 1.0)**(3.0/2.0))
          
      return beta

  
    
  def beta_v(self, N):
      
      BetaSubst_v = np.array([])
      for nn in range(0, N+1):
          BetaSubst_v = np.append(BetaSubst_v,
                            self.beta_jn(nn+1, N+1) -\
                                self.beta_jn(nn, N))
      
      return BetaSubst_v
  
    

  def calc_beta_mat(self, N):
      for nn in range(0, N-1):
          if nn == 0:
              beta_mat     = sp.csr_matrix(self.beta_v(nn), shape=(1, N))
          else:
              beta_v       = sp.csr_matrix(self.beta_v(nn), shape=(1, N))
              beta_mat     = sp.vstack([beta_mat, beta_v])
      self.beta_mat = beta_mat



  def AdamBashf2(self, t_v, flag=False):
      assert len(t_v) >= 2, "Time grid cannot be used for time stepping."
      
      h          = t_v[1] - t_v[0]
      
      # Obtaining first time step with a 1st order method
      t_np1           = np.linspace(t_v[0], t_v[1], self.euler_nodes)
      pos_vec, q0_vec = self.Euler(t_np1, flag=False)
      
      # Calculating the rest of the solution with a 2nd order method
      
      x_v   = np.array([self.x[0], pos_vec[-1,0]])
      y_v   = np.array([self.x[1], pos_vec[-1,1]])
      
      u0, v0  = self.vel.get_velocity(self.x[0], self.x[1], t_v[0])
      
      q_v   = np.array([q0_vec[-1,0], self.v[0] - u0])
      p_v   = np.array([q0_vec[-1,1], self.v[1] - v0])
     
      xi    = self.p.gamma * np.sqrt(h / np.pi)
     
      for nn in range(1, len(t_v)-1):
          Sum_q     = np.dot(self.beta_mat.toarray()[nn, :nn+1], q_v)
          Sum_p     = np.dot(self.beta_mat.toarray()[nn, :nn+1], p_v)
          
          u_n, v_n     = self.vel.get_velocity(x_v[nn], y_v[nn], t_v[nn])
          u_nm1, v_nm1 = self.vel.get_velocity(x_v[nn-1], y_v[nn-1], t_v[nn-1])
          
          x_np1     = x_v[nn] + (h/2.0) * ( 3.0 * (q_v[0] + u_n) - (q_v[1] + u_nm1))
          y_np1     = y_v[nn] + (h/2.0) * ( 3.0 * (p_v[0] + v_n) - (p_v[1] + v_nm1))
      
          f_n, g_n  = self.calculate_f(q_v[0], p_v[0],
                                      x_v[nn], y_v[nn],
                                      t_v[nn])
          f_nm1, g_nm1 = self.calculate_f(q_v[1], p_v[1],
                                      x_v[nn-1], y_v[nn-1],
                                      t_v[nn-1])
          
          
          Gq_n      = f_n - self.p.alpha * q_v[0]
          Gp_n      = g_n - self.p.alpha * p_v[0]
          
          Gq_nm1    = f_nm1 - self.p.alpha * q_v[1]
          Gp_nm1    = g_nm1 - self.p.alpha * p_v[1]
          
          q_np1     = ( q_v[0] + (h/2.0) * (3.0 * Gq_n - Gq_nm1) - xi * Sum_q ) / \
                          (1.0 + xi * self.beta_jn(0, nn+1))
          p_np1     = ( p_v[0] + (h/2.0) * (3.0 * Gp_n - Gp_nm1) - xi * Sum_p ) / \
                          (1.0 + xi * self.beta_jn(0, nn+1))
          
          if self.vel.periodic == True:
          
              try:
                  if x_np1 >= self.vel.x_right:
                      x_np1 -= self.vel.dx
                  elif x_np1 < self.vel.x_left:
                      x_np1 += self.vel.dx
              except:
                  pass
              
              try:
                  if y_np1 >= self.vel.y_up:
                      y_np1 -= self.vel.dy
                  elif y_np1 < self.vel.y_down:
                      y_np1 += self.vel.dy
              except:
                  pass
          
          x_v       = np.append(x_v, x_np1)
          y_v       = np.append(y_v, y_np1)
          q_v       = np.append(q_np1, q_v)
          p_v       = np.append(p_np1, p_v)
          
      pos_vec      = np.transpose(np.array([x_v, y_v]))
      q0_vec       = np.transpose(np.array([np.flip(q_v), np.flip(p_v)]))
      
      if flag == True:
          self.pos_vec = np.copy(pos_vec)
          self.q0_vec  = np.copy(q0_vec)
      
      return pos_vec, q0_vec



  def gamma_jn(self, j, n): 

      if n == 0:
          gamma = 0.0
          
      elif n == 1:
          gamma = self.alpha_jn(j, n)
          
      elif n == 2:
          gamma = self.beta_jn(j, n)
          
      elif n == 3:
          if j == 0:
              gamma = (68.0/105.0) * np.sqrt(3.0)
          elif j == 1:
              gamma = (6.0/7.0) * np.sqrt(3.0)
          elif j == 2:
              gamma = (12.0/35.0) * np.sqrt(3.0)
          elif j == 3:
              gamma = (16.0/105.0) * np.sqrt(3.0)
          
      elif n == 4:
          if j == 0:
              gamma = (244.0/315.0) * np.sqrt(2.0)
          elif j == 1:
              gamma = (1888.0 - 976.0 * np.sqrt(2.0)) / 315.0
          elif j == 2:
              gamma = (488.0 * np.sqrt(2.0) - 656.0 ) / 105.0
          elif j == 3:
              gamma = (544.0/105.0) - (976.0/315.0) * np.sqrt(2.0)
          elif j == 4:
              gamma = (244.0 * np.sqrt(2.0) - 292.0 ) / 315.0
          
      elif n == 5:
          if j == 0:
              gamma = (244.0/315.0) * np.sqrt(2.0)
          elif j == 1:
              gamma = (362.0/105.0) * np.sqrt(3.0) - (976.0/315.0) * np.sqrt(2.0)
          elif j == 2:
              gamma = (500.0/63.0) * np.sqrt(5.0) - \
                          (1448.0/105.0) * np.sqrt(3.0) + \
                              (488.0/105.0) * np.sqrt(2.0)
          elif j == 3:
              gamma = (-290.0/21.0) * np.sqrt(5.0) + \
                          (724.0/35.0) * np.sqrt(3.0) - \
                              (976.0/315.0) * np.sqrt(2.0)
          elif j == 4:
              gamma = (220.0/21.0) * np.sqrt(5.0) - \
                          (1448.0/105.0) * np.sqrt(3.0) + \
                              (244.0/315.0) * np.sqrt(2.0)
          elif j == 5:
              gamma = (362.0/105.0) * np.sqrt(3.0) - \
                          (164.0/63.0) * np.sqrt(5.0)
          
      elif n == 6:
          if j == 0:
              gamma = (244.0/315.0) * np.sqrt(2.0)
          elif j == 1:
              gamma = (362.0/105.0) * np.sqrt(3.0) - \
                          (976.0/315.0) * np.sqrt(2.0)
          elif j == 2:
              gamma = (5584.0/315.0) - \
                          (1448.0/105.0) * np.sqrt(3.0) + \
                              (488.0/105.0) * np.sqrt(2.0)
          elif j == 3:
              gamma = (344.0/21.0) * np.sqrt(6.0) - \
                          (22336.0/315.0) + (724.0/35.0) * np.sqrt(3.0) - \
                               (976.0/315.0) * np.sqrt(2.0)
          elif j == 4:
              gamma = (-1188.0/35.0) * np.sqrt(6.0) + \
                          (11168.0/105.0) - (1448.0/105.0) * np.sqrt(3.0) + \
                              (244.0/315.0) * np.sqrt(2.0)
          elif j == 5:
              gamma = (936.0/35.0) * np.sqrt(6.0) - \
                          (22336.0/315.0) + (362.0/105.0) * np.sqrt(3.0)
          elif j == 6:
              gamma = (5584.0/315.0) - (754.0/105.0) * np.sqrt(6.0)
          
      else:
          if j == 0:
              gamma = 244.0 * np.sqrt(2.0) / 315.0
              
          elif j == 1:
              gamma = (362.0/105.0) * np.sqrt(3.0) - \
                          (976.0/315.0) * np.sqrt(2.0)
              
          elif j == 2:
              gamma = (5584.0/315.0) - (1448.0/105.0) * np.sqrt(3.0) + \
                          (488.0/105.0) * np.sqrt(2.0)
              
          elif j == 3:
              gamma = (1130.0/63.0) * np.sqrt(5.0) - \
                          (22336.0/315.0) + (724.0/35.0) * np.sqrt(3.0) - \
                              (976.0/315.0) * np.sqrt(2.0)
              
          elif j == n-3:
              gamma = (16.0/105.0) * (n**(7.0/2.0) - \
                           4.0 * (n - 2.0)**(7.0/2.0) + \
                           6.0 * (n - 3.0)**(7.0/2.0) - \
                           4.0 * (n - 4.0)**(7.0/2.0) + \
                                 (n - 5.0)**(7.0/2.0)) - \
                            (8.0/15.0) * n**(5.0/2.0) + \
                             (4.0/9.0) * n**(3.0/2.0) + \
                     (8.0/9.0) * (n - 2.0)**(3.0/2.0) - \
                     (4.0/3.0) * (n - 3.0)**(3.0/2.0) + \
                     (8.0/9.0) * (n - 4.0)**(3.0/2.0) - \
                     (2.0/9.0) * (n - 5.0)**(3.0/2.0)
          elif j == n-2:
              gamma = (16.0/105.0) * ((n - 4.0)**(7.0/2.0) - \
                                4.0 * (n - 3.0)**(7.0/2.0) + \
                                6.0 * (n - 2.0)**(7.0/2.0) - \
                                       3.0 * n ** (7.0/2.0)) + \
                                (32.0/15.0) * n**(5.0/2.0) - \
                                        2.0 * n**(3.0/2.0) - \
                          (4.0/3.0) * (n - 2.0)**(3.0/2.0) + \
                          (8.0/9.0) * (n - 3.0)**(3.0/2.0) - \
                          (2.0/9.0) * (n - 4.0)**(3.0/2.0)
          elif j == n-1:
              gamma = (16.0/105.0) * ( 3.0 * n**(7.0/2.0) - \
                               4.0 * (n - 2.0)**(7.0/2.0) + \
                                     (n - 3.0)**(7.0/2.0)) - \
                                 (8.0/3.0) * n**(5.0/2.0) + \
                                      4.0 * n **(3.0/2.0) + \
                         (8.0/9.0) * (n - 2.0)**(3.0/2.0) - \
                         (2.0/9.0) * (n - 3.0)**(3.0/2.0)
          elif j == n:
              gamma = (16.0/105.0) * ((n - 2.0)**(7.0/2.0) - \
                                              n**(7.0/2.0)) + \
                                (16.0/15.0) * n**(5.0/2.0) - \
                                 (22.0/9.0) * n**(3.0/2.0) - \
                          (2.0/9.0) * (n - 2.0)**(3.0/2.0) + \
                                 2.0 * np.sqrt(n)
          else:
              gamma = (16.0/105.0) * ((j + 2.0)**(7.0/2.0) + \
                                      (j - 2.0)**(7.0/2.0) - \
                                4.0 * (j + 1.0)**(7.0/2.0) - \
                                4.0 * (j - 1.0)**(7.0/2.0) + \
                                        6.0 * j**(7.0/2.0)) + \
                  (2.0/9.0) * (4.0 * (j + 1.0)**(3.0/2.0) + \
                               4.0 * (j - 1.0)**(3.0/2.0) - \
                                     (j + 2.0)**(3.0/2.0) - \
                                     (j - 2.0)**(3.0/2.0) - \
                                       6.0 * j**(3.0/2.0))
      return gamma

  
    
  def gamma_v(self, N):
      
      GammaSubst_v = np.array([])
      for nn in range(0, N+1):
          GammaSubst_v = np.append(GammaSubst_v, self.gamma_jn(nn+1, N+1) -\
                                                 self.gamma_jn(nn, N))
      
      return GammaSubst_v
  


  def calc_gamma_mat(self, N):
      for nn in range(0, N-1):
          if nn == 0:
              gamma_mat    = sp.csr_matrix(self.gamma_v(nn), shape=(1, N))
          else:
              gamma_v      = sp.csr_matrix(self.gamma_v(nn), shape=(1, N))
              gamma_mat    = sp.vstack([gamma_mat, gamma_v])
      self.gamma_mat = gamma_mat



  def AdamBashf3(self, t_v, flag=False):
      assert len(t_v) >= 3, "Time grid cannot be used for time stepping."
      
      h               = t_v[1] - t_v[0]
      
      # Obtaining first and second time steps with a 1st order method
      assert self.euler_nodes % 2 == 1, "Please provide an odd number of nodes for the Euler computation of the first steps."
      t_np1           = np.linspace(t_v[0], t_v[2], self.euler_nodes)
      # pos_vec, q0_vec = self.Euler(t_np1, flag=False)
      pos_vec, q0_vec = self.AdamBashf2(t_np1, flag=False)
      
      # Calculating the rest of the solution with a 3rd order method
      x_v      = np.array([self.x[0], pos_vec[int((self.euler_nodes-1)/2),0], pos_vec[-1,0]])
      y_v      = np.array([self.x[1], pos_vec[int((self.euler_nodes-1)/2),1], pos_vec[-1,1]])
      
      u0, v0   = self.vel.get_velocity(self.x[0], self.x[1], t_v[0])
     
      q_v      = np.array([q0_vec[-1,0], q0_vec[int((self.euler_nodes-1)/2),0], self.v[0] - u0])
      p_v      = np.array([q0_vec[-1,1], q0_vec[int((self.euler_nodes-1)/2),1], self.v[1] - v0])
      
      for nn in range(2, len(t_v)-1):
          Sum_q        = np.dot(self.gamma_mat.toarray()[nn, :nn+1], q_v)
          Sum_p        = np.dot(self.gamma_mat.toarray()[nn, :nn+1], p_v)
          
          u_n, v_n     = self.vel.get_velocity(x_v[nn], y_v[nn], t_v[nn])
          u_nm1, v_nm1 = self.vel.get_velocity(x_v[nn-1], y_v[nn-1], t_v[nn-1])
          u_nm2, v_nm2 = self.vel.get_velocity(x_v[nn-2], y_v[nn-2], t_v[nn-2])
          
          x_np1        = x_v[nn] + (h/12.0) * ( 23.0 * (q_v[0] + u_n  ) -\
                                                16.0 * (q_v[1] + u_nm1) +\
                                                 5.0 * (q_v[2] + u_nm2))
          y_np1        = y_v[nn] + (h/12.0) * ( 23.0 * (p_v[0] + v_n) -\
                                                16.0 * (p_v[1] + v_nm1) +\
                                                 5.0 * (p_v[2] + v_nm2))
          
          xi           = self.p.gamma * np.sqrt(h / np.pi)
      
          f_n, g_n     = self.calculate_f(q_v[0],  p_v[0],
                                          x_v[nn], y_v[nn],
                                          t_v[nn])
          f_nm1, g_nm1 = self.calculate_f(q_v[1],    p_v[1],
                                          x_v[nn-1], y_v[nn-1],
                                          t_v[nn-1])
          f_nm2, g_nm2 = self.calculate_f(q_v[2],    p_v[2],
                                          x_v[nn-2], y_v[nn-2],
                                          t_v[nn-2])
          
          
          Gq_n         = f_n   - self.p.alpha * q_v[0]
          Gp_n         = g_n   - self.p.alpha * p_v[0]
          
          Gq_nm1       = f_nm1 - self.p.alpha * q_v[1]
          Gp_nm1       = g_nm1 - self.p.alpha * p_v[1]
          
          Gq_nm2       = f_nm2 - self.p.alpha * q_v[2]
          Gp_nm2       = g_nm2 - self.p.alpha * p_v[2]
          
          
          q_np1        = ( q_v[0] + (h/12.0) * (23.0 * Gq_n - \
                               16.0 * Gq_nm1 + 5.0 * Gq_nm2) - \
                                  xi * Sum_q ) / \
                          (1.0 + xi * self.gamma_jn(0, nn+1))
          p_np1        = ( p_v[0] + (h/12.0) * (23.0 * Gp_n - \
                               16.0 * Gp_nm1 + 5.0 * Gp_nm2) - \
                                  xi * Sum_p ) / \
                          (1.0 + xi * self.gamma_jn(0, nn+1))

          if self.vel.periodic == True:
          
              try:
                  if x_np1 >= self.vel.x_right:
                      x_np1 -= self.vel.dx
                  elif x_np1 < self.vel.x_left:
                      x_np1 += self.vel.dx
              except:
                  pass
              
              try:
                  if y_np1 >= self.vel.y_up:
                      y_np1 -= self.vel.dy
                  elif y_np1 < self.vel.y_down:
                      y_np1 += self.vel.dy
              except:
                  pass

          
          x_v          = np.append(x_v, x_np1)
          y_v          = np.append(y_v, y_np1)
          q_v          = np.append(q_np1, q_v)
          p_v          = np.append(p_np1, p_v)
          
      pos_vec  = np.transpose(np.array([x_v, y_v]))
      q0_vec   = np.transpose(np.array([np.flip(q_v), np.flip(p_v)]))
      
      if flag == True:
          self.pos_vec = np.copy(pos_vec)
          self.q0_vec  = np.copy(q0_vec)
      
      return pos_vec, q0_vec
      
