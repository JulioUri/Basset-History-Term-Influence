from a03_FIELD0_00000 import velocity_field
import numpy as np

class velocity_field_Bickley(velocity_field):

  def __init__(self, periodic=True):
    # Below, the characteristic length scales of the flow, used to nondimensionalise.
    self.U = 5.414 # Mm/days
    self.L = 1.770 # Mm
    self.T = self.L / self.U # ~0.327 days 
      
    # Set a variety of parameters; from bickleyjet.m provided by Prof Padberg-Gehle.
    # See e.g. Padberg-Gehle, Schneide: NPG, 2017.
    self.re = 6.371
    self.cc1 = 0.1446
    self.cc2 = 0.205
    self.cc3 = 0.461
    self.myA = np.array([0.0075, 0.15, 0.3])
    self.c1 = self.cc1*self.U
    self.c2 = self.cc2*self.U
    self.c3 = self.cc3*self.U
    self.myk = np.array([2.0/self.re, 4.0/self.re, 6.0/self.re])
    self.mys = np.array([self.k(1)*self.c1, self.k(2)*self.c2, self.k(3)*self.c3])
    
    '''
    Define the space domain for which we have data.
    if self.lmits == False, then the programme understands the field is defined
    in all R^2.
    if self.limits == True, then the programme understands the field is defined
    only within a subset of R^2 which needs to be defined.
    '''
    self.periodic = periodic
    self.limits   = False
    
    if self.periodic == True:
        self.x_left    = 0. / self.L
        self.x_right   = 20. / self.L
        self.dx        = self.x_right - self.x_left
    
  def impose_periodicity(self, x):
    if self.periodic:
        if isinstance(x, float):  # Handle scalar input
            if x < self.x_left:
                x += self.dx
            elif x >= self.x_right:
                x -= self.dx
        else:  # Handle array input
            # Use numpy's where to apply periodicity adjustments
            x = np.where(x < self.x_left, x + self.dx, x)
            x = np.where(x >= self.x_right, x - self.dx, x)
    return x
  
  def sechL(self,y):
    # With dimensional space
    # return 1.0/np.cosh(y/self.L)
    
    # With nondimensional space
    return 1.0/np.cosh(y)
  
  def tanhL(self,y):
    # With dimensional space
    # return np.tanh(y/self.L)
    
    # With nondimensional space
    return np.tanh(y)
  
  def sink(self,x,t,k):
    assert (k>=1 and k<=3), "k must be an integer between 1 and 3"
    # To match indexing in Padberg-Gehle's Matlab Script, accept values for k between 1 and 3 and subtract 1 to match Python's indexing which starts with 0.
    
    # With dimensional space and time
    # return np.sin(self.k(k)*x - self.s(k)*t)
    
    # With nondimensional space and time
    return np.sin(self.k(k)*self.L*x - self.s(k)*self.T*t)

  def cosk(self,x,t,k):
    assert (k>=1 and k<=3), "k must be an integer between 1 and 3"
    # To match indexing in Padberg-Gehle's Matlab Script, accept values for k between 1 and 3 and subtract 1 to match Python's indexing which starts with 0.
    
    # With dimensional space and time
    # return np.cos(self.k(k)*x - self.s(k)*t)
    
    # With nondimensional space and time
    return np.cos(self.k(k)*self.L*x - self.s(k)*self.T*t)
    
  
  def k(self, i):
    return self.myk[i-1]
  
  def A(self, i):
    return self.myA[i-1]
  
  def s(self, i):
    return self.mys[i-1]
  
  def f(self,x,t):
    return self.A(1)*self.cosk(x,t,1) \
      + self.A(2)*self.cosk(x,t,2) \
      + self.A(3)*self.cosk(x,t,3)
  
  def u1(self,y):
    # Dimensional velocity #
    # return self.U*(1.0 - self.tanhL(y)**2)
    # Nondimensional velocity #
    return (1.0 - self.tanhL(y)**2)
  
  def u2(self,y):
    # Dimensional velocity
    # return 2.0*self.U*self.sechL(y)**2*self.tanhL(y)
    # Nondimensional velocity
    return 2.0*self.sechL(y)**2*self.tanhL(y)
  
  def g(self,x,t):
    return self.A(1)*self.k(1)*self.sink(x,t,1) \
      + self.A(2)*self.k(2)*self.sink(x,t,2) \
      + self.A(3)*self.k(3)*self.sink(x,t,3)
  
  def v1(self, y):
    # Dimensional velocity
    # return -self.U*self.L*self.sechL(y)**2
    # Nondimensional velocity
    return -self.L*self.sechL(y)**2

  def du1dy(self,y):
    # Dimensional derivative
    # return -2.0*(self.U/self.L)*self.tanhL(y)*self.sechL(y)**2
    # Nondimensional derivative
    return -2.0*self.tanhL(y)*self.sechL(y)**2
  
  def du2dy(self,y):
    # Dimensional derivative
    # return 2.0*(self.U/self.L)*self.sechL(y)**2*( self.sechL(y)**2 - 2.0*self.tanhL(y)**2)
    # Nondimensional derivative
    return 2.0*self.sechL(y)**2*( self.sechL(y)**2 - 2.0*self.tanhL(y)**2)
  
  def dfdx(self, x, t):
    return -self.A(1)*self.k(1)*self.sink(x,t,1) \
      - self.A(2)*self.k(2)*self.sink(x,t,2) \
      - self.A(3)*self.k(3)*self.sink(x,t,3)
  
  def dfdt(self, x, t):
    return self.A(1)*self.s(1)*self.sink(x,t,1) \
      + self.A(2)*self.s(2)*self.sink(x,t,2) \
      + self.A(3)*self.s(3)*self.sink(x,t,3)
  
  def dv1dy(self,y):
    # Dimensional derivative
    # return 2*self.U*self.tanhL(y)*self.sechL(y)**2
    # Nondimensional derivative
    return 2*self.L*self.tanhL(y)*self.sechL(y)**2
  
  def dgdx(self, x, t):
    return self.A(1)*self.k(1)**2*self.cosk(x,t,1) \
      + self.A(2)*self.k(2)**2*self.cosk(x,t,2) \
      + self.A(3)*self.k(3)**2*self.cosk(x,t,3)
  
  def dgdt(self, x, t):
    return - self.A(1)*self.k(1)*self.s(1)*self.cosk(x,t,1) \
      - self.A(2)*self.k(2)*self.s(2)*self.cosk(x,t,2) \
      - self.A(3)*self.k(3)*self.s(3)*self.cosk(x,t,3)
  
  '''
  See the notes for the definition of the auxiliary functions u1, u2, f, v1, g
  '''
  def get_velocity(self, x, y, t):
    # Impose periodic horizontal domain (like a cylinder)
    x = self.impose_periodicity(x)
    
    u = self.u1(y) + self.u2(y)*self.f(x,t)
    v = self.v1(y)*self.g(x,t)
    
    return u, v
  
  def get_gradient(self, x, y, t):
    # Impose periodic horizontal domain (like a cylinder)
    x = self.impose_periodicity(x)
    
    # Dimensional version
    # ux = self.u2(y)*self.dfdx(x,t)
    # uy = self.du1dy(y) + self.du2dy(y)*self.f(x,t)
    # vx = self.v1(y)*self.dgdx(x,t)
    # vy = self.dv1dy(y)*self.g(x,t)
    
    # Nondimensional version
    ux = self.L * self.u2(y)*self.dfdx(x,t)
    uy = self.du1dy(y) + self.du2dy(y)*self.f(x,t)
    vx = self.L * self.v1(y)*self.dgdx(x,t)
    vy = self.dv1dy(y)*self.g(x,t)
    
    return ux, uy, vx, vy

  def get_dudt(self, x, y, t):
    # Impose periodic horizontal domain (like a cylinder)
    x = self.impose_periodicity(x)
      
    # Dimensional version
    # ut = self.u2(y)*self.dfdt(x,t)
    # vt = self.v1(y)*self.dgdt(x,t)
    
    # Nondimensional version
    ut = self.T * self.u2(y)*self.dfdt(x,t)
    vt = self.T * self.v1(y)*self.dgdt(x,t)
    
    return ut, vt
