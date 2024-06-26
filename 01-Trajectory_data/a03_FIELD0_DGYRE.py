import numpy as np
from a03_FIELD0_00000 import velocity_field

"""This script defines the velocity field of the DOUBLE GYRE.
We take the definition of the field from D. Garaboa-Paz and
V. Pérez-Muñuzuri (2015), with the same name of the parameters
as in the paper."""

class velocity_field_DoubleGyre(velocity_field):
    
    def __init__(self, periodic=False, A = 0.1, eps = 0.25, T = 10.0):
        self.periodic  = periodic
        self.limits    = periodic
        self.x_left    = 0.
        self.x_right   = 2.
        self.dx        = self.x_right - self.x_left
        self.y_down    = 0.
        self.y_up      = 2.
        self.dy        = self.y_up - self.y_down
        
        # Define parameters of the Double Gyre
        self.A         = A
        self.eps       = eps
        self.T         = T
        
        self.omega     = 2.0 * np.pi / self.T
    
    def impose_periodicity(self, x, y):
        if self.periodic == True:
            if isinstance(x, float) == True:
                if x < self.x_left:
                    x += self.dx
                elif x > self.x_right:
                    x -= self.dx
                    
                if y < self.y_down:
                    y += self.dy
                elif y > self.y_up:
                    y -= self.dy
                    
            else:
                x[x < self.x_left]  += self.dx
                x[x > self.x_right] -= self.dx
                y[y < self.y_up]    += self.dy
                y[y > self.y_down]  -= self.dy
        
        return x, y
    
    def a(self, t):
        return self.eps * np.sin(self.omega * t)
    
    def b(self, t):
        return 1.0 - 2.0 * self.eps * np.sin(self.omega * t)
    
    def f(self, x, t):
        return self.a(t) * x**2.0 + self.b(t) * x
    
    def df_dx(self, x, t):
        return 2.0 * self.a(t) * x + self.b(t)
    
    def d2f_dx2(self, t):
        return 2.0 * self.a(t)
    
    def da_dt(self, t):
        return self.eps * self.omega * np.cos(self.omega * t)
    
    def db_dt(self, t):
        return -2.0 * self.eps * self.omega * np.cos(self.omega * t)
    
    def df_dt(self, x, t):
        return self.da_dt(t) * x**2.0 + self.db_dt(t) * x
    
    def d2f_dxdt(self, x, t):
        return 2.0 * self.da_dt(t) * x + self.db_dt(t)
    
    def get_velocity(self, x, y, t):
        x, y = self.impose_periodicity(x, y)
        
        u    = - self.A * np.pi * np.sin(np.pi * self.f(x,t)) * np.cos(np.pi * y)
        v    = self.A * np.pi * np.cos(np.pi * self.f(x, t)) * \
               self.df_dx(x, t) * np.sin(np.pi * y)
        return u, v
    
    def get_gradient(self, x, y, t):
        x, y = self.impose_periodicity(x, y)
        
        ux = - self.A * np.pi**2.0 * np.cos(np.pi * self.f(x, t)) *\
             self.df_dx(x, t) * np.cos(np.pi * y)    
        uy = self.A * np.pi**2.0 * np.sin(np.pi * self.f(x, t)) *\
             np.sin(np.pi * y)
        vx = - self.A * np.pi**2.0 * np.sin(np.pi * self.f(x, t)) *\
             self.df_dx(x, t)**2.0 * np.sin(np.pi * y) +\
             self.A * np.pi * np.cos(np.pi * self.f(x, t)) *\
             self.d2f_dx2(t) * np.sin(np.pi * y)
        vy = self.A * np.pi**2.0 * np.cos(np.pi * self.f(x, t)) *\
             self.df_dx(x, t) * np.cos(np.pi * y)
        return ux, uy, vx, vy

    def get_dudt(self, x, y, t):
        x, y = self.impose_periodicity(x, y)
        
        ut = -self.A * np.pi**2.0 * np.cos(np.pi * self.f(x, t)) *\
             self.df_dt(x, t) * np.cos(np.pi * y)
        vt = -self.A * np.pi**2.0 * np.sin(np.pi * y) *\
             np.sin(np.pi * self.f(x, t)) *\
             self.df_dt(x, t) * self.df_dx(x, t) +\
             self.A * np.pi * np.sin(np.pi * y) *\
             np.cos(np.pi * self.f(x, t)) *\
             self.d2f_dxdt(x, t)
        return ut, vt
