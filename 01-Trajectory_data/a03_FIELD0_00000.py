from abc import ABC, abstractmethod

'''
This is the abstract method that defines the velocity field.
'''

class velocity_field(ABC):
    
    def __init__(self):
        self.limits   = False
        self.periodic = False
        self.x_left   = None
        self.x_right  = None
        self.y_down   = None
        self.y_up     = None
        pass
    
    @abstractmethod
    def get_velocity(self, x, y, t):
        u = 0 * x
        v = 0 * y
        return u, v
    
    @abstractmethod
    def get_gradient(self, x, y, t):
        ux, uy = 0 * x, 0 * y
        vx, vy = 0 * x, 0 * y
        return ux, uy, vx, vy

    @abstractmethod
    def get_dudt(self, x, y, t):
        ut = 0 * x
        vt = 0 * y
        return ut, vt