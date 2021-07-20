import numpy as np
from scipy.constants import elementary_charge, speed_of_light


class PointChargeModel:

    e_AA_in_debye = speed_of_light*elementary_charge*1e11  # 1/0.2081943

    def __init__(self, q, r, theta, r_M=0):

        self.theta_rad = theta * np.pi / 180

        self.r = r

        self.r_M = r_M

        self.q = q

    @property
    def theta(self):
        '''Getter for angle in degrees'''
        return self.theta_rad * 180/np.pi

    @theta.setter
    def theta(self, value):
        self.theta_rad = value*np.pi/180

    @property
    def y(self):
        return self.r * np.sin(self.theta_rad/2)

    @property
    def z2(self):
        return self.r * np.cos(self.theta_rad/2)

    @property
    def mu_e_AA(self):
        return 2*self.q * (self.z2-self.r_M)

    @property
    def quad_T_e_AA_sq(self):
        return 3*self.q*self.y**2/2

    @property
    def mu(self):
        return self.mu_e_AA * self.e_AA_in_debye

    @property
    def quad_T(self):
        '''in Debye Angstrom'''

        return self.quad_T_e_AA_sq * self.e_AA_in_debye

    def coords(self, include_dummy=False):
        ''' Get the coordinates of the atoms of a molecule lying in the x-y plane. TODO add rotation options?? '''
        # vector of coordinate locations

        vec_O = np.zeros(3)
        vec_H1 = np.array([self.y, -self.z2, 0])
        vec_H2 = vec_H1.copy()
        vec_H2[0] *= -1  # opposite sign
        vec_M = np.array([0, 0, -1*self.r_M])

        if not include_dummy:
            return np.stack([vec_O, vec_H1, vec_H2])
        else:
            return np.stack([vec_O, vec_M, vec_H1, vec_H2])

    def __repr__(self):
        return '\n'.join([f'r_0 = {self.r} Å, θ = {self.theta} °, q = {self.q} C', str(self.coords())])


pc_parameters = {'tip4p05': {'q': 0.5564, 'r': 0.9572, 'theta': 104.52, 'r_M': 0.1546},
                 'opc': {'q': 0.6791, 'r': 0.8724, 'theta': 103.6, 'r_M': 0.1594},
                 'spce': {'q': 0.4238, 'r': 1.0000, 'theta': 109.47, 'r_M': 0},
                 'tip3p': {'q': 0.417, 'r': 0.9572, 'theta': 104.52, 'r_M': 0}
                 }


# Create  dict of the PointChargeModel objects, using the above parameter sets
pc_objs = {model: PointChargeModel(**params)
           for model, params in pc_parameters.items()}


if __name__ == '__main__':
    print(pc_objs['opc'].mu)
