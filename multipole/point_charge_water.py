from dataclasses import dataclass
from typing import Dict
import numpy as np
from scipy.constants import elementary_charge, speed_of_light


class PointChargeModel:
    e_AA_in_debye = speed_of_light * elementary_charge * 1e11  # 1/0.2081943

    def __init__(self, q:float, r:float, theta:float, r_M:float=0.0):
        self.theta_rad = theta * np.pi / 180

        self.r = r

        self.r_M = r_M

        self.q = q

    @property
    def theta(self):
        """Getter for angle in degrees"""
        return self.theta_rad * 180 / np.pi

    @theta.setter
    def theta(self, value):
        self.theta_rad = value * np.pi / 180

    @property
    def y(self):
        return self.r * np.sin(self.theta_rad / 2)

    @property
    def z2(self):
        return self.r * np.cos(self.theta_rad / 2)

    @property
    def mu_e_AA(self):
        return 2 * self.q * (self.z2 - self.r_M)

    @property
    def quad_T_e_AA_sq(self):
        return 3 * self.q * self.y**2 / 2

    @property
    def mu(self):
        return self.mu_e_AA * self.e_AA_in_debye

    @property
    def quad_T(self):
        """in Debye Angstrom"""

        return self.quad_T_e_AA_sq * self.e_AA_in_debye

    @property
    def quad(self):
        """
        The quadrupole moment in eA^2
        Computed relative to the site of the negative charge
        Returned as a matrix
        """
        coords = self.coords(include_dummy=True)

        # rO = coords[0]
        rM = coords[1]
        rH1 = coords[2]
        rH2 = coords[3]
        ##
        r1 = rH1 - rM
        r2 = rH2 - rM

        return (
            1 / 2 * self.q * (np.outer(r1, r1) + np.outer(r2, r2))
        )  # Factor of half by the definition in our papers

    @property
    def mu_alt(self):
        """
        The alternate definition of mu, relative to the negative site, in debye angstrom
        """

        # coords =

        rO, rM, rH1, rH2 = self.coords(include_dummy=True)

        # rH1 = coords[2]
        # rH2 = coords [3]
        ##
        r1 = rH1 - rM
        r2 = rH2 - rM
        # print('r1, r2 ',r1,r2)

        # print('-z2 + m',self.z2-self.r_M)
        # return 2*self.q*(self.r*np.cos(self.theta_rad/2) - self.r_M)

        return self.q * r1 + self.q * r2

    @property
    def trace_quad(self):
        """In e^2 angstrom"""

        return np.trace(self.quad)

    @property
    def mu_abs(self):
        mu_v = self.mu_alt
        # print(mu_v)
        return np.sqrt(mu_v[0] ** 2 + mu_v[1] ** 2 + mu_v[2] ** 2)

    def coords(self, include_dummy=False):
        """Get the coordinates of the atoms of a molecule lying in the y-z plane.
        TODO add rotation options??"""
        # vector of coordinate locations

        vec_O = np.zeros(3)
        vec_H1 = np.array(
            [
                0,
                self.y,
                -self.z2,
            ]
        )
        vec_H2 = vec_H1.copy()
        vec_H2[1] *= -1  # opposite sign
        vec_M = np.array([0, 0, -1 * self.r_M])

        if not include_dummy:
            return np.stack([vec_O, vec_H1, vec_H2])
        else:
            return np.stack([vec_O, vec_M, vec_H1, vec_H2])

    def __repr__(self):
        return "\n".join(
            [
                f"r_0 = {self.r} Å, θ = {self.theta} °, q = {self.q} C",
                str(self.coords()),
            ]
        )


@dataclass
class PCParams:
    q: float
    r: float
    theta: float
    r_M: float

    @staticmethod
    def from_dict(pdict: Dict[str, float]):
        '''
            Create a PCParams object from a dictionary
            TODO: Add type hint. Could not get working with dataclass
        '''
        try:
            return PCParams(
                q=pdict["q"], r=pdict["r"], theta=pdict["theta"], r_M=pdict["r_M"]
            )
        except KeyError:
            raise KeyError("Dictionary must contain q, r, theta, r_M")

    # Index the PCParams object like a dictionary
    def __getitem__(self, key):
        return getattr(self, key)


pc_parameters: Dict[str, PCParams] = {
    "tip4p05": PCParams.from_dict(
        {"q": 0.5564, "r": 0.9572, "theta": 104.52, "r_M": 0.1546}
    ),
    "opc": PCParams.from_dict(
        {"q": 0.6791, "r": 0.8724, "theta": 103.6, "r_M": 0.1594}
    ),
    "spce": PCParams.from_dict({"q": 0.4238, "r": 1.0000, "theta": 109.47, "r_M": 0}),
    "tip3p": PCParams.from_dict({"q": 0.417, "r": 0.9572, "theta": 104.52, "r_M": 0}),
}


# Create  dict of the PointChargeModel objects, using the above parameter sets
pc_objs = {
    model: PointChargeModel(params.q, params.r, params.theta, params.r_M)
    for model, params in pc_parameters.items()
}


if __name__ == "__main__":
    # print(pc_objs['opc'].coords(include_dummy=True))
    print(pc_objs["opc"].mu)
    print(2 * (pc_objs["opc"].z2 - pc_objs["opc"].r_M) ** 2 * pc_objs["opc"].q)
    print(pc_objs["opc"].quad)
    print(pc_objs["opc"].trace_quad / 3)
    print(pc_objs["opc"].mu_alt)
    print(pc_objs["opc"].mu_abs)
    print(pc_objs["opc"].mu_e_AA)

    print(pc_parameters["tip4p05"]["r_M"])