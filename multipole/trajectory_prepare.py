from multipole.point_charge_water import PointChargeModel

import MDAnalysis as mda

import argparse

from .multipole_dens import Multipoles
import MDAnalysis.transformations


class TrajectoryPreparer:
    ''' 
        Prepare the trajectory by setting the water model and overriding charges

    '''

    def __init__(self, args_parser: argparse.ArgumentParser, models: PointChargeModel = None):

        args = args_parser.parse_args()
        self.args = args

        # Verify if the model objects are requested
        if args.water_model and models is not None:
            try:
                self.model: PointChargeModel = models[args.water_model]
            except IndexError:
                raise IndexError(f'Water Model f{args.water_model} not found')

        if args.trajfile:
            self.u = mda.Universe(
                args.topfile, args.trajfile, in_memory=args.inmem)
            # set timestep to last for checking of unwrapping
            self.u.trajectory[-1]
        else:
            self.u = mda.Universe(args.topfile, in_memory=args.inmem)

        self.set_up_args()

        if self.model is not None:
            self.set_water_model()

    def set_up_args(self):

        if self.args.types_or_names:
            self.types_or_names = self.args.types_or_names
        else:
            self.types_or_names = Multipoles.check_types_or_names(self.u)

        self.H_name = self.args.H_name
        self.M_name = self.args.M_name

    def set_water_model(self):
        ''' Override some of the MDA Charges with those of the water model verifies that the charges are the correct ones... 
            I'm not sure why the requirement for this came about... 
            I think the out put of tip4p/2005 dummy wasn't giving the correct charges or the topology file types were not saving charges properly'''
        if self.args.verbose:
            print(f'Overwriting with {self.args.water_model} charges:')
            print(f'q_H = {self.model.q} e')
        self.u.add_TopologyAttr('charge')
        self.u.select_atoms(
            f"{self.types_or_names} {' '.join(self.H_name)}").charges = self.model.q

        self.u.select_atoms(
            f"{self.types_or_names} {' '.join(self.args.H_name)}").masses = 1.008
        self.u.select_atoms(
            f'{self.types_or_names} {self.args.M_name}').charges = -2*self.model.q

        if self.model.r_M == 0:
            M_mass = 15.9994  # For three point models, set mass of negative charge to oxygen mass
        else:
            M_mass = 0
        self.u.select_atoms(
            f'{self.types_or_names} {self.args.M_name}').masses = M_mass

        pass

    def wrap_check(self):
        if ((self.u.dimensions[:3]-self.u.atoms.positions < 0).any()) or not self.args.check_unwrapped:
         # then coords are definitely already unwrapped, nothing needs to be done
            if self.args.verbose:
                print(
                    'Coordinates definitely unwrapped or not checked. Proceeding to calculation')
            else:
                # coords may or may not be unwrapped. Apply transformation just to be safe
                if self.args.verbose:
                    print('Coordinates might not be unwrapped, unwrapping in case.')
                ag = self.u.atoms
                unwrap_transform = MDAnalysis.transformations.unwrap(ag)
                self.u.trajectory.add_transformations(unwrap_transform)

    def rewind_traj(self):
        if self.args.trajfile:
            self.u.trajectory[0]
