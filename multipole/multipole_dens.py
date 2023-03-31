from typing import Dict, Union
import numpy as np
import numpy.linalg as la
import MDAnalysis as mda
from MDAnalysis.analysis.base import AnalysisBase
from scipy.constants import elementary_charge
from .point_charge_water import PCParams, pc_parameters
from numpy.typing import NDArray
from numba import jit
from numba import double,float32 

# using fast histogram
from fast_histogram import histogram1d

# import matplotlib.pyplot as plt

# profiling code


class Multipoles(AnalysisBase):
    _axis_map = {"x": 0, "y": 1, "z": 2}

    def __init__(
        self,
        select,
        centre="M",
        grouping="water",
        axis="z",
        ll: float = 0.0,
        ul: Union[float, None] = None,
        binsize: float = 0.25,
        H_types=["H"],
        type_or_name=None,
        calculate_dummy: bool = False,  # Calculate the dummy atom positions on the fly
        model_params: Union[str, Dict, PCParams] = "tip4p05",
        unwrap: bool = False,
        **kwargs,
    ):
        """
        Prepare the Analysis Class.

        select : Trajectory Selection

        centre : Atom label for defining the origin. default: 'M' the dummy/oxygen atom
                location
        grouping : default 'water'

        TODO! Add contributions to charge from other parts of the system
            - Find a way to select atoms that aren't in the water molecule
              independently

        Options
            model_params - Name of water model params (see point_charge_water.pc_params)
                            or set of parameters

        """
        # Use the AnalysisBase Init Function. Set everything up
        # grab init
        super(Multipoles, self).__init__(select.universe.trajectory, **kwargs)

        # allows use of run(parallel=True)
        self._ags = [select]
        self._universe: mda.Universe = select.universe

        # Set options

        self.calculate_dummy = calculate_dummy
        if self.calculate_dummy:
            print("Calculating dummy atom positions on the fly!")
        # Set the dummy parameters dict
        if isinstance(model_params, str):
            try:
                self.model_params: PCParams = pc_parameters[model_params]
            except KeyError:
                raise KeyError(f"Parameters for water model {model_params} not found!")
        elif isinstance(model_params, dict):
            self.model_params: PCParams = PCParams.from_dict(model_params)
        else:
            self.model_params = model_params

        self.binsize = binsize

        # group of atoms on which to compute the COM (same as used in
        # AtomGroup.wrap())

        # whether to select defined atoms for water (e.g. H/M) using the molecules type
        # or name.
        # if selected manually, takes priority, else determines if names or types are
        # defined

        if type_or_name is not None:
            self.type_or_name = type_or_name
        else:
            self.type_or_name = check_types_or_names(self._universe)

        # Box sides
        self.centretype = centre  # name or type of atom central atom per molecule
        self.grouping = grouping

        if self.grouping == "water":
            self.H_types = H_types

        # currently only supports orthogonal boxes
        # get the index for the axis used here
        self._axind = self._axis_map[axis]

        self.dimensions = self._universe.dimensions[:3]
        self.Lx, self.Ly, self.Lz = self.dimensions
        self.volume = np.prod(self.dimensions)

        # if ul is defined and smaller than the axis extent
        if ul and ul < self.dimensions[self._axind]:
            self.ul = ul
        else:
            self.ul = self.dimensions[self._axind]
        # Raise error if ll < ul
        self.ll = ll
        if self.ll > self.ul:
            raise ValueError(
                f"Lower limit of axis {self._axind} must be smaller "
                "than the upper limit!"
            )
        self.unwrap = unwrap

        self.range = (self.ll, self.ul)

        # number of bins
        bins = int((self.ul - self.ll) // self.binsize)

        # Here we choose a number of bins of the largest cell side so that
        # x, y and z values can use the same "coord" column in the output file
        self.nbins = bins
        self.slice_vol = self.volume / bins

        self.keys_common = ["pos", "pos_std", "char", "char_std"]

        # Initialize results array with zeros/
        # these are all profiles
        # TODO Make these dicts of x y z
        self.quadrapole = np.zeros((3, 3, self.nbins))
        self.dipole = np.zeros((3, self.nbins))
        self.charge_dens = np.zeros(self.nbins)
        self.cos_theta = np.zeros(self.nbins)
        self.angular_moment_2 = np.zeros(self.nbins)
        # just the oxygen/Mw locations!
        self.mol_density = np.zeros(self.nbins)
        self.qdens_water = np.zeros(self.nbins)
        self.qdens_ions = np.zeros(self.nbins)

        # Variables later defined in _prepare() method
        # can be variant
        self.masses = None
        self.charges = None
        self.totalmass = None

    def _prepare(self):
        # group must be a local variable, otherwise there will be
        # issues with parallelization
        # group = getattr(self._ags[0], self.grouping)

        # Get masses and charges for the selection
        # try:  # in case it's not an atom
        #     self.masses = np.array([elem.total_mass() for elem in group])
        #     self.charges = np.array([elem.total_charge() for elem in group])
        # except AttributeError:  # much much faster for atoms
        #     self.masses = self._ags[0].masses
        #     self.charges = self._ags[0].charges

        # for now just doing water generalise later

        # self.totalmass = np.sum(self.masses)

        self.Matoms: mda.AtomGroup = self._universe.select_atoms(
            f"{self.type_or_name} {self.centretype}"
        )
        self.Hatoms: mda.AtomGroup = self._universe.select_atoms(
            f"{self.type_or_name} {' '.join(self.H_types)}"
        )  # joining all the H types with a string
        # self.Oatoms = self._universe.select_atoms('name O')
        # OLD CODE: SELECTED WATER CHARGED ATOMS
        # self.chargedatoms = self._universe.select_atoms(
        #     f"{self.type_or_name} {' '.join(self.H_types)} {self.centretype}"
        # )
        # TODO Select charged atoms that are not Water atoms
        self.chargedatoms: mda.AtomGroup = self._universe.atoms
        # Select atoms that are not hydrogen or oxygen
        self.ions: mda.AtomGroup = self._universe.select_atoms(
            f"not {self.type_or_name} {self.centretype} {' '.join(self.H_types)}"
        )
    # @profile
    def _single_frame(self):
        """
        Operations performed on a single frame to calculate the binning of the charge
        densities
        """
        # if self._universe.trajectory.frame % 1000 == 0:
        #     print(self._universe.trajectory.frame)
        # print(self._universe.trajectory.frame)
        # TODO access this information before the frame?
        qH: NDArray = self.Hatoms.charges[0]

        rH: NDArray = self.Hatoms.positions
        rH1: NDArray = rH[::2]  # select everyother H
        rH2: NDArray = rH[1::2]
        rM: NDArray = self.Matoms.positions

        # TODO add flag for deciding if to unwrap or not
        if self.unwrap:
            rM, rH1, rH2 = unwrap_water_coords(rM, rH1, rH2, self.dimensions)
        # Apply find positions of the dummy atom
        if self.calculate_dummy:
            # TODO unwrap the positions of the oxygen and hydrogen atoms
            rM = get_dummy_position(rM, rH1, rH2, self.model_params["r_M"])
            # update positions in charged atoms group TODO Decide if necessary
            self.Matoms.positions = rM

        # wrapping coords. Needed for the calculation of the
        rMw = (
            rM % self.dimensions
        )  # mod of the positions with respect to the box dimensions

        # calculate the dipole and quadrupole moments of the water molecules
        mus = calculate_dip(rM, [rH1, rH2], qH)
        # print(Q.shape)
        quad = calculate_Q(rM, rH1, rH2, qH)  # full tensor

        # calculate the first angle moment by projecting on z axis
        # first axis is atoms, second axis are the cartesian components
        # TODO deal with warning that occurs on first frame
        cos_theta = mus[:, self._axind] / np.linalg.norm(mus, axis=1)

        # if self._universe.trajectory.frame % 1000 == 0:
        # print('cos(theta)',cos_theta[100],mus[100])

        cos_moment_2 = 0.5 * (3 * cos_theta**2 - 1)

        # z position
        zMw = rMw[:, self._axind]
        # zposes = vstack([zM]*3).T

        # Calculate the histograms, with positions on the dummy atom, weighted by the property being binned
        mu_hist = np.vstack(
            [
                histogram1d(zMw, bins=self.nbins, weights=mus[:, i], range=self.range)[
                    0
                ]
                for i in range(3)
            ]
        )

        Q_hist = np.stack(
            [
                [
                    histogram1d(
                        zMw, bins=self.nbins, range = self.range,weights=quad[:, i, j], 
                    )
                    for i in range(3)
                ]
                for j in range(3)
            ]
        )

        # assert (np.abs(Q_hist_fast-Q_hist) < 1e-17).all()

        mol_hist = histogram1d(zMw, bins=self.nbins, range=self.range)
        # un weighted!

        # NB These are divided by the number of molecules in each bin; i.e., average of
        # cos(theta) per bin
        angle_hist = np.nan_to_num(
            histogram1d(zMw, bins=self.nbins, weights=cos_theta, range=self.range)
            / mol_hist
        )  # divided by the number of molecules

        # r = np.random.randint(0, self.nbins)
        # if self._universe.trajectory.frame % 1000 == 0:
        #     print(
        #         "cos(theta)",
        #         angle_hist[r],
        #       np.histogram(zMw, bins=self.nbins, weights=cos_theta, range=self.range)[
        #             0
        #         ][r],
        #         mol_hist[r],
        #     )

        # convert nans in profiles to zeros -> assumes that if density is zero then
        # can't have any angles!!!!
        angle_sq_hist = np.nan_to_num(histogram1d(zMw, bins=self.nbins, weights=cos_moment_2, range=self.range)/mol_hist)
        # print(Q_hist.shape)

        # print(mu_hist.dtype)
        # print(self.dipole.dtype)
        # TODO Change to use the moved positions of the oxygen charges!
        charge_hist = histogram1d(
            (self.chargedatoms.positions % self.dimensions)[:, self._axind],
            bins=self.nbins,
            weights=self.chargedatoms.charges,
            range=self.range,
        )

        # Calculate charge hist for ions and water separately
        charge_ion_hist = histogram1d(
            (self.ions.positions % self.dimensions)[:, self._axind],
            bins=self.nbins,
            weights=self.ions.charges,
            range=self.range,
        )
        # positions of water atoms
        water_positions = np.concatenate([rM, rH1, rH2])
        # Charges of water atoms
        water_charges = np.concatenate([self.Matoms.charges, self.Hatoms.charges])
        assert water_positions.shape[0] == water_charges.shape[0]
        charge_water_hist = histogram1d(
            (water_positions % self.dimensions)[:, self._axind],
            bins=self.nbins,
            weights=water_charges,
            range=self.range,
        )

        # TODO Add potential calculations here?

        # running sum!
        self.dipole += mu_hist
        self.quadrapole += Q_hist
        self.charge_dens += charge_hist
        self.qdens_ions += charge_ion_hist
        self.qdens_water += charge_water_hist
        self.mol_density += mol_hist

        self.cos_theta += angle_hist
        self.angular_moment_2 += angle_sq_hist

        # if self._universe.trajectory.frame % 1000 == 0:
        #   print(self.cos_theta/(self._universe.trajectory.frame+1))
        # if self._universe.trajectory.frame % 1000 == 0:
        #   plt.plot(self.cos_theta/(self._universe.trajectory.frame+1)); plt.show()

        # self.left_edges = left_edges[:-1]  # exclude the right most edge

        # self.group = getattr(self._ags[0], self.grouping)
        # self._ags[0].wrap(compound=self.grouping)

        # # Find position of atom/group of atoms
        # if self.grouping == 'atoms':
        #     positions = self._ags[0].positions  # faster for atoms
        # else:
        #     # COM for res/frag/etc
        #     positions = np.array([elem.centroid() for elem in self.group])

        # for dim in ['x', 'y', 'z']:
        #     idx = self.results[dim]['dim']

        #     key = 'pos'
        #     key_std = 'pos_std'
        #     # histogram for positions weighted on masses
        #     hist, _ = np.histogram(positions[:, idx],
        #                            weights=self.masses,
        #                            bins=self.nbins,
        #                            range=(0.0, max(self.dimensions)))

        #     self.results[dim][key] += hist
        #     self.results[dim][key_std] += np.square(hist)

        # key = 'char'
        # key_std = 'char_std'
        # # histogram for positions weighted on charges
        # hist, _ = np.histogram(positions[:, idx],
        #                        weights=self.charges,
        #                        bins=self.nbins,
        #                        range=(0.0, max(self.dimensions)))

        # self.results[dim][key] += hist
        # self.results[dim][key_std] += np.square(hist)

    def _conclude(self):
        # k = 6.022e-1  # divide by avodagro and convert from A3 to cm3
        """Units are AA
        mu - eAA/AA^3 -> C/AA^2
        Q - eAA^2/AA^3 -> C/AA
        rho_q - e/AA^3 -> C/AA^3
        TODO: Add option to keep in e/AA^N
        """

        # Average results over the  number of configurations
        for prop in [
            "charge_dens",
            "dipole",
            "quadrapole",
            "qdens_ions",
            "qdens_water",
        ]:
            # divide by the volume and number of frames used and convert to
            #  Coloumbs per unit
            prop_val = (
                getattr(self, prop) / self.n_frames / self.slice_vol * elementary_charge
            )
            setattr(self, prop, prop_val)
        for prop in ["mol_density"]:
            # divide by the volume and number of frames used and
            prop_val = getattr(self, prop) / self.n_frames / self.slice_vol
            setattr(self, prop, prop_val)
        for prop in ["cos_theta", "angular_moment_2"]:
            # divide by the number of frames used. NB. Already divided by the number of
            # molecules per chunk.
            prop_val = getattr(self, prop) / self.n_frames
            setattr(self, prop, prop_val)

        #     self.results[dim][key] /= self.n_frame
        #     # Compute standard deviation for the error
        #     self.results[dim]['pos_std'] = np.sqrt(self.results[dim][
        #         'pos_std'] - np.square(self.results[dim]['pos']))
        #     self.results[dim]['char_std'] = np.sqrt(self.results[dim][
        #         'char_std'] - np.square(self.results[dim]['char']))

        # for dim in ['x', 'y', 'z']:
        #     self.results[dim]['pos'] /= self.results[dim]['slice volume'] * k
        #     self.results[dim]['char'] /= self.results[dim]['slice volume'] * k
        #     self.results[dim]['pos_std'] /= self.results[dim]['slice volume'] * k
        #     self.results[dim]['char_std'] /= self.results[dim]['slice volume'] * k

    def _add_other_results(self, other):
        # TODO doesn;t currently work, need to implement a results dict???
        # For parallel analysis
        self.results = {self.charge_dens, self.dipole, self.quadrapole}
        results = self.results
        for dim in ["x", "y", "z"]:
            key = "pos"
            key_std = "pos_std"
            results[dim][key] += other[dim][key]
            results[dim][key_std] += other[dim][key_std]

            key = "char"
            key_std = "char_std"
            results[dim][key] += other[dim][key]
            results[dim][key_std] += other[dim][key_std]


# Library Functions
def calculate_dip(Mpos, Hposes, qH):
    # TODO let qH be a vector for each H, for use with reaxff
    # will need to split charges into a list for each H
    # I guess not maybe...
    # TODO: Allow a different origin for dipole moment calculation
    return ((Hposes[0] - Mpos) + (Hposes[1] - Mpos)) * qH


# @staticmethod
def check_types_or_names(universe: mda.Universe) -> str:
    """
    Determine whether the mda Universe is using names or types for the atom types
    """
    try:
        universe.atoms.names
        # if no error raised automatically use names
        type_or_name = "name"
    except mda.NoDataError:
        try:
            universe.atoms.types
            type_or_name = "type"
        except mda.NoDataError:
            raise mda.NoDataError("Universe has neither atom type or name information.")
    return type_or_name


def calculate_Q_no_numba(Mpos, H1, H2, qH):
    """New version of calculating the charges, rather than iterating over per atom,
    iterate over per component??? iterate over 9 things rather than 2000
    TODO add more general implementation for residue/fragment groupings
    and a water calculation.
    """

    # return (
    #     np.array([np.outer(v, v) for v in Hposes[0] - Mpos])
    #     + np.array([np.outer(v, v) for v in Hposes[1] - Mpos])
    # ) * qH

    # transpose so each component can be looked at individually

    v1 = H1 - Mpos
    v2 = H2 - Mpos

    # the outer product of the same N*3 array to give a N*3*3 array
    v1_sq = v1.reshape([-1, 3, 1]) @ v1.reshape([-1, 1, 3])
    # the -1 in the first dimension indicates reshaping so that
    v2_sq = v2.reshape([-1, 3, 1]) @ v2.reshape([-1, 1, 3])

    return 0.5 * (qH * v1_sq + qH * v2_sq)


# @jit((float32[:,:],float32[:,:],float32[:,:],double),nopython=True)
@jit(nopython=True)
def calculate_Q(Mpos, H1, H2, qH):
    """New version of calculating the charges, rather than iterating over per atom,
    iterate over per component??? iterate over 9 things rather than 2000
    TODO add more general implementation for residue/fragment groupings
    and a water calculation.

    No numba version is likely faster if only run for a few frames, but
    numba version is faster for many frames, because of the compilation time

    """

    # return (
    #     np.array([np.outer(v, v) for v in Hposes[0] - Mpos])
    #     + np.array([np.outer(v, v) for v in Hposes[1] - Mpos])
    # ) * qH

    # transpose so each component can be looked at individually

    v1 = H1 - Mpos
    v2 = H2 - Mpos

    # the outer product of the same N*3 array to give a N*3*3 array
    # the -1 in the first dimension indicates reshaping so that
    result = np.zeros((len(v1), 3, 3))

    for i in range(len(v1)):
        for j in range(3):
            for k in range(3):
                result[i, j, k] = v1[i, j] * v1[i, k] + v2[i, j] * v2[i, k]

    return 0.5 * (qH * result)


def get_dummy_position_no_numba(
    xO: NDArray, xH1: NDArray, xH2: NDArray, dM: float
) -> NDArray:
    """
    Calculate the position of the dummy atoms from the positions of the oxygen
    and hydrogen atoms
    Assumes that hydrogen atoms for a given molecule have neighbouring indices.
    """
    dx1 = xH1 - xO
    dx2 = xH2 - xO
    # TODO, faster to pass in bond length manually? Not useful for the case of flexible
    # water models
    dxM = dx1 + dx2

    # Normalise the vector
    # Indexing ensures normalisation over correct axis and gives correct shape
    dxM = dxM / la.norm(dxM, axis=1)[:, None]
    return xO + dM * dxM

# import the types

# @jit((float32[:,:],float32[:,:],float32[:,:],double),nopython=True) # type annotations allow for precompilation
@jit(nopython=True)
def get_dummy_position(xO: NDArray, xH1: NDArray, xH2: NDArray, dM: float) -> NDArray:
    """
    Calculate the position of the dummy atoms from the positions of the oxygen
    and hydrogen atoms
    Assumes that hydrogen atoms for a given molecule have neighbouring indices.
    """
    dx1 = xH1 - xO
    dx2 = xH2 - xO
    # TODO, faster to pass in bond length manually? Not useful for the case of flexible
    # water models
    dxM = dx1 + dx2

    # Normalise the vector
    # Indexing ensures normalisation over correct axis and gives correct shape
    norms = np.zeros(len(dxM))
    for i in range(len(dxM)):
        norms[i] = np.linalg.norm(dxM[i])  # numba doesn't support axis argument

    dxM = (dxM.T / norms).T

    return xO + dM * dxM


# TODO have a no numba version
# def unwrap_water_coords(rO, rH1, rH2, box):
#     # check if the bond is longer than half box length
#     # save check in vector an apply unwrapping based on this
#     unwrap_me = rH1 - rO > box / 2
#     rH1_new = rH1 - box * unwrap_me

#     unwrap_me = rH2 - rO > box / 2
#     rH2_new = rH2 - box * unwrap_me
#     return rO, rH1_new, rH2_new


# @jit(nopython=True)
# TODO These type annotations don't seem to work when called
@jit((float32[:,:],float32[:,:],float32[:,:],float32[:]),nopython=True)
def unwrap_water_coords(rO, rH1, rH2, box):
    """
    Iterate over atoms then over dimensions, check if bond is longer than half box length
    and apply unwrapping if necessary
    """
    for i in range(rO.shape[0]):
        for j in range(3):
            if rH1[i, j] - rO[i, j] > box[j] / 2:
                rH1[i, j] -= box[j]
            if rH2[i, j] - rO[i, j] > box[j] / 2:
                rH2[i, j] -= box[j]
    return rO, rH1, rH2


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time

    # from trajectory_prepare import TrajectoryPreparer

    print("Started", time.strftime("%a, %d %b %Y %H:%M:%S"))

    # u = mda.Universe("../300K/rep3/addedMs.gro", "../300K/rep3/addedMs.dcd")
    u = mda.Universe(
        "../test_data_files/tip4p05_LV_400K.data",
    )

    # add charges and masses for tip4p/2005

    # u.add_TopologyAttr("charge")
    # u.select_atoms("name O").charges = 0
    # u.select_atoms("name O").masses = 15.9994
    # u.select_atoms("name H").charges = 0.5564
    # u.select_atoms("name H").masses = 1.008
    # u.select_atoms("name M").charges = -1.1128
    # u.select_atoms("name M").masses = 0

    # TrajectoryPreparer(u,)

    #
    multip = Multipoles(u.atoms, verbose=True, H_types=1, centre=1)
    multip.run()
    z = np.linspace(multip.ll, multip.ul, multip.nbins)

    np.savetxt(
        "profs.dat",
        [
            z,
            multip.charge_dens,
            multip.dipole[0, :],
            multip.dipole[1, :],
            multip.dipole[2, :],
            multip.quadrapole[0, 0, :],
            multip.quadrapole[1, 1, :],
            multip.quadrapole[2, 2, :],
            multip.quadrapole[0, 1, :],
            multip.quadrapole[0, 2, :],
            multip.quadrapole[1, 2, :],
        ],
        header="coord charge_dens P_x P_y P_z Q_xx Q_yy Q_zz Q_xy Q_xz Q_yz",
    )

    plt.plot(z, multip.charge_dens, label="rho")
    plt.plot(z, multip.quadrapole[2, 2, :], label="$Q_zz$")
    plt.plot(z, multip.dipole[2, :], label="$P_z$")
    plt.savefig("profs.png")
    plt.show()

    print("Finished", time.strftime("%a, %d %b %Y %H:%M:%S"))
