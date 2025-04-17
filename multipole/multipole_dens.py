from dataclasses import dataclass
from .point_charge_water import PCParams, pc_parameters

from typing import Dict, List, Optional, Union
import warnings

import numpy as np
import numpy.linalg as la
from numpy.typing import NDArray
import MDAnalysis as mda
from MDAnalysis.analysis.base import AnalysisBase, Results
from scipy.constants import elementary_charge
from numba import jit
from numba import float32
from fast_histogram import histogram1d

# Structs for Results


@dataclass
class FrameResult:
    """Histogram results per frame"""

    charge_dens: NDArray
    dipole: NDArray
    quadrupole: NDArray
    qdens_ions: NDArray
    qdens_water: NDArray
    mol_density: NDArray
    cos_theta: NDArray
    angular_moment_2: NDArray


class Multipoles(AnalysisBase):
    _axis_map = {"x": 0, "y": 1, "z": 2}

    def __init__(
        self,
        select,
        *,
        grouping="water",
        axis: str = "z",
        lower_limit: float = 0.0,
        upper_limit: Union[float, None] = None,
        binsize: float = 0.25,
        centre: str | int = "M",
        H_types: List[str | int] = ["H"],
        type_or_name: Optional[str] = None,
        calculate_dummy: bool = False,  # Calculate the dummy atom positions on the fly
        model_params: Union[str, Dict, PCParams] = "tip4p05",
        origin_dist: float = 0.0,
        unwrap: bool = False,
        **kwargs,
    ):
        """
        Prepares the Multipoles analysis class.

        Arguments
        ---------
        select :
            Trajectory Selection
        centre :
            Atom name or type for defining the dummy/oxygen atom location.
            This is used as the origin for binning and caculation of molecular dipoles
            for `grouping`="water".
            default: 'M'
        H_types :
            The name or type(s) used for the Hydrogen atoms.
        grouping :
            "water" or "residue": default 'water'
            For grouping water, the non-water atoms have their charge densities tallied in
            the `qdens_ions` field.
        model_params :
            Name of water model or a set of parameters in a Dictionary.
            (see point_charge_water.pc_params)
            Valid names are ["tip4p05", "opc", "spce", "tip3p"]
        axis:
            The name of the axis to bin along.
            Valid options are: "x", "y" or "z"
        binsize:
            The spacing (in distance units) to use for binning the results.
        lower_limit:
            The lower bounds of binning. Defaults to zero.
        upper_limit:
            The upper bounds of binning. Defaults to edge of the box.
        calculate_dummy: bool = False,
            Calculate the position of dummy atoms in water molecules on the fly.
        unwrap: bool = False
            If true, will unwrap the atomic coordinates on the fly, otherwise, will
            assume they are already unwrapped. Unwrapping is required for molecular
            multipole moments.
        origin: float = 0.0
            Distance from the oxygen along the dipole moment vector used for the
            calculation of multipole moments and binning.
        kwargs:
            Dictionary of keyword arguments passed down to the MDAnalysis AnalysisBase
            class.

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

        self.centretype = centre  # name or type of atom central atom per molecule

        # Grouping
        self.grouping = grouping
        if self.grouping == "water":
            self.H_types = H_types
        elif self.grouping == "residue":
            self.H_types = []  # No hydrogen types needed.
        else:
            # TODO: Add `molecular` grouping based on molecule ids
            raise ValueError("Only grouping='water' or 'residue' are supported.")

        # Currently only supports orthogonal boxes
        # Get the index for the axis used here
        self._axind = self._axis_map[axis]

        self.dimensions = self._universe.dimensions[:3]
        self.Lx, self.Ly, self.Lz = self.dimensions
        self.volume = np.prod(self.dimensions)

        # if ul is defined and smaller than the axis extent
        if upper_limit and upper_limit < self.dimensions[self._axind]:
            self.ul = upper_limit
        else:
            self.ul = self.dimensions[self._axind]
        # Raise error if ll < ul
        self.ll = lower_limit
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

        self.origin_dist = origin_dist

        self.keys_common = ["pos", "pos_std", "char", "char_std"]

        # Initialize results array with zeros/
        # these are all profiles
        # TODO: Make these dicts of x y z to support all three dimensions.
        self._results_tally = FrameResult(
            quadrupole=np.zeros((3, 3, self.nbins)),
            dipole=np.zeros((3, self.nbins)),
            charge_dens=np.zeros(self.nbins),
            cos_theta=np.zeros(self.nbins),
            angular_moment_2=np.zeros(self.nbins),
            # just the oxygen/Mw locations!
            mol_density=np.zeros(self.nbins),
            qdens_water=np.zeros(self.nbins),
            qdens_ions=np.zeros(self.nbins),
        )

        # Variables later defined in _prepare() method
        # can be variant
        self.masses = None
        self.charges = None
        self.totalmass = None

    def _prepare(self):
        # TODO: for now just doing water generalise later

        if self.grouping == "water":
            self.Matoms: mda.AtomGroup = self._universe.select_atoms(
                f"{self.type_or_name} {self.centretype}"
            )
            self.Hatoms: mda.AtomGroup = self._universe.select_atoms(
                f"{self.type_or_name} {' '.join((str(t) for t in self.H_types))}"
            )

            # All water atoms
            self.water_atoms: mda.AtomGroup = self._universe.select_atoms(
                f"{self.type_or_name} {self.centretype} {' '.join((str(t) for t in self.H_types))}"
            )

            self.charged_atoms: mda.AtomGroup = self._universe.atoms  # pyright: ignore

            # Select atoms that are not hydrogen or oxygen
            self.ions: mda.AtomGroup = self._universe.select_atoms(
                f"not {self.type_or_name} {self.centretype} {' '.join((str(t) for t in self.H_types))}"
            )
        else:
            # TODO: Verify _universe.atoms is not None
            self.charged_atoms: mda.AtomGroup = self._universe.atoms  # pyright: ignore
            self.ions: mda.AtomGroup = self._universe.atoms  # pyright: ignore

    def _single_frame(self):
        """
        Operations performed on a single frame to calculate the binning of the charge
        and dipole densities
        """
        if self.grouping == "water":
            self._single_frame_water()
        elif self.grouping == "residue":
            self._single_frame_residue()
        else:
            raise ValueError(
                "Invalid Grouping: valid options are 'water' and 'residue' "
            )

    def _single_frame_residue(self):
        """
        Operations performed on a single frame to calculate the binning of the charge
        and dipole densities

        Version for grouping = "residue"
        """

        # TODO: Deal with the horrendous amount of duplication found here.

        residues = self._universe.residues

        if residues is None:
            raise ValueError(
                'Expected residues to be defined in topology for grouping="residue"'
            )

        rM = com_residues(residues)

        if self.unwrap:
            # TODO:
            warnings.warn("Unwrap is not currently applied for residues")

        # Dipoles per residue
        mus = calculate_dip_residues(residues, rM)

        # TODO: Implement quadrupole calculations
        quad = np.zeros((residues.n_residues, 3, 3))
        # TODO: Add custom origins
        quad = calculate_Q_residues(residues, rM)

        # wrapping coords. Needed for the calculation of the
        rMw = (
            rM % self.dimensions
        )  # mod of the positions with respect to the box dimensions

        # calculate the first angle moment by projecting on z axis
        # first axis is atoms, second axis are the cartesian components
        # TODO: deal with warning that occurs on first frame
        cos_theta = mus[:, self._axind] / np.linalg.norm(mus, axis=1)

        # if self._universe.trajectory.frame % 1000 == 0:
        # print('cos(theta)',cos_theta[100],mus[100])

        cos_moment_2 = 0.5 * (3 * cos_theta**2 - 1)

        # z position
        zMw = rMw[:, self._axind]
        # zposes = vstack([zM]*3).T

        # Calculate the histograms, with positions on the dummy atom,
        # weighted by the property being binned
        mu_hist = np.vstack(
            [
                histogram1d(zMw, bins=self.nbins, weights=mus[:, i], range=self.range)
                for i in range(3)
            ]
        )

        Q_hist = np.stack(
            [
                [
                    histogram1d(
                        zMw,
                        bins=self.nbins,
                        range=self.range,
                        weights=quad[:, i, j],
                    )
                    for i in range(3)
                ]
                for j in range(3)
            ]
        )

        mol_hist = histogram1d(zMw, bins=self.nbins, range=self.range)
        # un weighted!

        # NOTE: These are divided by the number of molecules in each bin;
        # i.e., average of  cos(theta) per bin
        angle_hist = np.nan_to_num(
            histogram1d(zMw, bins=self.nbins, weights=cos_theta, range=self.range)
            / mol_hist
        )  # divided by the number of molecules

        # convert nans in profiles to zeros -> assumes that if density is zero then
        # can't have any angles!!!!
        angle_sq_hist = np.nan_to_num(
            histogram1d(zMw, bins=self.nbins, weights=cos_moment_2, range=self.range)
            / mol_hist
        )

        # TODO: Change to use the moved positions of the oxygen charges!
        charge_hist = histogram1d(
            (self.charged_atoms.positions % self.dimensions)[:, self._axind],
            bins=self.nbins,
            weights=self.charged_atoms.charges,
            range=self.range,
        )

        # Calculate charge hist for ions and water separately
        charge_ion_hist = histogram1d(
            (self.ions.positions % self.dimensions)[:, self._axind],
            bins=self.nbins,
            weights=self.ions.charges,
            range=self.range,
        )

        results = FrameResult(
            dipole=mu_hist,
            quadrupole=Q_hist,
            charge_dens=charge_hist,
            qdens_ions=charge_ion_hist,
            qdens_water=np.zeros(self.nbins),
            mol_density=mol_hist,
            cos_theta=angle_hist,
            angular_moment_2=angle_sq_hist,
        )

        self._tally_results(results)

    def _single_frame_water(self):
        """
        Operations performed on a single frame to calculate the binning of the charge
        and dipole densities

        Version for grouping = "water"
        """
        # TODO access this information before the frame?
        qH: NDArray = self.Hatoms.charges[0]

        rH: NDArray = self.Hatoms.positions
        rH1: NDArray = rH[::2]  # select everyother H
        rH2: NDArray = rH[1::2]
        rM: NDArray = self.Matoms.positions

        # TODO: Double check the order is the expected order
        r_water: NDArray = self.water_atoms.positions

        # TODO: add flag for deciding if to unwrap or not
        if self.unwrap:
            rM, rH1, rH2 = unwrap_water_coords(rM, rH1, rH2, self.dimensions)

        # TODO: Add the calculation of different reference point here.
        origin = get_dummy_position(rM, rH1, rH2, self.origin_dist)

        # Apply find positions of the dummy atom
        if self.calculate_dummy:
            # TODO unwrap the positions of the oxygen and hydrogen atoms
            rM = get_dummy_position(rM, rH1, rH2, self.model_params["r_M"])
            # update positions in charged atoms group TODO Decide if necessary
            self.Matoms.positions = rM

        # wrapping coords. Needed for the calculation of the
        # mod of the positions with respect to the box dimensions
        origin_w = origin % self.dimensions
        r_water_w = r_water % self.dimensions

        # TODO: Make this an option.

        # calculate the dipole and quadrupole moments of the water molecules
        mus = calculate_dip(rM, [rH1, rH2], qH)
        # print(Q.shape)
        quad = calculate_Q(rM, rH1, rH2, qH, origin)  # full tensor

        # calculate the first angle moment by projecting on z axis
        # first axis is atoms, second axis are the cartesian components
        # TODO: deal with warning that occurs on first frame
        cos_theta = mus[:, self._axind] / np.linalg.norm(mus, axis=1)

        # if self._universe.trajectory.frame % 1000 == 0:
        # print('cos(theta)',cos_theta[100],mus[100])

        cos_moment_2 = 0.5 * (3 * cos_theta**2 - 1)

        # TODO: Move this to some sort of external
        # TODO: Support more kinds of spread
        SPREAD = True

        # z position used for binning
        if SPREAD:
            zMw = r_water_w[:, self._axind]

            def spread_f(x: NDArray) -> NDArray:
                return x.repeat(3)
        else:
            zMw = origin_w[:, self._axind]

            def spread_f(x: NDArray) -> NDArray:
                return x

        # zposes = vstack([zM]*3).T

        # Calculate the histograms, with positions on the dummy atom,
        # weighted by the property being binned
        mu_hist = np.vstack(
            [
                histogram1d(
                    zMw, bins=self.nbins, weights=spread_f(mus[:, i]), range=self.range
                )
                for i in range(3)
            ]
        )

        Q_hist = np.stack(
            [
                [
                    histogram1d(
                        zMw,
                        bins=self.nbins,
                        range=self.range,
                        weights=spread_f(quad[:, i, j]),
                    )
                    for i in range(3)
                ]
                for j in range(3)
            ]
        )

        # assert (np.abs(Q_hist_fast-Q_hist) < 1e-17).all()

        # TODO: This is now just number density when using spread, not molecule density...
        mol_hist = histogram1d(zMw, bins=self.nbins, range=self.range)
        # un weighted!

        # NOTE: These are divided by the number of molecules in each bin;
        # i.e., average of  cos(theta) per bin
        angle_hist = np.nan_to_num(
            histogram1d(
                zMw, bins=self.nbins, weights=spread_f(cos_theta), range=self.range
            )
            / mol_hist
        )  # divided by the number of molecules

        # convert nans in profiles to zeros -> assumes that if density is zero then
        # can't have any angles!!!!
        angle_sq_hist = np.nan_to_num(
            histogram1d(
                zMw, bins=self.nbins, weights=spread_f(cos_moment_2), range=self.range
            )
            / mol_hist
        )

        # NOTE: These are not spread (via `.repeat(3)`) because they are already per atom
        # TODO: Change to use the moved positions of the oxygen charges!
        charge_hist = histogram1d(
            (self.charged_atoms.positions % self.dimensions)[:, self._axind],
            bins=self.nbins,
            weights=self.charged_atoms.charges,
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

        results = FrameResult(
            dipole=mu_hist,
            quadrupole=Q_hist,
            charge_dens=charge_hist,
            qdens_ions=charge_ion_hist,
            qdens_water=charge_water_hist,
            mol_density=mol_hist,
            cos_theta=angle_hist,
            angular_moment_2=angle_sq_hist,
        )

        self._tally_results(results)

    def _conclude(self):
        # k = 6.022e-1  # divide by avodagro and convert from A3 to cm3
        """
        Distance units are AA
        Charge units are C

        mu - eAA/AA^3 -> C/AA^2
        Q - eAA^2/AA^3 -> C/AA
        rho_q - e/AA^3 -> C/AA^3
        TODO: Add option to keep in e/AA^N

        NOTE:
        Results are saved by normalising cumulative sums of the properties.
        Also now returns a `results` attribute.
        """

        # Average results over the  number of configurations

        n_frames = self.n_frames
        vol = self.slice_vol
        e = elementary_charge

        # Put all fields into a results dict
        tally = self._results_tally

        # TODO: Should this clone, so re-run doesn't overwrite these refs?
        self.results = Results(
            # divide by the volume and number of frames used and convert to
            #  Coloumbs per unit
            charge_dens=tally.charge_dens / n_frames / vol * e,
            dipole=tally.dipole / n_frames / vol * e,
            quadrupole=tally.quadrupole / n_frames / vol * e,
            qdens_ions=tally.qdens_ions / n_frames / vol * e,
            qdens_water=tally.qdens_water / n_frames / vol * e,
            # divide by the volume and number of frames used and
            mol_density=tally.mol_density / n_frames / vol,
            # divide by the number of frames used. NB. Already divided by the number of
            # molecules per chunk.
            cos_theta=tally.cos_theta / n_frames,
            angular_moment_2=tally.angular_moment_2 / n_frames,
        )

    ## Helper methods
    def _tally_results(self, result: FrameResult):
        # running sum!
        self._results_tally.dipole += result.dipole
        self._results_tally.quadrupole += result.quadrupole

        self._results_tally.charge_dens += result.charge_dens
        self._results_tally.qdens_ions += result.qdens_ions
        self._results_tally.qdens_water += result.qdens_water

        self._results_tally.mol_density += result.mol_density

        self._results_tally.cos_theta += result.cos_theta
        self._results_tally.angular_moment_2 += result.angular_moment_2


# Library Functions


def com(ag: mda.AtomGroup) -> NDArray:
    return (ag.positions * ag.masses[:, np.newaxis]).sum(axis=0) / ag.masses.sum()


def com_residues(residues: mda.ResidueGroup) -> NDArray:
    """
    Calculate the Centre of Mass for each residue in the system.
    """
    output = np.zeros((residues.n_residues, 3))

    # TODO: Find a way to avoid this likely slow iteration over residues.
    for i, res in enumerate(residues):
        output[i] = com(res.atoms)

    return output


def calculate_dip(Mpos, Hposes, qH):
    """
    Old version of calculating dipole moment that doesn't use Numba
    """
    # TODO: let qH be a vector for each H, for use with reaxff
    # will need to split charges into a list for each H
    # I guess not maybe...
    # TODO: Allow a different origin for dipole moment calculation

    dip_dir = (Hposes[0] - Mpos) + (Hposes[1] - Mpos)

    return dip_dir * qH


def calculate_dip_residues(residues: mda.ResidueGroup, origins: NDArray) -> NDArray:
    """
    Calculate the dipole moment of each residue (molecule) in the system.
    """

    output = np.zeros((residues.n_residues, 3))

    # TODO: Find a way to avoid this likely slow iteration over residues.
    for i, (com, res) in enumerate(zip(origins, residues)):
        output[i] = calculate_dip_ag(res.atoms.positions, res.atoms.charges, com)

    return output


def calculate_dip_ag(positions: NDArray, charges: NDArray, origin: NDArray) -> NDArray:
    """
    Calculate the dipole moment of an atom group, relative to an origin

    Arguments
    ---------
    positions:
        A (N,3) numpy array of atom positions
    charges:
        A (N,) numpy array of atomic charges
    origin:
        The origin used for the dipole calculation. Should have shape (3,)
    """

    return (charges[:, np.newaxis] * (positions - origin)).sum(axis=0)


def calculate_Q_residues(residues: mda.ResidueGroup, origins: NDArray) -> NDArray:
    """
    Calculate the dipole moment of each residue (molecule) in the system.
    """

    output = np.zeros((residues.n_residues, 3, 3))

    # TODO: Find a way to avoid this likely slow iteration over residues.
    for i, (com, res) in enumerate(zip(origins, residues)):
        output[i] = calculate_Q_ag(res.atoms.positions, res.atoms.charges, com)

    return output


@jit()
def calculate_Q_ag(positions: NDArray, charges: NDArray, origin: NDArray) -> NDArray:
    """
    Calculate the Quadrupole Moment of an atom group, relative to a provided

    Arguments
    ---------
    positions:
        A (N,3) numpy array of atom positions
    charges:
        A (N,) numpy array of atomic charges in the molecule
    origin:
        The origin used for the quadrupole moment calculation. Should have shape (3,)
    """

    pos_rel = positions - origin

    result = np.zeros((3, 3))

    # TODO: probably could write this in numpy primatives without
    for i in range(len(pos_rel)):
        q = charges[i]
        p = pos_rel[i]
        for j in range(3):
            for k in range(3):
                result[j, k] += q * p[j] * p[k]

    return 0.5 * result


def check_types_or_names(universe: mda.Universe) -> str:
    """
    Determine whether the `mda.Universe` is using names or types for the atom types
    """
    try:
        universe.atoms.names  # pyright: ignore
        # if no error raised automatically use names
        type_or_name = "name"
    except mda.NoDataError:
        try:
            universe.atoms.types  # pyright: ignore
            type_or_name = "type"
        except mda.NoDataError:
            raise mda.NoDataError("Universe has neither atom type or name information.")
    return type_or_name


def calculate_Q_no_numba(Mpos, H1, H2, qH):
    """Calculating the quadrupoles. Rather than iterating over per atom,
    iterates over per component. Iterate over 9 things rather than 2000
    """

    # TODO: add more general implementation for residue/fragment groupings
    # and a water calculation.

    v1 = H1 - Mpos
    v2 = H2 - Mpos

    # the outer product of the same N*3 array to give a N*3*3 array
    v1_sq = v1.reshape([-1, 3, 1]) @ v1.reshape([-1, 1, 3])
    # the -1 in the first dimension indicates reshaping so that
    v2_sq = v2.reshape([-1, 3, 1]) @ v2.reshape([-1, 1, 3])

    return 0.5 * (qH * v1_sq + qH * v2_sq)


@jit(nopython=True)
def calculate_Q(Mpos, H1, H2, qH, origin):
    """New version of calculating the charges, rather than iterating over per atom,
    iterate over per component??? iterate over 9 things rather than 2000
    TODO add more general implementation for residue/fragment groupings
    and a water calculation.

    No numba version is likely faster if only run for a few frames, but
    numba version is faster for many frames, because of the compilation time

    """

    v1 = H1 - origin
    v2 = H2 - origin
    v3 = Mpos - origin

    result = np.zeros((len(v1), 3, 3))

    for i in range(len(v1)):
        for j in range(3):
            for k in range(3):
                result[i, j, k] = (
                    v1[i, j] * v1[i, k]
                    + v2[i, j] * v2[i, k]
                    - 2.0 * v3[i, j] * v3[i, k]  # Term due to -ve charge
                )

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
    # TODO: faster to pass in bond length manually? Not useful for the case of flexible
    # water models
    dxM = dx1 + dx2

    # Normalise the vector
    # Indexing ensures normalisation over correct axis and gives correct shape
    dxM = dxM / la.norm(dxM, axis=1)[:, None]
    return xO + dM * dxM


@jit(nopython=True)
def get_dummy_position(xO: NDArray, xH1: NDArray, xH2: NDArray, dM: float) -> NDArray:
    """
    Calculate the position of the dummy atoms from the positions of the oxygen
    and hydrogen atoms
    Assumes that hydrogen atoms for a given molecule have neighbouring indices.
    """
    dx1 = xH1 - xO
    dx2 = xH2 - xO
    # TODO: faster to pass in bond length manually? Not useful for the case of flexible
    # water models
    dxM = dx1 + dx2

    # Normalise the vector
    # Indexing ensures normalisation over correct axis and gives correct shape
    norms = np.zeros(len(dxM))
    for i in range(len(dxM)):
        norms[i] = np.linalg.norm(dxM[i])  # numba doesn't support axis argument

    dxM = (dxM.T / norms).T

    return xO + dM * dxM


# TODO: These type annotations don't seem to work when called
@jit((float32[:, :], float32[:, :], float32[:, :], float32[:]), nopython=True)
def unwrap_water_coords(rO, rH1, rH2, box):
    """
    Iterate over atoms then over dimensions, check if bond is longer
    than half box length and apply unwrapping if necessary
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

    print("Started", time.strftime("%a, %d %b %Y %H:%M:%S"))

    u = mda.Universe(
        "../test_data_files/tip4p05_LV_400K.data",
    )

    multip = Multipoles(u.atoms, verbose=True, H_types=1, centre=1)  # pyright: ignore
    multip.run()
    z = np.linspace(multip.ll, multip.ul, multip.nbins)

    np.savetxt(
        "profs.dat",
        [
            z,
            multip.results.charge_dens,
            multip.results.dipole[0, :],
            multip.results.dipole[1, :],
            multip.results.dipole[2, :],
            multip.results.quadrupole[0, 0, :],
            multip.results.quadrupole[1, 1, :],
            multip.results.quadrupole[2, 2, :],
            multip.results.quadrupole[0, 1, :],
            multip.results.quadrupole[0, 2, :],
            multip.results.quadrupole[1, 2, :],
        ],
        header="coord charge_dens P_x P_y P_z Q_xx Q_yy Q_zz Q_xy Q_xz Q_yz",
    )

    plt.plot(z, multip.results.charge_dens, label="rho")
    plt.plot(z, multip.results.quadrupole[2, 2, :], label="$Q_zz$")
    plt.plot(z, multip.results.dipole[2, :], label="$P_z$")
    plt.savefig("profs.png")
    plt.show()

    print("Finished", time.strftime("%a, %d %b %Y %H:%M:%S"))
