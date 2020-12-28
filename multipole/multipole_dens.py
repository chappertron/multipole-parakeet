import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.base import AnalysisBase
from scipy.constants import elementary_charge



class Multipoles(AnalysisBase):

    _axis_map = {'x':0,'y':1,'z':2}

    def __init__(self, select, centre = 'M',grouping = 'water', axis = 'z' ,ll = 0,ul = None, binsize=0.25, **kwargs):
        super(Multipoles, self).__init__(select.universe.trajectory,
                                            **kwargs)
        # grab init 
        # allows use of run(parallel=True)
        self._ags = [select]
        self._universe = select.universe



        self.binsize = binsize

        # group of atoms on which to compute the COM (same as used in
        # AtomGroup.wrap())
        
        # Box sides
        self.centretype = 'M' # name or type of atom central atom per molecule
        self.grouping = grouping

        # currently only supports orthogonal boxes
        self._axind = self._axis_map[axis] # get the index for the axis used here
        
        self.dimensions= self._universe.dimensions[:3]
        self.Lx,self.Ly,self.Lz = self.dimensions
        self.volume = np.prod(self.dimensions)
        
        # if ul is defined and smaller than the axis extent
        if ul and ul < self.dimensions[self._axind]:
            self.ul = ul
        else:
            self.ul = self.dimensions[self._axind]
        # TODO Raise error if ll < ul
        self.ll = ll
        if self.ll > self.ul:
            raise ValueError(f'Lower limit of axis {self._axind} must be smaller than the upper limit!')

        self.range = (self.ll,self.ul)
        
        # number of bins
        bins = ((self.ul-self.ll) // self.binsize).astype(int)

        # Here we choose a number of bins of the largest cell side so that
        # x, y and z values can use the same "coord" column in the output file
        self.nbins = bins
        self.slice_vol = self.volume / bins


        self.keys_common = ['pos', 'pos_std', 'char', 'char_std']

        # Initialize results array with zeros/
        # these are all profiles
        # TODO Make these dicts of x y z
        self.quadrapole = np.zeros((3, 3, self.nbins))
        self.dipole = np.zeros((3,self.nbins))
        self.charge_dens = np.zeros(self.nbins)

        # Variables later defined in _prepare() method
        # can be variant
        self.masses = None
        self.charges = None
        self.totalmass = None

    def _prepare(self):
        # group must be a local variable, otherwise there will be
        # issues with parallelization
        #group = getattr(self._ags[0], self.grouping)

        # Get masses and charges for the selection
        # try:  # in case it's not an atom
        #     self.masses = np.array([elem.total_mass() for elem in group])
        #     self.charges = np.array([elem.total_charge() for elem in group])
        # except AttributeError:  # much much faster for atoms
        #     self.masses = self._ags[0].masses
        #     self.charges = self._ags[0].charges

        # for now just doing water generalise later



        #self.totalmass = np.sum(self.masses)

        self.Matoms = self._universe.select_atoms(f'name {self.centretype}')
        self.Hatoms = self._universe.select_atoms('name H')
        self.Oatoms = self._universe.select_atoms('name O')
        self.chargedatoms = self._universe.select_atoms(f'name H {self.centretype}')

    def _single_frame(self):

        if self._universe.trajectory.frame % 1000 == 0:
            print(self._universe.trajectory.frame)

        qH = self.Hatoms.charges[0] # TODO generalise by making this charge a vector, can use with reaxff then.

        rH1 = self.Hatoms.positions[::2] #select everyother H
        rH2 = self.Hatoms.positions[1::2]
        rM = self.Matoms.positions

        Q = self.calculate_Q(rM,[rH1,rH2],qH) # full tensor
        #print(Q.shape)
        mus = self.calculate_dip(rM,[rH1,rH2],qH)
        
        zM = rM[:,self._axind] # z position 
        #zposes = vstack([zM]*3).T
        
        #print(shape(zposes))
        mu_hist = np.vstack([np.histogram(zM,bins = self.nbins,weights = mus[:,i],range = self.range)[0] for i in range(3)])
        Q_hist = np.stack([[np.histogram(zM,bins = self.nbins,weights= Q[:,i,j],range = self.range)[0] for i in range(3)] for j in range(3)])
        #print(Q_hist.shape)
        
        # print(mu_hist.dtype)
        # print(self.dipole.dtype)
        
        self.dipole += mu_hist
        self.quadrapole += Q_hist
        self.charge_dens += np.histogram(self.chargedatoms.positions[:,self._axind],bins = self.nbins,weights=self.chargedatoms.charges ,range = self.range)[0]



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
        #k = 6.022e-1  # divide by avodagro and convert from A3 to cm3

        '''Units are AA
            mu - eAA/AA^3 -> C/AA^2
            Q - eAA^2/AA^3 -> C/AA
            rho_q - e/AA^3 -> C/AA^3
        '''

        # Average results over the  number of configurations
        for prop in ['charge_dens','dipole','quadrapole']:
            prop_val = getattr(self,prop)/self.n_frames/self.slice_vol * elementary_charge # divide by the volume and number of frames used and convert to Coloumbs per unit
            setattr(self,prop,prop_val)
        
        
        
        
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
        self.results = {self.charge_dens,self.dipole,self.quadrapole}
        results = self.results
        for dim in ['x', 'y', 'z']:
            key = 'pos'
            key_std = 'pos_std'
            results[dim][key] += other[dim][key]
            results[dim][key_std] += other[dim][key_std]

            key = 'char'
            key_std = 'char_std'
            results[dim][key] += other[dim][key]
            results[dim][key_std] += other[dim][key_std]




    def calculate_dip(self,Mpos,Hposes,qH):
        # TODO let qH be a vector for each H, for use with reaxff
        # will need to split charges into a list for each H
        # I guess not maybe...
        return ((Hposes[0]-Mpos) + (Hposes[1]-Mpos)) * qH
    
    def old_calculate_Q(self,Mpos,Hposes,qH):
        return (np.array([np.outer(v,v) for v in Hposes[0]-Mpos]) + np.array([np.outer(v,v) for v in Hposes[1]-Mpos])) * qH

    def calculate_Q(self,Mpos,Hposes,qH):
        ''' New version of calculating the charges, rather than iterating over per atom, iterate over per component??? iterate over 9 things rather than 2000'''

        # TODO make this and calculate_dip be static methods -> they don't use any information from the object!!!

        #return (np.array([np.outer(v,v) for v in Hposes[0]-Mpos]) + np.array([np.outer(v,v) for v in Hposes[1]-Mpos])) * qH

        # transpose so each component can be looked at individually

        v1 = Hposes[0] - Mpos
        v2 = Hposes[1] - Mpos

        v1_sq = v1.reshape([-1, 3, 1]) @ v1.reshape([-1, 1, 3]) # the outer product of the same N*3 array to give a N*3*3 array 
        v2_sq = v2.reshape([-1, 3, 1]) @ v2.reshape([-1, 1, 3]) # the -1 in the first dimension indicates reshaping so that 

        return qH * v1_sq + qH * v2_sq

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import time

    print('Started', time.strftime("%a, %d %b %Y %H:%M:%S"))

    u = mda.Universe('300K/rep3/addedMs.gro','300K/rep3/addedMs.dcd')

    # add charges and masses for tip4p/2005

    u.add_TopologyAttr('charge')
    u.select_atoms('name O').charges = 0
    u.select_atoms('name O').masses = 15.9994
    u.select_atoms('name H').charges = 0.5564
    u.select_atoms('name H').masses = 1.008
    u.select_atoms('name M').charges = -1.1128
    u.select_atoms('name M').masses = 0


    #
    multip = Multipoles(u.atoms,verbose=True)
    multip.run()
    z = np.linspace(multip.ll,multip.ul,multip.nbins)
    
    np.savetxt('profs.dat',[z, multip.charge_dens,multip.dipole[0,:],multip.dipole[1,:], multip.dipole[2,:],multip.quadrapole[0,0,:],
    multip.quadrapole[1,1,:],multip.quadrapole[2,2,:],multip.quadrapole[0,1,:],multip.quadrapole[0,2,:],multip.quadrapole[1,2,:]],header='coord charge_dens P_x P_y P_z Q_xx Q_yy Q_zz Q_xy Q_xz Q_yz')

    plt.plot(z,multip.charge_dens,label='rho')
    plt.plot(z, multip.quadrapole[2,2,:],label='P_z')
    plt.plot(z,multip.dipole[2,:],label='Q_ZZ')
    plt.savefig('profs.png')
    plt.show()
    
    print('Finished', time.strftime("%a, %d %b %Y %H:%M:%S"))
