#       _                                   _              _               _               _  _              
#  ___ (_) _ __ ___     __ _  _ __    __ _ | | _   _  ___ (_) ___   _ __  (_) _ __    ___ | |(_) _ __    ___ 
# / __|| || '_ ` _ \   / _` || '_ \  / _` || || | | |/ __|| |/ __| | '_ \ | || '_ \  / _ \| || || '_ \  / _ \
# \__ \| || | | | | | | (_| || | | || (_| || || |_| |\__ \| |\__ \ | |_) || || |_) ||  __/| || || | | ||  __/
# |___/|_||_| |_| |_|  \__,_||_| |_| \__,_||_| \__, ||___/|_||___/ | .__/ |_|| .__/  \___||_||_||_| |_| \___|
#                                              |___/               |_|       |_|                             


# snapshots.py
# This file contains classes for different snapshot objects.
# Currently supported: x

# Import necessary packages
import os
import time
import h5py
import pickle
import numpy as np
import pandas as pd
import astropy.units as apy_units
import astropy.constants as apy_const
import astropy.cosmology as apy_cosmo
from scipy.spatial import cKDTree

# Import relevant functions from other files
from .tools import *
from .halofinder import *
from .analysis import *
from .bhinfo import *
from .plotting import *

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Define the Snapshot class
class gadget_snapshot_hki:

    """
    A class to represent a GADGET-3 snapshot from an idealised merger run from Helsinki.
    Tag: "gadget_snapshot_hki"

    Attributes:
    -----------
    snapshot_file: str
        The path to the snapshot file.
    snapshot_idx: int
        The index of the snapshot if provided.
    redshift: float
        The redshift of the snapshot.
    scalefac: float
        The scale factor of the snapshot.
    time: float
        The time of the snapshot.
    boxsize: float
        The size of the box in the snapshot.
    npart: list
        The number of particles in the snapshot.
    conversions: dict
        The conversions used to convert the code outputs to physical units.
    units: dict
        The units of the snapshot once conversions used -- ** might need to change for different simulations. **
    Om0: float
        The value of Omega_m from the adopted cosmology.
    hubble: float
        The value of H0/100 from the adopted cosmology.
    cosmology: astropy.cosmology
        The cosmology of the snapshot (from astropy).
    XH: float
        The mass fraction of hydrogen.
    XHe: float
        The mass fraction of helium.

    Methods:
    -----------
    get_particle_data(keys, types, center, radius)
        Returns the requested particle data in physical units.
    get_derived_field(key, type)
        Returns the requested derived field in physical units.
    list_particle_fields(ptype)
        Lists the available particle fields for a given particle type.
    sphere_mask(center, radius, ptype)
        Returns a mask for particles within a sphere of a given radius and center. 

        
    """
    def __init__(self, snapshot_file,snapshot_idx=None,cosmorun=False):
        self.snapshot_file = snapshot_file
        self.snapshot_idx = snapshot_idx
        with h5py.File(self.snapshot_file, 'r') as snapshot:
            self.cosmorun=cosmorun
            self.hubble=snapshot['Header'].attrs['HubbleParam']
            self.Om0=snapshot['Header'].attrs['Omega0']
            self.cosmology = apy_cosmo.FlatLambdaCDM(H0=self.hubble*100, Om0=self.Om0)
            self.timeraw=snapshot['Header'].attrs['Time']
            if not self.cosmorun:
                self.redshift=0 #not a cosmological simulation
                self.time=snapshot['Header'].attrs['Time']/0.72611
            else:
                self.redshift=snapshot['Header'].attrs['Redshift']
                self.time=self.cosmology.age(self.redshift).value

            self.scalefac=1/(1+self.redshift)
            self.boxsize=snapshot['Header'].attrs['BoxSize']
            self.npart=snapshot['Header'].attrs['NumPart_ThisFile']
            self.XH=0.76
            self.XHe=0.24

            #dm particle mass
            if 'MassTable' in snapshot['Header'].attrs.keys():
                mvalraw=snapshot['Header'].attrs['MassTable'][1]*1e10/self.hubble
                if mvalraw>0:
                    self.mass_dm=mvalraw
                else:
                    self.mass_dm=None

        snapshot.close()

        #these conversions take the GADGET outputs and give them in physical units (as defined below)
        self.conversions={'ParticleIDs': 1,
                          'Masses': 1e10/(self.hubble),
                          'Coordinates': 1/(self.hubble)*self.scalefac,
                          'Velocities': 1,
                          'Potential': 1,
                          'Density': 1.989e43*1/(self.hubble)/((3.08568e21/self.hubble)**3),
                          'StarFormationRate': 1,
                          'StellarFormationTime': 1/0.72611,
                          'CS Temperature':1,
                          'Metallicity': 1,
                          'InternalEnergy':1,
                          'ElectronAbundance':1}
        
        #these are the physical units of the fields in the snapshot once the conversions are applied
        self.units={}
        self.units["ParticleIDs"] = apy_units.Unit('1')
        self.units["Masses"] = apy_units.Unit('Msun')
        self.units["Coordinates"] = apy_units.Unit('kpc')
        self.units["Velocities"] = apy_units.Unit('km/s')
        self.units["Potential"] = apy_units.Unit('km**2/s**2')
        self.units["Density"] = apy_units.Unit('g/cm**3')
        self.units["StarFormationRate"] = apy_units.Unit('Msun/yr')
        self.units["StellarFormationTime"] = apy_units.Unit('Gyr')
        self.units["Metallicity"] = apy_units.Unit('1')
        self.units["InternalEnergy"] = apy_units.Unit('km**2/s**2')
        self.units["ElectronAbundance"] = apy_units.Unit('1')
        self.units["Temperature"] = apy_units.Unit('K')
        self.units["nH"] = apy_units.Unit('cm**-3')

        # these are the derived fields that are available and the particle type they are associated with
        self.derived_fields_available = ['Temperature', 'nH','Ztot']
        self.derived_fields_ptype={'Temperature': [0],'nH':[0],'Ztot':[0,4]}

       #initialize the haloes and galaxies lists
        self.haloes=[]
        self.galaxies=[]
    

        #initialise output directories
        if not os.path.exists('postprocessing'):
            os.makedirs('postprocessing')

    #simple method to list the available particle fields
    def list_particle_fields(self, ptype=0):
        with h5py.File(self.snapshot_file, 'r') as pfile:
            keys=list(pfile[f'PartType{ptype}'].keys())
            pfile.close()
        return keys
    
    #method to get requested field of particle data (and type) in physical units. return pandas dataframs with the requested field(s) in physical units with a field for the particle type. dynamically allocate memory for the dataframes to avoid memory issues. 
    def get_particle_data(self, keys=None, types=None, kdtree=None,center=None, radius=None, return_rrel=False):

        """
        Returns the requested particle data in physical units.

        Parameters:
        -----------
        keys: list
            The names of the fields to be returned.
        types: list
            The particle types to be returned.
        center: array 1x3 as astropy.units.Quantity
            The center of the sphere to be returned.
        radius: float as astropy.units.Quantity
            The radius of the sphere to be returned.
        
        Returns:
        -----------
        pd.DataFrame
            The requested particle data in physical units.

        """

        #if no keys are provided, print an error
        if not keys:
            print('Error: no keys provided')
            return None

        #if no types are provided, return all types
        if types is None:
            types = [0,1,4,5]
        elif isinstance(types, int):
            types = [types]

        #initialize the particle data dictionary
        particle_data = {type: {} for type in types}

        #open the snapshot file and get the requested fields
        with h5py.File(self.snapshot_file, 'r') as pfile:
            for ptype in types:
                print(f'Getting particle data for type {ptype}')
                try:
                    part = pfile[f'PartType{ptype}']
                except:
                    print(f'Note: Particle type {ptype} not found in snapshot')
                    particle_data[ptype] = pd.DataFrame()
                    continue

                #apply any spatial cuts
                if center is not None and radius is not None:
                    if kdtree is None:
                        #check if the KDTree file exists
                        if self.snapshot_idx is None:
                            snapfname=self.snapshot_file.split('/')[-1].split('.hdf5')[0]
                            kdpath=f'postprocessing/kdtrees/{snapfname}_kdtree.pkl'
                        else:
                            kdpath=f'postprocessing/kdtrees/snap_{str(self.snapshot_idx).zfill(3)}.pkl'
                        if os.path.exists(kdpath):
                            with open(kdpath,'rb') as kdfile:
                                kdtree=pickle.load(kdfile)
                            kdfile.close()
                            print('Found KDTree.')
                        else:
                            print('No KDTree found. Generating and caching (may take a while for a cosmo snapshot)...')
                            kdtree=make_particle_kdtree(self)
                            if not os.path.exists(kdpath.split('/')[0]):
                                os.makedirs(kdpath.split('/')[0])
                            with open(kdpath, 'wb') as kdfile:
                                pickle.dump(kdtree, kdfile)
                            return None
                    
                    mask,rrel=sphere_mask(snapshot=self, ptype=ptype, center=center, radius=radius, kdtree=kdtree, return_rrel=return_rrel)
                    if return_rrel:
                        particle_data[ptype]['R']=rrel
                else: 
                    mask=np.array(list(range(part['ParticleIDs'].shape[0])))

                num_particles = mask.shape[0]

                t0_load=time.time()
                if num_particles:
                    #iterate over the requested keys
                    for key in keys:
                        #if the key is available directly from file, get the data and apply the conversion
                        if key in part.keys():
                            particle_data[ptype][key] = part[key][:][mask]*self.conversions[key]
                            if len(particle_data[ptype][key].shape)==2 and particle_data[ptype][key].shape[1] == 3:
                                del particle_data[ptype][key]
                                particle_data[ptype][key+'_x'] = part[key][:][mask][:,0]*self.conversions[key]
                                particle_data[ptype][key+'_y'] = part[key][:][mask][:,1]*self.conversions[key]
                                particle_data[ptype][key+'_z'] = part[key][:][mask][:,2]*self.conversions[key]
                            elif len(particle_data[ptype][key].shape)==2:
                                del particle_data[ptype][key]
                                particle_data[ptype][key+f'_{str(0).zfill(2)}'] = part[key][:][mask][:,0]*self.conversions[key]
                        
                        #if the key is a derived field, get the data and apply the conversion
                        elif key in self.derived_fields_available and ptype in self.derived_fields_ptype[key]:
                            particle_data[ptype][key] = self.get_derived_field(key, ptype)[mask]

                        #if key is Masses and ptype is 1, return the mass_dm
                        elif key == 'Masses' and ptype == 1 and self.mass_dm is not None:
                            particle_data[ptype][key] = np.ones(num_particles)*self.mass_dm

                        #if the key is not available for this type, fill with NaNs
                        else:
                            particle_data[ptype][key]=np.zeros(num_particles)+np.nan
                
                    #add a column for the particle type
                    particle_data[ptype] = pd.DataFrame(particle_data[ptype])
                    particle_data[ptype]['ParticleTypes']=np.ones(num_particles)*ptype

                else:
                    particle_data[ptype] = pd.DataFrame()

            pfile.close()

        #stack the data into a pandas dataframe 
        particle_data = pd.concat([particle_data[ptype] for ptype in types])
        particle_data.reset_index(drop=True, inplace=True)

        return particle_data

    #method to get the derived fields in physical units
    def get_derived_field(self, key, type):
        """"
        Returns the requested derived field in physical units.

        Parameters:
        -----------
        key: str
            The name of the derived field.
        type: int
            The particle type of the derived field.
        
        Returns:
        -----------
        np.array
            The derived field in physical units.

        """
        
        #open the snapshot file and get the requested field
        with h5py.File(self.snapshot_file, 'r') as pfile:
            part = pfile[f'PartType{type}']

            ## TEMPERATURE DERIVED FIELD
            if key == 'Temperature':
                # calculate the temperature from the internal energy and electron abundance
                u = np.array(part['InternalEnergy'][:])*self.conversions['InternalEnergy']*self.units["InternalEnergy"]
                ne = np.array(part['ElectronAbundance'][:])*self.conversions['ElectronAbundance']*self.units["ElectronAbundance"]
                tmp = self.XH / 1.008 + self.XHe / 4.003
                av_m = apy_const.m_p*(tmp + ne * self.XH)
                return (u / (3 / 2.) * av_m / apy_const.k_B).to(self.units["Temperature"]).value
            
            ## nH DERIVED FIELD
            elif key == 'nH':
                # calculate the hydrogen number density from the density and the mean molecular weight
                rho = np.array(part['Density'][:])*self.conversions['Density']*self.units["Density"] #g/cm^3
                ne = np.array(part['ElectronAbundance'][:])*self.conversions['ElectronAbundance']*self.units["ElectronAbundance"]
                tmp = self.XH / 1.008 + self.XHe / 4.003
                av_m = (apy_const.m_p.value*apy_units.Unit('kg')*(tmp)).to(apy_units.Unit("g"))
                return (rho / av_m).to(self.units["nH"]).value
            
            elif key=='Ztot':
                #metals are: 0 = 3He, 1 = 12C, 2 = 24Mg, 3 = 16O, 4 = 56e, 5 = 28Si, 6 = H, 7 = 14N, 8 = 20Ne, 9 = 32S, 10 = 40Ca, 11 = 62Zn
                #return total metallicity subracting the hydrogen and helium mass fractions
                metals=(np.nansum(np.array(part['Metallicity'][:,:]),axis=1)-part['Metallicity'][:,0]-part['Metallicity'][:,6])/np.array(part['Masses'][:])*self.units["Metallicity"]
                return metals
            
            ## ERROR
            else:
                print(f'Error: key {key} not found for particle type', type)
                
            pfile.close()
            return None

    # Add method to render the snapshot
    def render_snap(self,type='baryons',frame=None,center=None,staralpha=1,clims=None):
        """
        Function to render the snapshot.

        Parameters:
        -----------
        type: str
            The type of particles to render -- either 'baryons', 'stars' or 'dm'.
        frame: str  
            The frame size (in kpc) to render the snapshot.
        center: array 1x3 as astropy.units.Quantity
            The center to be used for the rendering.
        staralpha: float
            The alpha value for the stars in the rendering if type='baryons'.
        clims: list
            The color limits for the rendering.

        """
        
        # Use the render_snap function to render the snapshot
        fig, ax=render_snap(self,type=type,frame=frame,center=center,staralpha=1,clims=clims)
        return fig, ax
        

# KDTrees for particle data
def make_particle_kdtree(snapshot):
    """
    Function to make a KDTree for the particle data.

    Parameters:
    -----------
    snapshot: gadget_cosmo_snapshot_hki
        The snapshot object to make the KDTree for.
    verbose: bool
        If True, print out the time taken to make the KDTree.

    Returns:
    -----------
    cKDTree
        The KDTree for the particle data.

    """
    #initialize the KDTree
    kdtree={}
    
    #get the particle types
    ptypes=[0,1,4,5]


    #file
    pdatafile=h5py.File(snapshot.snapshot_file, 'r')

    if 'PartType5' not in pdatafile.keys():
        ptypes.remove(5)

    #get the particle data
    for ptype in ptypes:
        print("Getting particle data for KDTree, type = ", ptype)
        coordinates=snapshot.get_particle_data(keys=['Coordinates'], types=ptype)
        coordinates=coordinates[['Coordinates_x','Coordinates_y','Coordinates_z']].values
        kdtree[ptype]=cKDTree(coordinates)

    #close the file
    pdatafile.close()

    return kdtree


def stack_kdtrees_worker(snaplist,iproc,verbose=False):
    """
    Function to stack the KDTree for particle data. Parallelised across snapshots.

    Parameters:
    -----------
    snaplist: list
        The list of snapshots to stack.
    ptype: int
        The particle type to stack.

    Returns:
    -----------
    cKDTree
        The stacked KDTree for the particle data.

    """
    
    #log file
    logfile=f'logs/kdtrees/iproc{str(iproc).zfill(3)}.log'
    if not os.path.exists('logs/kdtrees/'):
        os.makedirs('logs/kdtrees/')
    else:
        for file in os.listdir('logs/kdtrees/'):
            os.remove(f'logs/kdtrees/{file}')

    if os.path.exists(logfile):
        os.remove(logfile)

    logging.basicConfig(filename=logfile, level=logging.INFO)

    logging.info(f'')
    logging.info(f'************{datetime.now()}************')
    logging.info(f'')

    logging.info(f'===================================================================================================================')
    logging.info(f'Making KDtrees for snapshots {[snapshot.snapshot_idx for snapshot in snaplist]}... (iproc {iproc}) ...')
    logging.info(f'===================================================================================================================')
    logging.info(f'')


    if verbose:
        print(f'===================================================================================================================')
        print(f'Making KDtrees for snapshots {[snapshot.snapshot_idx for snapshot in snaplist]} (iproc {iproc}) ...')
        print(f'===================================================================================================================')
        print()

    t0=time.time()
    if not os.path.exists('outputs/kdtrees'):
        os.makedirs('outputs/kdtrees')

    for snapshot in snaplist:
        logging.info(f'Processing snapshot {snapshot.snapshot_idx}... [runtime {time.time()-t0:.2f} s]')
        kdtree=make_particle_kdtree(snapshot)

        with open(f'outputs/kdtrees/kdtree_{str(snapshot.snapshot_idx).zfill(3)}.pkl', 'wb') as kdfile:
            pickle.dump(kdtree, kdfile)
        kdfile.close()

    logging.info(f'Finished making KDtrees for snapshots {[snapshot.snapshot_idx for snapshot in snaplist]} [runtime {time.time()-t0:.2f} s]')


#mask for particles within a sphere of a given radius and center
def sphere_mask(snapshot, center, radius, kdtree, ptype, return_rrel=False):
    
    #if center is an np array, assume kpc and convert to apy
    if isinstance(center, np.ndarray):
        center = center*apy_units.kpc
    
    #if radius is a flat, assume kpc and convert to apy
    if isinstance(radius, (int, float)):
        radius = radius*apy_units.kpc

    #convert the center and radius to physical units
    if isinstance(center, apy_units.Quantity):
        center = center.to(snapshot.units["Coordinates"]).value
    if isinstance(radius, apy_units.Quantity):
        radius = radius.to(snapshot.units["Coordinates"]).value
    
    #get the KDTree for the particle data
    kdtree_ptype=kdtree[ptype]

    #find the particles within the radius
    idxs=kdtree_ptype.query_ball_point(x=center, r=radius)
    mask=np.array(idxs)

    #calculate the relative position
    if return_rrel and len(mask):
        rrel=np.linalg.norm(kdtree_ptype.data[mask]-center, axis=1)
    else:
        rrel=None
    return mask,rrel





# # Define the Snapshot class FOR COSMO RUNS FROM HELSINKI
# class gadget_cosmo_snapshot_hki:

#     """
#     A class to represent a GADGET-3 snapshot from an cosmo run from Helsinki.
#     Tag: "gadget_cosmo_snapshot_hki"

#     Attributes:
#     -----------
#     snapshot_file: str
#         The path to the snapshot file.
#     snapshot_idx: int
#         The index of the snapshot if provided.
#     redshift: float
#         The redshift of the snapshot.
#     scalefac: float
#         The scale factor of the snapshot.
#     time: float
#         The time of the snapshot.
#     boxsize: float
#         The size of the box in the snapshot.
#     npart: list
#         The number of particles in the snapshot.
#     conversions: dict
#         The conversions used to convert the code outputs to physical units.
#     units: dict
#         The units of the snapshot once conversions used -- ** might need to change for different simulations. **
#     Om0: float
#         The value of Omega_m from the adopted cosmology.
#     hubble: float
#         The value of H0/100 from the adopted cosmology.
#     cosmology: astropy.cosmology
#         The cosmology of the snapshot (from astropy).
#     XH: float
#         The mass fraction of hydrogen.
#     XHe: float
#         The mass fraction of helium.

#     Methods:
#     -----------
#     get_particle_data(keys, types, center, radius)
#         Returns the requested particle data in physical units.
#     get_derived_field(key, type)
#         Returns the requested derived field in physical units.
#     list_particle_fields(ptype)
#         Lists the available particle fields for a given particle type.
#     sphere_mask(center, radius, ptype)
#         Returns a mask for particles within a sphere of a given radius and center.
    

        
#     """
#     def __init__(self, snapshot_file,snapshot_idx=None):
#         self.snapshot_file = snapshot_file
#         self.snapshot_idx = snapshot_idx
#         with h5py.File(self.snapshot_file, 'r') as snapshot:
#             self.scalefac=snapshot['Header'].attrs['Time']
#             self.redshift=1/self.scalefac-1
#             self.boxsize=snapshot['Header'].attrs['BoxSize']
#             self.npart=snapshot['Header'].attrs['NumPart_ThisFile']
#             self.Om0=snapshot['Header'].attrs['Omega0']
#             self.hubble=snapshot['Header'].attrs['HubbleParam']
#             self.cosmology = apy_cosmo.FlatLambdaCDM(H0=self.hubble*100, Om0=self.Om0)
#             self.XH=0.76
#             self.XHe=0.24
#             self.mass_dm=snapshot['Header'].attrs['MassTable'][1]*1e10/self.hubble
#             self.cosmorun=True

#             #get time from cosmology and afac
#             self.time=self.cosmology.age(self.redshift).value

#         snapshot.close()



#         #these conversions take the GADGET outputs and give them in physical units (as defined below)
#         self.conversions={'ParticleIDs': 1,
#                           'Masses': 1e10/(self.hubble),
#                           'Coordinates': 1/(self.hubble)*self.scalefac,
#                           'Velocities': 1,
#                           'Potential': 1,
#                           'Density': 1.989e43*1/(self.hubble)/((3.08568e21/self.hubble)**3),
#                           'StarFormationRate': 1,
#                           'Metallicity': 1,
#                           'InternalEnergy':1,
#                           'ElectronAbundance':1}
        
#         #these are the physical units of the fields in the snapshot once the conversions are applied
#         self.units={}
#         self.units["ParticleIDs"] = apy_units.Unit('1')
#         self.units["Masses"] = apy_units.Unit('Msun')
#         self.units["Coordinates"] = apy_units.Unit('kpc')
#         self.units["Velocities"] = apy_units.Unit('km/s')
#         self.units["Potential"] = apy_units.Unit('km**2/s**2')
#         self.units["Density"] = apy_units.Unit('g/cm**3')
#         self.units["StarFormationRate"] = apy_units.Unit('Msun/yr')
#         self.units["Metallicity"] = apy_units.Unit('1')
#         self.units["InternalEnergy"] = apy_units.Unit('km**2/s**2')
#         self.units["ElectronAbundance"] = apy_units.Unit('1')
#         self.units["Temperature"] = apy_units.Unit('K')
#         self.units["nH"] = apy_units.Unit('cm**-3')

#         # these are the derived fields that are available and the particle type they are associated with
#         self.derived_fields_available = ['Temperature', 'nH','Metallicity']
#         self.derived_fields_ptype={'Temperature': [0],'nH':[0],'Metallicity':[0,4]}

#         self.haloes=[]
#         self.galaxies=[]
    
    
#     #simple method to list the available particle fields
#     def list_particle_fields(self, ptype=0):
#         with h5py.File(self.snapshot_file, 'r') as pfile:
#             keys=list(pfile[f'PartType{ptype}'].keys())
#             pfile.close()
#         return keys


#     #method to get requested field of particle data (and type) in physical units. return pandas dataframs with the requested field(s) in physical units with a field for the particle type. dynamically allocate memory for the dataframes to avoid memory issues. 
#     def get_particle_data(self, keys=None, types=None, kdtree=None,center=None, radius=None, return_rrel=False):

#         """
#         Returns the requested particle data in physical units.

#         Parameters:
#         -----------
#         keys: list
#             The names of the fields to be returned.
#         types: list
#             The particle types to be returned.
#         center: array 1x3 as astropy.units.Quantity
#             The center of the sphere to be returned.
#         radius: float as astropy.units.Quantity
#             The radius of the sphere to be returned.

#         Returns:
#         -----------
#         pd.DataFrame
#             The requested particle data in physical units.

#         """

#         #if no keys are provided, print an error
#         if not keys:
#             print('Error: no keys provided')
#             return None

#         #if no types are provided, return all types
#         if types is None:
#             types = [0,1,4,5]
#         elif isinstance(types, int):
#             types = [types]

#         #initialize the particle data dictionary
#         particle_data = {type: {} for type in types}

#         #open the snapshot file and get the requested fields
#         with h5py.File(self.snapshot_file, 'r') as pfile:
#             for ptype in types:
#                 print(f'Getting particle data for type {ptype}')
#                 try:
#                     part = pfile[f'PartType{ptype}']
#                 except:
#                     print(f'Note: Particle type {ptype} not found in snapshot')
#                     particle_data[ptype] = pd.DataFrame()
#                     continue

#                 #apply any spatial cuts
#                 if center is not None and radius is not None:
#                     if kdtree is None:
#                         #check if the KDTree file exists
#                         if self.snapshot_idx is None:
#                             snapfname=self.snapshot_file.split('/')[-1].split('.hdf5')[0]
#                             kdpath=f'kdtrees/{snapfname}_kdtree.pkl'
#                         else:
#                             kdpath=f'{self.snapshot_file.split("output")[0]+"/analysis/"}/outputs/kdtrees/snap_{str(self.snapshot_idx).zfill(3)}.pkl'
#                         if os.path.exists(kdpath):
#                             with open(kdpath,'rb') as kdfile:
#                                 kdtree=pickle.load(kdfile)
#                             kdfile.close()
#                             print('Found KDTree.')
#                         else:
#                             print('No KDTree found. Generating...')
#                             kdtree=make_particle_kdtree(self)
#                             if not os.path.exists(kdpath.split('/')[0]):
#                                 os.makedirs(kdpath.split('/')[0])
#                             with open(kdpath, 'wb') as kdfile:
#                                 pickle.dump(kdtree, kdfile)
#                             return None

#                     mask,rrel=sphere_mask(snapshot=self, ptype=ptype, center=center, radius=radius, kdtree=kdtree, return_rrel=return_rrel)
#                     if return_rrel:
#                         particle_data[ptype]['R']=rrel
#                 else:
#                     mask=np.where(np.ones(part['ParticleIDs'].shape[0]))[0]

#                 num_particles = len(mask)
#                 t0_load=time.time()

#                 if num_particles:
#                     #iterate over the requested keys
#                     for key in keys:
#                         #if the key is available directly from file, get the data and apply the conversion
#                         if key in part.keys():
#                             particle_data[ptype][key] = part[key][:][mask]*self.conversions[key]
#                             if len(particle_data[ptype][key].shape)==2 and particle_data[ptype][key].shape[1] == 3:
#                                 del particle_data[ptype][key]
#                                 particle_data[ptype][key+'_x'] = part[key][:][mask][:,0]*self.conversions[key]
#                                 particle_data[ptype][key+'_y'] = part[key][:][mask][:,1]*self.conversions[key]
#                                 particle_data[ptype][key+'_z'] = part[key][:][mask][:,2]*self.conversions[key]
#                             elif len(particle_data[ptype][key].shape)==2:
#                                 del particle_data[ptype][key]
#                                 particle_data[ptype][key+f'_{str(0).zfill(2)}'] = part[key][:][mask][:,0]*self.conversions[key]
                        
#                         #if the key is a derived field, get the data and apply the conversion
#                         elif key in self.derived_fields_available and ptype in self.derived_fields_ptype[key]:
#                             particle_data[ptype][key] = self.get_derived_field(key, ptype)[mask]

#                         #if key is Masses and ptype is 1, return the mass_dm
#                         elif key == 'Masses' and ptype == 1 and self.mass_dm is not None:
#                             particle_data[ptype][key] = np.ones(num_particles)*self.mass_dm

#                         #if the key is not available for this type, fill with NaNs
#                         else:
#                             particle_data[ptype][key]=np.zeros(num_particles)+np.nan
                
#                     #add a column for the particle type
#                     particle_data[ptype] = pd.DataFrame(particle_data[ptype])
#                     particle_data[ptype]['ParticleTypes']=np.ones(num_particles)*ptype
#                 else:
#                     particle_data[ptype] = pd.DataFrame()

#             pfile.close()

#         #stack the data into a pandas dataframe 
#         particle_data = pd.concat([particle_data[ptype] for ptype in types])
#         particle_data.reset_index(drop=True, inplace=True)

#         return particle_data

#     #method to get the derived fields in physical units
#     def get_derived_field(self, key, type):
#         """"
#         Returns the requested derived field in physical units.

#         Parameters:
#         -----------
#         key: str
#             The name of the derived field.
#         type: int
#             The particle type of the derived field.
        
#         Returns:
#         -----------
#         np.array
#             The derived field in physical units.

#         """
        
#         #open the snapshot file and get the requested field
#         with h5py.File(self.snapshot_file, 'r') as pfile:
#             part = pfile[f'PartType{type}']

#             ## TEMPERATURE DERIVED FIELD
#             if key == 'Temperature':
#                 # calculate the temperature from the internal energy and electron abundance
#                 u = np.array(part['InternalEnergy'][:])*self.conversions['InternalEnergy']*self.units["InternalEnergy"]
#                 ne = np.array(part['ElectronAbundance'][:])*self.conversions['ElectronAbundance']*self.units["ElectronAbundance"]
#                 tmp = self.XH / 1.008 + self.XHe / 4.003
#                 av_m = apy_const.m_p*(tmp + ne * self.XH)
#                 return (u / (3 / 2.) * av_m / apy_const.k_B).to(self.units["Temperature"]).value
            
#             ## nH DERIVED FIELD
#             elif key == 'nH':
#                 # calculate the hydrogen number density from the density and the mean molecular weight
#                 rho = np.array(part['Density'][:])*self.conversions['Density']*self.units["Density"] #g/cm^3
#                 ne = np.array(part['ElectronAbundance'][:])*self.conversions['ElectronAbundance']*self.units["ElectronAbundance"]
#                 tmp = self.XH / 1.008 + self.XHe / 4.003
#                 av_m = (apy_const.m_p.value*apy_units.Unit('kg')*(tmp + ne * self.XH)).to(apy_units.Unit("g"))
#                 return (rho / av_m).to(self.units["nH"]).value
            
#             elif key=='Metallicity':
#                 #metals are: 0 = 3He, 1 = 12C, 2 = 24Mg, 3 = 16O, 4 = 56e, 5 = 28Si, 6 = H, 7 = 14N, 8 = 20Ne, 9 = 32S, 10 = 40Ca, 11 = 62Zn
#                 #return total metallicity subracting the hydrogen and helium mass fractions
#                 metals=(np.nansum(np.array(part['Metallicity'][:,:]),axis=1)-part['Metallicity'][:,0]-part['Metallicity'][:,6])/np.array(part['Masses'][:])*self.units["Metallicity"]
#                 return metals
            
#             ## ERROR
#             else:
#                 print(f'Error: key {key} not found for particle type', type)
                
#             pfile.close()
#             return None


#     # Add method to render the snapshot
#     def render_snap(self,type='baryons',frame=None,center=None,staralpha=1,clims=None):
#         """
#         Function to render the snapshot.

#         Parameters:
#         -----------
#         type: str
#             The type of particles to render -- either 'baryons', 'stars' or 'dm'.
#         frame: str  
#             The frame size (in kpc) to render the snapshot.
#         center: array 1x3 as astropy.units.Quantity
#             The center to be used for the rendering.
#         staralpha: float
#             The alpha value for the stars in the rendering if type='baryons'.
#         clims: list
#             The color limits for the rendering.

#         """
        
#         # Use the render_snap function to render the snapshot
#         fig, ax=render_snap(self,type=type,frame=frame,center=center,staralpha=1,clims=clims)
#         return fig, ax
        

