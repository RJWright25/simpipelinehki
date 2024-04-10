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
import numpy as np
import pandas as pd
import astropy.units as apy_units
import astropy.constants as apy_const
import astropy.cosmology as apy_cosmo
import multiprocessing

# Import relevant functions from other files
from .tools import *
from .halofinder import *
from .analysis import *
from .bhinfo import *
from .plotting import *

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Define the Snapshot class FOR IDEALISED MERGER RUNS FROM HELSINKI
class gadget_idealised_snapshot_hki:

    """
    A class to represent a GADGET-3 snapshot from an idealised merger run from Helsinki.
    Tag: "gadget_idealised_snapshot_hki"

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
    get_particle_data(keys, types, center, radius, subsample)
        Returns the requested particle data in physical units.
    get_derived_field(key, type)
        Returns the requested derived field in physical units.
    list_particle_fields(ptype)
        Lists the available particle fields for a given particle type.
    sphere_mask(center, radius, ptype)
        Returns a mask for particles within a sphere of a given radius and center. 

        
    """
    def __init__(self, snapshot_file,snapshot_idx=None):
        self.snapshot_file = snapshot_file
        self.snapshot_idx = snapshot_idx
        with h5py.File(self.snapshot_file, 'r') as snapshot:
            self.redshift=0 #not a cosmological simulation
            self.scalefac=1/(1+self.redshift)
            self.time=snapshot['Header'].attrs['Time']
            self.boxsize=snapshot['Header'].attrs['BoxSize']
            self.npart=snapshot['Header'].attrs['NumPart_ThisFile']
            self.Om0=snapshot['Header'].attrs['Omega0']
            self.hubble=snapshot['Header'].attrs['HubbleParam']
            self.cosmology = apy_cosmo.FlatLambdaCDM(H0=self.hubble*100, Om0=self.Om0)
            self.mass_dm=None#only for cosmological sims
            self.cosmorun=False
            self.XH=0.76
            self.XHe=0.24
        snapshot.close()

        self.haloes=[]
        self.galaxies=[]

        #these conversions take the GADGET outputs and give them in physical units (as defined below)
        self.conversions={'ParticleIDs': 1,
                          'Masses': 1e10/(self.hubble),
                          'Coordinates': 1/(self.hubble),
                          'Velocities': 1,
                          'Potential': 1,
                          'Density': 1.989e43*1/(self.hubble)/((3.08568e21/self.hubble)**3),
                          'StarFormationRate': 1,
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
        self.units["Metallicity"] = apy_units.Unit('1')
        self.units["InternalEnergy"] = apy_units.Unit('km**2/s**2')
        self.units["ElectronAbundance"] = apy_units.Unit('1')
        self.units["Temperature"] = apy_units.Unit('K')
        self.units["nH"] = apy_units.Unit('cm**-3')

        # these are the derived fields that are available and the particle type they are associated with
        self.derived_fields_available = ['Temperature', 'nH']
        self.derived_fields_ptype={'Temperature': 0,'nH':0}


    
    #simple method to list the available particle fields
    def list_particle_fields(self, ptype=0):
        with h5py.File(self.snapshot_file, 'r') as pfile:
            keys=list(pfile[f'PartType{ptype}'].keys())
            pfile.close()
        return keys
    
    #method to return a mask for particles within a sphere of a given radius and center
    def sphere_mask(self, center, radius, ptype=0):
        center = center.to(self.units["Coordinates"]).value

        with h5py.File(self.snapshot_file, 'r') as pfile:
            rrel = pfile[f'PartType{ptype}']['Coordinates'][:]*self.conversions['Coordinates']-center
            pfile.close()
        
        rrel=np.sqrt(np.sum(rrel**2, axis=1))
        rargsorted=np.argsort(rrel)
        rsorted=rrel[rargsorted]
        lastidx=np.searchsorted(rsorted, radius)
        mask=np.zeros(rrel.shape[0], dtype=bool)
        mask[rargsorted[:lastidx]]=True
        return mask
    
    #method to get requested field of particle data (and type) in physical units. return pandas dataframs with the requested field(s) in physical units with a field for the particle type. dynamically allocate memory for the dataframes to avoid memory issues. 
    def get_particle_data(self, keys=None, types=None, center=None, radius=None,subsample=1):

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
        subsample: int
            The subsampling factor for the particle data.
        
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
        if not types:
            types = [0,1,4,5]
        if not isinstance(types, list):
            types = [types]

        #initialize the particle data dictionary
        particle_data = {type: {} for type in types}

        #open the snapshot file and get the requested fields
        with h5py.File(self.snapshot_file, 'r') as pfile:
            for ptype in types:
                try:
                    part = pfile[f'PartType{ptype}']
                except:
                    print(f'Note: Particle type {ptype} not found in snapshot')
                    particle_data[ptype] = pd.DataFrame()
                    continue

                #apply any spatial cuts
                mask=np.ones(part['ParticleIDs'].shape[0], dtype=bool)
                if center is not None and radius is not None:
                    mask=self.sphere_mask(center, radius, ptype)
                num_particles = np.sum(mask)
        
                #iterate over the requested keys
                for key in keys:
                    #if the key is available directly from file, get the data and apply the conversion
                    raw=part[key][:][mask]
                    print(f'key: {key}, shape: {raw.shape}')
                    print(raw)
                    
                    if key in part.keys():
                        particle_data[ptype][key] = part[key][:][mask]*self.conversions[key]
                        if len(particle_data[ptype][key].shape)==2 and particle_data[ptype][key].shape[1] == 3:
                            del particle_data[ptype][key]
                            particle_data[ptype][key+'_x'] = part[key][:][mask][:,0][::subsample]*self.conversions[key]
                            particle_data[ptype][key+'_y'] = part[key][:][mask][:,1][::subsample]*self.conversions[key]
                            particle_data[ptype][key+'_z'] = part[key][:][mask][:,2][::subsample]*self.conversions[key]
                        elif len(particle_data[ptype][key].shape)==2:
                            del particle_data[ptype][key]
                            particle_data[ptype][key+f'_{str(0).zfill(2)}'] = part[key][:][mask][:,0][::subsample]*self.conversions[key]
                    
                    #if the key is a derived field, get the data and apply the conversion
                    elif key in self.derived_fields_available and ptype == self.derived_fields_ptype[key]:
                        particle_data[ptype][key] = self.get_derived_field(key, ptype)[mask][::subsample]

                    #if the key is not available for this type, fill with NaNs
                    else:
                        print(f'Error: key {key} not found for particle type', ptype)
                        particle_data[ptype][key]=np.zeros(num_particles)[::subsample]+np.nan
                
                #add a column for the particle type
                particle_data[ptype] = pd.DataFrame(particle_data[ptype])
                particle_data[ptype]['ParticleTypes']=np.ones(num_particles)[::subsample]*ptype

            pfile.close()

        #stack the data into a pandas dataframe 
        particle_data = pd.concat([particle_data[ptype] for ptype in types])
        particle_data.reset_index(drop=True, inplace=True)

        #add R column if center is provided (NB: coordinates not recentered)
        if center is not None:
            center = center.to(self.units["Coordinates"]).value
            particle_data['R'] = np.sqrt((particle_data['Coordinates_x'].values-center[0])**2 + (particle_data['Coordinates_y'].values-center[1])**2 + (particle_data['Coordinates_z'].values-center[2])**2)

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
            elif key == 'nH' and type == 0:
                # calculate the hydrogen number density from the density and the mean molecular weight
                rho = np.array(part['Density'][:])*self.conversions['Density']*self.units["Density"] #g/cm^3
                ne = np.array(part['ElectronAbundance'][:])*self.conversions['ElectronAbundance']*self.units["ElectronAbundance"]
                tmp = self.XH / 1.008 + self.XHe / 4.003
                av_m = (apy_const.m_p.value*apy_units.Unit('kg')*(tmp + ne * self.XH)).to(apy_units.Unit("g"))
                return (rho / av_m).to(self.units["nH"]).value
            
            ## ERROR
            else:
                print(f'Error: key {key} not found for particle type', type)
                
            pfile.close()
            return None


    # Add method to render the snapshot
    def render_snap(self,type='baryons',frame=300,galaxies=None,useminpot=False,subsample=1,verbose=False):
        """
        Function to render the snapshot.

        Parameters:
        -----------
        type: str
            The type of particles to render -- either 'baryons' or 'dm'.
        frame: str  
            The frame size (in kpc) to render the snapshot.
        galaxies: pd.DataFrame
            The dataframe of galaxies from the halo finder (optional). 
            If provided, the galaxy centres will be rendered as well.
        useminpot: bool
            If True (and galaxies provided), use the minimum potential to centre the galaxies.
        subsample: int
            The subsampling factor for the particle data.
        verbose: bool
            If True, print out the time taken to render the snapshot.
        
        """
        
        # Use the render_snap function to render the snapshot
        fig, ax=render_snap(self,type=type,frame=frame,galaxies=galaxies,useminpot=useminpot,subsample=subsample,verbose=verbose)
        return fig, ax
        





# Define the Snapshot class FOR COSMO RUNS FROM HELSINKI
class gadget_cosmo_snapshot_hki:

    """
    A class to represent a GADGET-3 snapshot from an cosmo run from Helsinki.
    Tag: "gadget_cosmo_snapshot_hki"

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
    get_particle_data(keys, types, center, radius, subsample)
        Returns the requested particle data in physical units.
    get_derived_field(key, type)
        Returns the requested derived field in physical units.
    list_particle_fields(ptype)
        Lists the available particle fields for a given particle type.
    sphere_mask(center, radius, ptype)
        Returns a mask for particles within a sphere of a given radius and center.
    

        
    """
    def __init__(self, snapshot_file,snapshot_idx=None):
        self.snapshot_file = snapshot_file
        self.snapshot_idx = snapshot_idx
        with h5py.File(self.snapshot_file, 'r') as snapshot:
            self.scalefac=snapshot['Header'].attrs['Time']
            self.redshift=1/self.scalefac-1
            self.boxsize=snapshot['Header'].attrs['BoxSize']
            self.npart=snapshot['Header'].attrs['NumPart_ThisFile']
            self.Om0=snapshot['Header'].attrs['Omega0']
            self.hubble=snapshot['Header'].attrs['HubbleParam']
            self.cosmology = apy_cosmo.FlatLambdaCDM(H0=self.hubble*100, Om0=self.Om0)
            self.XH=0.76
            self.XHe=0.24
            self.mass_dm=snapshot['Header'].attrs['MassTable'][1]*1e10/self.hubble
            self.cosmorun=True
            #get time from cosmology and afac
            self.time=self.cosmology.age(self.redshift).value

        snapshot.close()

        self.haloes=[]
        self.galaxies=[]

        #these conversions take the GADGET outputs and give them in physical units (as defined below)
        self.conversions={'ParticleIDs': 1,
                          'Masses': 1e10/(self.hubble),
                          'Coordinates': 1/(self.hubble)*self.scalefac,
                          'Velocities': 1,
                          'Potential': 1,
                          'Density': 1.989e43*1/(self.hubble)/((3.08568e21/self.hubble)**3),
                          'StarFormationRate': 1,
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
        self.units["Metallicity"] = apy_units.Unit('1')
        self.units["InternalEnergy"] = apy_units.Unit('km**2/s**2')
        self.units["ElectronAbundance"] = apy_units.Unit('1')
        self.units["Temperature"] = apy_units.Unit('K')
        self.units["nH"] = apy_units.Unit('cm**-3')

        # these are the derived fields that are available and the particle type they are associated with
        self.derived_fields_available = ['Temperature', 'nH']
        self.derived_fields_ptype={'Temperature': 0,'nH':0}

    
    #simple method to list the available particle fields
    def list_particle_fields(self, ptype=0):
        with h5py.File(self.snapshot_file, 'r') as pfile:
            keys=list(pfile[f'PartType{ptype}'].keys())
            pfile.close()
        return keys
    
    #method to return a mask for particles within a sphere of a given radius and center
    def sphere_mask(self, center, radius, ptype=0):
        center = center.to(self.units["Coordinates"]).value

        with h5py.File(self.snapshot_file, 'r') as pfile:
            rrel = pfile[f'PartType{ptype}']['Coordinates'][:]*self.conversions['Coordinates']-center
            pfile.close()
        
        rrel=np.sqrt(np.sum(rrel**2, axis=1))
        rargsorted=np.argsort(rrel)
        rsorted=rrel[rargsorted]
        lastidx=np.searchsorted(rsorted, radius)
        mask=np.zeros(rrel.shape[0], dtype=bool)
        mask[rargsorted[:lastidx]]=True
        return mask
    
    #method to get requested field of particle data (and type) in physical units. return pandas dataframs with the requested field(s) in physical units with a field for the particle type. dynamically allocate memory for the dataframes to avoid memory issues. 
    def get_particle_data(self, keys=None, types=None, center=None, radius=None,subsample=1,verbose=False):

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
        subsample: int
            The subsampling factor for the particle data.
        
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
        if not types:
            types = [0,1,4,5]
        if not isinstance(types, list):
            types = [types]

        #initialize the particle data dictionary
        particle_data = {type: {} for type in types}

        #open the snapshot file and get the requested fields
        with h5py.File(self.snapshot_file, 'r') as pfile:
            for ptype in types:
                try:
                    part = pfile[f'PartType{ptype}']
                except:
                    print(f'Note: Particle type {ptype} not found in snapshot')
                    particle_data[ptype] = pd.DataFrame()
                    continue

                #apply any spatial cuts
                mask=np.ones(part['ParticleIDs'].shape[0], dtype=bool)
                if center is not None and radius is not None:
                    mask=self.sphere_mask(center, radius, ptype)
                num_particles = np.sum(mask)
        
                #iterate over the requested keys
                for key in keys:
                    #if the key is available directly from file, get the data and apply the conversion
                    if key in part.keys():
                        particle_data[ptype][key] = part[key][:][mask]*self.conversions[key]
                        if len(particle_data[ptype][key].shape)==2 and particle_data[ptype][key].shape[1] == 3:
                            del particle_data[ptype][key]
                            particle_data[ptype][key+'_x'] = part[key][:][mask][:,0][::subsample]*self.conversions[key]
                            particle_data[ptype][key+'_y'] = part[key][:][mask][:,1][::subsample]*self.conversions[key]
                            particle_data[ptype][key+'_z'] = part[key][:][mask][:,2][::subsample]*self.conversions[key]
                        elif len(particle_data[ptype][key].shape)==2:
                            del particle_data[ptype][key]
                            particle_data[ptype][key+f'_{str(0).zfill(2)}'] = part[key][:][mask][:,0][::subsample]*self.conversions[key]
                    
                    #if the key is a derived field, get the data and apply the conversion
                    elif key in self.derived_fields_available and ptype == self.derived_fields_ptype[key]:
                        particle_data[ptype][key] = self.get_derived_field(key, ptype)[mask][::subsample]

                    #if key is Masses and ptype is 1, return the mass_dm
                    elif key == 'Masses' and ptype == 1:
                        particle_data[ptype][key] = np.ones(num_particles)[::subsample]*self.mass_dm

                    #if the key is not available for this type, fill with NaNs
                    else:
                        if verbose:
                            print(f'Error: key {key} not found for particle type', ptype)
                        particle_data[ptype][key]=np.zeros(num_particles)[::subsample]+np.nan
                
                #add a column for the particle type
                particle_data[ptype] = pd.DataFrame(particle_data[ptype])
                particle_data[ptype]['ParticleTypes']=np.ones(num_particles)[::subsample]*ptype

            pfile.close()

        #stack the data into a pandas dataframe 
        particle_data = pd.concat([particle_data[ptype] for ptype in types])
        particle_data.reset_index(drop=True, inplace=True)

        #add R column if center is provided (NB: coordinates not recentered)
        if center is not None:
            center = center.to(self.units["Coordinates"]).value
            particle_data['R'] = np.sqrt((particle_data['Coordinates_x'].values-center[0])**2 + (particle_data['Coordinates_y'].values-center[1])**2 + (particle_data['Coordinates_z'].values-center[2])**2)

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
            elif key == 'nH' and type == 0:
                # calculate the hydrogen number density from the density and the mean molecular weight
                rho = np.array(part['Density'][:])*self.conversions['Density']*self.units["Density"] #g/cm^3
                ne = np.array(part['ElectronAbundance'][:])*self.conversions['ElectronAbundance']*self.units["ElectronAbundance"]
                tmp = self.XH / 1.008 + self.XHe / 4.003
                av_m = (apy_const.m_p.value*apy_units.Unit('kg')*(tmp + ne * self.XH)).to(apy_units.Unit("g"))
                return (rho / av_m).to(self.units["nH"]).value
            
            ## ERROR
            else:
                print(f'Error: key {key} not found for particle type', type)
                
            pfile.close()
            return None


    # Add method to render the snapshot
    def render_snap(self,type='baryons',frame=None,galaxies=None,useminpot=False,subsample=1,verbose=False):
        """
        Function to render the snapshot.

        Parameters:
        -----------
        type: str
            The type of particles to render -- either 'baryons' or 'dm'.
        frame: str  
            The frame size (in kpc) to render the snapshot.
        galaxies: pd.DataFrame
            The dataframe of galaxies from the halo finder (optional). 
            If provided, the galaxy centres will be rendered as well.
        useminpot: bool
            If True (and galaxies provided), use the minimum potential to centre the galaxies.
        subsample: int
            The subsampling factor for the particle data.
        verbose: bool
            If True, print out the time taken to render the snapshot.
        
        """
        
        # Use the render_snap function to render the snapshot
        fig, ax=render_snap(self,type=type,frame=frame,galaxies=galaxies,useminpot=useminpot,subsample=subsample,verbose=verbose)
        return fig, ax
        