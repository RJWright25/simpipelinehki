#       _                                   _              _               _               _  _              
#  ___ (_) _ __ ___     __ _  _ __    __ _ | | _   _  ___ (_) ___   _ __  (_) _ __    ___ | |(_) _ __    ___ 
# / __|| || '_ ` _ \   / _` || '_ \  / _` || || | | |/ __|| |/ __| | '_ \ | || '_ \  / _ \| || || '_ \  / _ \
# \__ \| || | | | | | | (_| || | | || (_| || || |_| |\__ \| |\__ \ | |_) || || |_) ||  __/| || || | | ||  __/
# |___/|_||_| |_| |_|  \__,_||_| |_| \__,_||_| \__, ||___/|_||___/ | .__/ |_|| .__/  \___||_||_||_| |_| \___|
#                                              |___/               |_|       |_|                             


# simlations.py
# This file contains the simulation class.

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
import multiprocessing

# Import relevant functions from other files
from .tools import *
from .halofinder import *
from .analysis import *
from .bhinfo import *
from .snapshots import *
from .plotting import *
from .groupfinder import basic_groupfinder

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Define the Simulation class (read: a collection of snapshots)
class gadget_simulation:

    """
    A class to represent a collection of GADGET-3 snapshots.

    Parameters:
    -----------
    snapshot_file_list: list
        The list of paths to the snapshot files.

    snapshot_type: str
        The type of snapshot to use (e.g. 'gadget_idealised_snapshot_hki' or 'gadget_cosmo_snapshot_hki').



    Attributes:
    -----------
    snapshot_file_list: list
        The list of paths to the snapshot files.
    snapshots: list
        The list of snapshot objects.
    snapshot_idxlist: list
        The list of snapshot indices.
    timelist: list
        The list of times of the snapshots.
    redshiftlist: list
        The list of redshifts of the snapshots.
    hubble: float
        The value of H0/100 from the adopted cosmology.
    cosmology: astropy.cosmology
        The adopted cosmology.

    """

    # Initialize the simulation object, take a list of snapshot files and create a list of snapshot objects
    def __init__(self, snapshot_file_list, snapshot_type=None):
        
        if snapshot_type is None:
            snapshot_type = gadget_idealised_snapshot_hki
            self.snapshot_type = 'idealised'
        elif snapshot_type=='gadget_idealised_snapshot_hki':
            snapshot_type = gadget_idealised_snapshot_hki
            self.snapshot_type = 'idealised'
        elif snapshot_type=='gadget_cosmo_snapshot_hki':
            snapshot_type = gadget_cosmo_snapshot_hki
            self.snapshot_type = 'cosmo'
        else:
            print('Error: snapshot type not recognized.')
            return None


        self.snapshot_flist = snapshot_file_list;times=[h5py.File(snapshot_file, 'r')['Header'].attrs['Time'] for snapshot_file in self.snapshot_flist]
        self.snapshot_flist = [snapshot_file for _,snapshot_file in sorted(zip(times,self.snapshot_flist))]
        self.snapshots = [snapshot_type(snapshot_file,snapshot_idx=snapshot_idx) for snapshot_idx,snapshot_file in enumerate(self.snapshot_flist)]
        self.snapshot_idxlist = [snapshot.snapshot_idx for snapshot in self.snapshots]
        self.timelist = [snapshot.time for snapshot in self.snapshots]
        self.redshiftlist = [snapshot.redshift for snapshot in self.snapshots]
        self.hubble = self.snapshots[0].hubble
        self.cosmology = self.snapshots[0].cosmology

    # Method to get a snapshot by index or redshift
    def get_snapshot(self, time=None, redshift=None):

        """
        Returns the requested snapshot by index or redshift.

        Parameters:
        -----------
        time: float
            The time of the requested snapshot.
        redshift: float
            The redshift of the requested snapshot.

        Returns:
        -----------
        snapshot : snapshot class
            The requested snapshot.

        """
        #if time is provided, find the snapshot with the closest time
        if time is not None:
            idx = np.argmin(np.abs(np.array(self.timelist)-time))
        #if redshift is provided, find the snapshot with the closest redshift
        elif redshift is not None:
            idx = np.argmin(np.abs(np.array(self.redshiftlist)-redshift))
        else:
            print('Error: must provide either time or redshift')
            return None
        return self.snapshots[idx]
    
    
    # Method to load the black hole details from a directory
    def load_bhdata(self,path=None,bhids=None,subsample=1):
            
        """
        Load the black hole details from a file.

        Parameters:
        -----------
        path: str
            The path to the directory containing the black hole details files.
        bhids: list
            The IDs of the black hole to read (may not be all bhs).
        subsample: int
            The subsampling factor to use when reading the data.

        Returns:
        -----------
        bhdata : dict
            A dictionary containing a pandas dataframe of the data for each black hole.
            Keys are the black hole IDs.

        """

        if not path:
            path=self.snapshots[-1].snapshot_file.split('/')[:-1]
            path='/'.join(path)+'/blackhole_details_post_processing/'
        if not os.path.exists(path):
            print('Error: path does not exist.')
            return None        
        
        # use the read_bhdata function from bhprocessing.py to read the black hole details
        bhdata= read_bhdata(self,path=path,bhids=bhids,subsample=subsample)

        # save the black hole details to the simulation object
        self.bhdetails=bhdata

        return bhdata
    
    def load_ketjubhdata(self, path=None):

        """
        Read the KETJU black hole details from a directory using the KETJU GW package.

        Parameters:
        -----------
        path: str
            The path to the directory containing the black hole details files.

        Returns:
        -----------
        ketjubhs : dict of IDs 
            The black hole data as ketjugw.particle objects.
        ketjubinaries : dict of ID pairs
            The binary orbital params from ketjugw.binary objects.

        """

        ketjubhs,ketjubinaries=read_ketjubhdata(self,path=path)

        self.ketjubhs=ketjubhs
        self.ketjubinaries=ketjubinaries

        return ketjubhs,ketjubinaries


    # Method to find haloes in all snapshots using multiprocessing
    def find_haloes(self,numproc=1,delta=200,useminpot=False,verbose=False):
        
        """
        Find haloes in all snapshots using multiprocessing.

        Parameters:
        -----------
        numproc: int
            The number of processes to use.
        delta: float
            The overdensity criteria for the halo finder.
        useminpot: bool
            If True, use the minimum potential of the star particles as the halo centre.
        verbose: bool
            If True, print the progress of the halo finding.

        Returns:
        -----------
        haloes : pd.DataFrame
            The halo data (see halofinder.py for details).

        """

        print()
        print(f'===========================================================================================')
        print(f'Finding haloes in {len(self.snapshots)} snapshots using {numproc} processes...')
        print(f'===========================================================================================')
        print()

        t0stack=time.time()

        #make a temporary directory for the outputs
        if not os.path.exists(os.getcwd()+'/tmphalo/'):
            os.mkdir(os.getcwd()+'/tmphalo/')
        else:
            for fname in os.listdir(os.getcwd()+'/tmphalo/'):
                if os.path.exists(os.getcwd()+'/tmphalo/'+fname):
                    os.remove(os.getcwd()+'/tmphalo/'+fname)

        #split the snapshots into chunks for multiprocessing
        snapshot_list=self.snapshots
        snapshot_chunks=split_list(snapshot_list,numproc)

        procs=[]
        for iproc in range(numproc):
            snapshots_ichunk=snapshot_chunks[iproc]
            if verbose:
                print(f'Process {iproc} getting snaps: ', [snapshot.snapshot_idx for snapshot in snapshots_ichunk])
            # instantiating process with arguments
            proc = multiprocessing.Process(target=stack_haloes_worker, args=(snapshots_ichunk,iproc,delta,useminpot,verbose))
            procs.append(proc)
            proc.start()

        #complete the processes
        for proc in procs:
            proc.join()
        time.sleep(1)

        #load in outputs and save
        print()
        print('Consolidating halo outputs...')
        chunk_fnames=[os.getcwd()+'/tmphalo/'+file for file in os.listdir(os.getcwd()+'/tmphalo/')]
        chunk_dfs=[pd.read_hdf(fname,key='chunk') for fname in chunk_fnames]
        haloes=pd.concat(chunk_dfs)
        haloes.sort_values(by=['Time','ID'],ascending=[True,True],inplace=True)
        haloes.reset_index(drop=True,inplace=True)

        print()
        print(f'----> Halo finding for {len(self.snapshots)} snaps complete in {time.time()-t0stack:.2f} seconds.')

        self.haloes=haloes
        return haloes
        
    # Method to analyse galaxies in all snapshots using multiprocessing
    def analyse_galaxies(self,numproc=1,shells_kpc=None,useminpot=False,rfac_offset=0.1,groupfinder=True,verbose=False):
        """
        
        Analyse galaxies in all snapshots using multiprocessing.

        Parameters:
        -----------
        numproc: int
            The number of processes to use.
        shells_kpc: list (of floats, in kpc)
            The radii of the shells to use for the galaxy analysis.
        useminpot: bool
            If True, use the minimum potential of the star particles as the galaxy centre.
        rfac_offset: float
            Fractional value of sphere/shell radius to identify relevant particles.
        verbose: bool
            If True, print the progress of the galaxy analysis.

        Returns:
        -----------
        galaxies : pd.DataFrame
            The galaxy data (see galaxyanalysis.py for details).
        
        """

        print()
        print(f'===========================================================================================')
        print(f'Analysing galaxies in {len(self.snapshots)} snapshots using {numproc} processes...')
        print(f'===========================================================================================')
        print()


        t0stack=time.time()

        #make a temporary directory for the outputs
        if not os.path.exists(os.getcwd()+'/tmpgalx/'):
            os.mkdir(os.getcwd()+'/tmpgalx/')
        else:
            for fname in os.listdir(os.getcwd()+'/tmpgalx/'):
                if os.path.exists(os.getcwd()+'/tmpgalx/'+fname):
                    os.remove(os.getcwd()+'/tmpgalx/'+fname)
        
        #split the snapshots into chunks for multiprocessing
        snapshot_list=self.snapshots
        snapshot_chunks=split_list(snapshot_list,numproc)
        haloes=self.haloes

        #start the processes
        procs=[]
        for iproc in range(numproc):
            time.sleep(0.1)
            snapshots_ichunk=snapshot_chunks[iproc]
            if verbose:
                print(f'Process {iproc} getting snaps: ', [snapshot.snapshot_idx for snapshot in snapshots_ichunk])
            proc = multiprocessing.Process(target=stack_galaxies_worker, args=(snapshots_ichunk,haloes,iproc,shells_kpc,useminpot,rfac_offset,verbose))
            procs.append(proc)
            proc.start()

        # complete the processes
        for proc in procs:
            proc.join()
        time.sleep(1)

        #load in outputs and save
        print()
        print('Consolidating galaxy outputs...')
        chunk_fnames=[os.getcwd()+'/tmpgalx/'+file for file in os.listdir(os.getcwd()+'/tmpgalx/')]
        chunk_dfs=[pd.read_hdf(fname,key='chunk') for fname in chunk_fnames]
        galaxies=pd.concat(chunk_dfs)
        galaxies.sort_values(by=['Time','ID'],ascending=[True,True],inplace=True)
        galaxies.reset_index(drop=True,inplace=True)

        if groupfinder:
            print()
            print('Grouping galaxies...')
            try:
                galaxies=basic_groupfinder(galaxies,verbose=verbose)
            except:
                print('Error: groupfinder failed.')
                return None

        print()
        print(f'----> Galaxy analysis for {len(self.snapshots)} snaps complete in {time.time()-t0stack:.2f} seconds.')

        self.galaxies=galaxies
        return galaxies
    
    def save_as_pickle(self, fname):
        """
        Save the simulation object as a pickle file.

        Parameters:
        -----------
        fname: str
            The name of the file to save the simulation object to.

        """
        if os.path.exists(fname):
            os.remove(fname)
        with open(fname, 'wb') as f:
            pickle.dump(self, f)
        f.close()

        print(f'Simulation object saved as {fname}.')
    

    ######################################################################################################
    ########################################## Plotting methods ##########################################
    ######################################################################################################


    # Method to plot the evolution of the galaxy properties
    def plot_glxevol(self,id=None):
        
        """
        Plot the evolution of the galaxy properties.

        Parameters:
        -----------
        simulation: simulation class
            The simulation object.
        id: int
            The ID of the galaxy to analse.
        
        """
        fig,axes=plot_glxevol(self,id=id)
        return fig,axes

    # Method to plot the evolution pair separation/velocity
    def plot_glxsep(self,ids=None,bh_subsample=10):
        """
        Plot the evolution of the pair separation/velocity.
        
        Parameters:
        -----------
        simulation: simulation class
            The simulation object.
        id1: int
            The ID of the first galaxy.
        id2: int
            The ID of the second galaxy.

        """
        fig,axes=plot_glxsep(self,ids=ids,bh_subsample=bh_subsample)
        return fig,axes
    
    # Method to render all simulation snapshots
    def gen_sim_animation(self,numproc=1,fps=10,type='baryons',frame=None,galaxies=None,useminpot=False,subsample=1,verbose=False):
        """
        Render all simulation snapshots.

        Parameters:
        -----------
        numproc: int
            The number of processes to use.
        fps: int
            The frames per second for the animation.
        type: str
            The type of particles to render.
        frame: float
            The size of the frame to render (in kpc)
        galaxies: pd.DataFrame
            The galaxy data from analyse_galaxies (see galaxyanalysis.py for details).
        useminpot: bool
            If True, use the minimum potential of the star particles as the galaxy centre.
        subsample: int
            The subsampling factor to use when loading the particle data.
        verbose: bool
            If True, print the progress of the rendering.

        """

        gen_sim_animation(self,numproc=numproc,fps=fps,type=type,frame=frame,galaxies=galaxies,useminpot=useminpot,subsample=subsample,verbose=verbose)

        

    # Method to render a merger 
    def gen_merger_animation(self,numproc=1,fps=5,ids=None,useminpot=False,verbose=False):
        """
        Render an animation of the interaction between two galaxies specified by their IDs.

        Parameters
        ----------
        numproc : int
            The number of processes to use.
        fps : int
            The frames per second for the animation.
        ids : list
            List of galaxy IDs to use in the animation.
        useminpot : bool
            If True, use the minimum potential of the star particles as the halo centre.
        verbose : bool
            If True, print the progress of the rendering.


        Returns
        ----------
        None (writes the output to a file).

        """

        gen_merger_animation(self,numproc=numproc,fps=fps,ids=ids,useminpot=useminpot,verbose=verbose)




    # KETJU binaries
    def plot_bhbinarypars(self,binaries=None):

        """
        Plot the binary parameters of the black holes.

        Parameters:
        -----------
        binaries: dict
            The binary orbital params from ketjugw.binary objects.

        """

        plot_bhbinarypars(self,binaries=binaries)

