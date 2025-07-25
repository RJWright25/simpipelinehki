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
import logging
from datetime import datetime

# Import relevant functions from other files
from .tools import *
from .halofinder import *
from .analysis import *
from .bhinfo import *
from .snapshots import *
from .plotting import *

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

    snapshot_idxlist: list
        The list of snapshot indices.

    cosmo: bool
        If True, assumes run is a comoving simulation.

    Attributes:
    -----------
    snapshot_file_list: list
        The list of paths to the snapshot files.
    snapshots: list
        The list of snapshot objects.
    snapshot_idxs: list
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
    def __init__(self, snapshot_file_list, snapshot_idxs=None,cosmorun=False):
    
        self.snapshot_flist = snapshot_file_list;times=[h5py.File(snapshot_file, 'r')['Header'].attrs['Time'] for snapshot_file in self.snapshot_flist]
        self.snapshot_flist = [snapshot_file for _,snapshot_file in sorted(zip(times,self.snapshot_flist))]
        if not snapshot_idxs:
            self.snapshots = {snapshot_idx:gadget_snapshot_hki(snapshot_file,snapshot_idx=snapshot_idx,cosmorun=cosmorun) for snapshot_idx,snapshot_file in enumerate(self.snapshot_flist)}
        else:
            self.snapshots = {snapshot_idx:gadget_snapshot_hki(snapshot_file,snapshot_idx=snapshot_idx,cosmorun=cosmorun) for snapshot_idx,snapshot_file in zip(snapshot_idxs,self.snapshot_flist)}
        
        self.snapshot_list= [self.snapshots[i] for i in self.snapshots.keys()]
        snapshot_idxs=list(self.snapshots.keys())
        self.snapshot_idxlist = snapshot_idxs
        self.timelist = [snapshot.time for snapshot in [self.snapshots[i] for i in snapshot_idxs]]
        self.redshiftlist = [snapshot.redshift for snapshot in [self.snapshots[i] for i in snapshot_idxs]]
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


    # Method to generate KD trees for all snapshots
    def gen_kdtrees(self,snapshotidxs=None,numproc=1, verbose=False):
            
        """
        Generate KD trees for all snapshots using multiprocessing. Parallelized across snapshots.

        Parameters:
        -----------
        numproc: int
            The number of processes to use. This parallelizes the KD tree generation over the snapshots.
        verbose: bool
            If True, print the progress of the KD tree generation.

        """

        if snapshotidxs is not None:
            snapshot_list=[self.snapshots[i] for i in snapshotidxs]
        else:
            snapshot_list=self.snapshots

        print()
        print(f'===========================================================================================')
        print(f'Generating KD trees for {len(snapshot_list)} snapshots using {numproc} processes...')
        print(f'===========================================================================================')
        print()

        t0stack=time.time()

        #make a directory for the outputs
        if not os.path.exists(os.getcwd()+'/postprocessing/kdtrees/'):
            os.makedirs(os.getcwd()+'/postprocessing/kdtrees/')
        else:
            for fname in os.listdir(os.getcwd()+'/postprocessing/kdtrees/'):
                if os.path.exists(os.getcwd()+'/postprocessing/kdtrees/'+fname) and fname.endswith('.pkl'):
                    os.remove(os.getcwd()+'/postprocessing/kdtrees/'+fname)

        #make a directory for the logs
        if not os.path.exists(os.getcwd()+'/postprocessing/kdtrees/logs'):
            os.makedirs(os.getcwd()+'/postprocessing/kdtrees/logs')

        #split the snapshots into chunks for multiprocessing
        snapshot_chunks=split_list(snapshot_list,numproc)

        procs=[]
        for iproc in range(numproc):
            snapshots_ichunk=snapshot_chunks[iproc]
            proc = multiprocessing.Process(target=stack_kdtrees_worker, args=(snapshots_ichunk,iproc,verbose))
            procs.append(proc)
            proc.start()

        #complete the processes
        for proc in procs:
            proc.join()
        time.sleep(1)

        print()
        print(f'----> KD tree generation for {len(self.snapshots)} snaps complete in {time.time()-t0stack:.2f} seconds.')



    # Method to find haloes in all snapshots using multiprocessing
    def find_haloes(self,snapshotidxs=None,numproc=1,delta=200,mcut=5,useminpot=False,verbose=False):
        
        """
        Find haloes in desired snapshots using multiprocessing. Parallelised across haloes in a given snapshot.

        Parameters:
        -----------
        snapshotidxs: list
            The indices of the snapshots to analyse.
        numproc: int
            The number of processes to use. This parallelizes the halo finding in a given snapshot.
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
        if snapshotidxs is not None:
            snapshot_list=[self.snapshots[i] for i in snapshotidxs]
        else:
            snapshot_list=self.snapshots

        print()
        print(f'===========================================================================================')
        print(f'Finding haloes in {len(snapshot_list)} snapshots using {numproc} processes...')
        print(f'===========================================================================================')
        print()

        t0stack=time.time()

        #make a directory for the outputs
        if not os.path.exists(os.getcwd()+'/postprocessing/haloes/'):
            os.makedirs(os.getcwd()+'/postprocessing/haloes/')
        else:
            for snapdir in os.listdir(os.getcwd()+'/postprocessing/haloes/'):
                if os.path.exists(os.getcwd()+'/postprocessing/haloes/'+snapdir) and 'log'  not in snapdir:
                    for fname in os.listdir(os.getcwd()+'/postprocessing/haloes/'+snapdir):
                        if os.path.exists(os.getcwd()+'/postprocessing/haloes/'+snapdir+'/'+fname):
                            os.remove(os.getcwd()+'/postprocessing/haloes/'+snapdir+'/'+fname)

        #make a directory for the logs
        if not os.path.exists(os.getcwd()+'/postprocessing/haloes/logs'):
            os.makedirs(os.getcwd()+'/postprocessing/haloes/logs')

        for snapshot in snapshot_list:
            print("Analysing haloes in snapshot = ", snapshot.snapshot_idx, ' --  z = ', snapshot.redshift)
            print("Time: ", datetime.now())

            if not os.path.exists(os.getcwd()+f'/postprocessing/haloes/snap_{str(snapshot.snapshot_idx).zfill(3)}/'):
                try:
                    os.mkdir(os.getcwd()+f'/postprocessing/haloes/snap_{str(snapshot.snapshot_idx).zfill(3)}/')
                except:
                    pass
            
            if not os.path.exists(os.getcwd()+f'/postprocessing/haloes/logs/snap_{str(snapshot.snapshot_idx).zfill(3)}/'):
                try:
                    os.mkdir(os.getcwd()+f'/postprocessing/haloes/logs/snap_{str(snapshot.snapshot_idx).zfill(3)}/')
                except:
                    pass

            #load in the KD tree for the snapshot
            if snapshot.snapshot_idx is None:
                snapfname=snapshot.snapshot_file.split('/')[-1].split('.hdf5')[0]
                kdpath=f'postprocessing/kdtrees/{snapfname}_kdtree.pkl'
            else:
                kdpath=f'postprocessing/kdtrees/kdtree_{str(snapshot.snapshot_idx).zfill(3)}.pkl'

            if os.path.exists(kdpath):
                with open(kdpath,'rb') as kdfile:
                    kdtree_snap=pickle.load(kdfile)
            else:
                print(f'Error: KD tree not found at {kdpath}. Please generate first.')
                return None

            #split the snapshots into chunks for multiprocessing
            procs=[]
            for iproc in range(numproc):
                proc = multiprocessing.Process(target=basic_halofinder, args=(snapshot,kdtree_snap,iproc,numproc,delta,mcut,useminpot,verbose))
                procs.append(proc)
                proc.start()

            #complete the processes
            for proc in procs:
                proc.join()
            time.sleep(1)

        print()
        print(f'----> Halo finding for {len(self.snapshots)} snaps complete in {time.time()-t0stack:.2f} seconds.')

        self.haloes=read_hdf_chunks('postprocessing/haloes/')        

    # Method to analyse galaxies in all snapshots using multiprocessing
    def analyse_galaxies(self,haloes=None,snapshotidxs=None,numproc=1,shells_kpc=None,useminpot=False,rfac_offset=0.1,verbose=False):
        """
        
        Analyse galaxies in all snapshots using multiprocessing.

        Parameters:
        -----------
        numproc: int
            The number of processes to use. This parallelizes the galaxy analysis in a given snapshot.
        snapshotidxs: list
            The indices of the snapshots to analyse.
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
        #if no snapshot indices are provided, analyse all snapshots
        if snapshotidxs is not None:
            snapshot_list=[self.snapshots[i] for i in snapshotidxs]
        else:
            snapshot_list=self.snapshots

        print()
        print(f'===========================================================================================')
        print(f'Analysing galaxies in {len(snapshot_list)} snapshots using {numproc} processes...')
        print(f'===========================================================================================')
        print()


        t0stack=time.time()

        #make a temporary directory for the outputs
        if not os.path.exists(os.getcwd()+'/postprocessing/galaxies/'):
            os.makedirs(os.getcwd()+'/postprocessing/galaxies/')
        else:
            for snapdir in os.listdir(os.getcwd()+'/postprocessing/galaxies/'):
                if os.path.exists(os.getcwd()+'/postprocessing/galaxies/'+snapdir) and 'log'  not in snapdir:
                    for fname in os.listdir(os.getcwd()+'/postprocessing/galaxies/'+snapdir):
                        if os.path.exists(os.getcwd()+'/postprocessing/galaxies/'+snapdir+'/'+fname):
                            os.remove(os.getcwd()+'/postprocessing/galaxies/'+snapdir+'/'+fname)

        #make a directory for the logs
        if not os.path.exists(os.getcwd()+'/postprocessing/galaxies/logs'):
            os.makedirs(os.getcwd()+'/postprocessing/galaxies/logs')

        #if haloes are not a dataframe, use the cached haloes
        if not isinstance(haloes,pd.DataFrame):
            try:
                haloes=read_hdf_chunks('postprocessing/haloes/')
            except:
                print('Error: No haloes found. Please run find_haloes first.')
                return None
            
        for snapshot in snapshot_list:
            procs=[]
            snapshotidx=snapshot.snapshot_idx
            print("Analysing galaxies in snapshot: ", snapshotidx)
            print("Time: ", datetime.now())

            if not os.path.exists(os.getcwd()+f'/postprocessing/galaxies/snap_{str(snapshotidx).zfill(3)}/'):
                try:
                    os.mkdir(os.getcwd()+f'/postprocessing/galaxies/snap_{str(snapshotidx).zfill(3)}/')
                except:
                    pass
            
            if not os.path.exists(os.getcwd()+f'postprocessing/galaxies/logs/snap_{str(snapshotidx).zfill(3)}/'):
                try:
                    os.mkdir(os.getcwd()+f'postprocessing/galaxies/logs/snap_{str(snapshotidx).zfill(3)}/')
                except:
                    pass

            #load in the KD tree for the snapshot
            if snapshot.snapshot_idx is None:
                snapfname=snapshot.snapshot_file.split('/')[-1].split('.hdf5')[0]
                kdpath=f'postprocessing/kdtrees/{snapfname}_kdtree.pkl'
            else:
                kdpath=f'postprocessing/kdtrees/kdtree_{str(snapshot.snapshot_idx).zfill(3)}.pkl'
            if os.path.exists(kdpath):
                with open(kdpath,'rb') as kdfile:
                    kdtree_snap=pickle.load(kdfile)
            else:
                print('Error: KD trees not found. Generate first.')
                return None
            
            print(f'Loaded KD tree for all processes in snapshot {snapshot.snapshot_idx}')
            for iproc in range(numproc):
                    time.sleep(0.1)
                    proc = multiprocessing.Process(target=galaxy_analysis, args=(snapshot,haloes,kdtree_snap,iproc,numproc,shells_kpc,useminpot,rfac_offset,verbose))
                    procs.append(proc)
                    proc.start()
                
            #complete the processes
            for proc in procs:
                proc.join()
            time.sleep(1)

            #load in outputs and save
            print('Finished analysing galaxies in snapshot: ', snapshotidx)
            print()

        print()
        print(f'----> Galaxy analysis for {len(self.snapshots)} snaps complete in {time.time()-t0stack:.2f} seconds.')

        self.galaxies=read_hdf_chunks('postprocessing/galaxies/')


    def load_saved_postprocessing(self):

        """
        Load the cached data from the postprocessing directory.

        """
        try:
            self.haloes=read_hdf_chunks('postprocessing/haloes/')
            print('Loaded saved halo data.')
        except:
            print('No halo data found.')
        try:
            self.galaxies=read_hdf_chunks('postprocessing/galaxies/')
            print('Loaded saved galaxy data.')
        except:
            print('No galaxy data found.')
        try:
            self.bhdetails=read_hdf_chunks('postprocessing/blackhole_details_post_processing/')
            print('Loaded saved halo data.')
        except:
            print('No black hole data found.')


    
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
    def plot_glxevol(self,radstr='2p00restar',id=None):
        
        """
        Plot the evolution of the galaxy properties.

        Parameters:
        -----------
        simulation: simulation class
            The simulation object.
        id: int
            The ID of the galaxy to analse.
        
        """
        fig,axes=plot_glxevol(self,radstr=radstr,id=id)
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
    def gen_sim_animation(self,numproc=1,fps=10,type='baryons',frame=None,staralpha=0.03,clims=None):
        """
        Render all simulation snapshots.

        Parameters:
        -----------
        numproc: int
            The number of processes to use.
        fps: int
            The frames per second for the animation.
        type: str
            The type of particles to render ('baryons', 'stars' or 'darkmatter').
        frame: float
            The size of the frame to render (in kpc)
        staralpha: float
            The alpha value of the star particles.
        verbose: bool
            If True, print the progress of the rendering.

        """

        gen_sim_animation(self,numproc=numproc,fps=fps,type=type,frame=frame,staralpha=staralpha,clims=clims)

        

    # Method to render a merger 
    def gen_merger_animation(self,numproc=1,ids=None,fps=10,staralpha=10,clims=(1e3,3e8),useminpot=False,verbose=False):
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

