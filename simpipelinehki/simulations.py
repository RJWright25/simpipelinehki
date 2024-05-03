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
        
        if snapshot_type=='gadget_idealised_snapshot_hki':
            snapshot_type = gadget_idealised_snapshot_hki
            self.snapshot_type = 'idealised'
        elif snapshot_type=='gadget_cosmo_snapshot_hki':
            snapshot_type = gadget_cosmo_snapshot_hki
            self.snapshot_type = 'cosmo'
        else:
            print('Error: snapshot type not recognized. Available types are "gadget_idealised_snapshot_hki" and "gadget_cosmo_snapshot_hki".')
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


    # Method to generate KD trees for all snapshots
    def gen_kdtrees(self,snapshotidxs=None,numproc=1, verbose=False):
            
        """
        Generate KD trees for all snapshots using multiprocessing.

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
        if not os.path.exists(os.getcwd()+'/outputs/'):
            os.mkdir(os.getcwd()+'/outputs/')
        if not os.path.exists(os.getcwd()+'/outputs/kdtrees/'):
            os.mkdir(os.getcwd()+'/outputs/kdtrees/')
        else:
            for fname in os.listdir(os.getcwd()+'/outputs/kdtrees/'):
                if os.path.exists(os.getcwd()+'/outputs/kdtrees/'+fname):
                    os.remove(os.getcwd()+'/outputs/kdtrees/'+fname)

        #make a directory for the logs
        if not os.path.exists(os.getcwd()+'/logs/'):
            os.mkdir(os.getcwd()+'/logs/')
        if not os.path.exists(os.getcwd()+'/logs/kdtrees/'):
            os.mkdir(os.getcwd()+'/logs/kdtrees/')
        else:
            for fname in os.listdir(os.getcwd()+'/logs/kdtrees/'):
                if os.path.exists(os.getcwd()+'/logs/kdtrees/'+fname):
                    os.remove(os.getcwd()+'/logs/kdtrees/'+fname)

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
    def find_haloes(self,snapshotidxs=None,numproc=1,delta=200,useminpot=False,verbose=False):
        
        """
        Find haloes in desired snapshots using multiprocessing.

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

        #make a temporary directory for the outputs
        if not os.path.exists(os.getcwd()+'/outputs/'):
            os.mkdir(os.getcwd()+'/outputs/')
        
        if not os.path.exists(os.getcwd()+'/outputs/haloes/'):
            os.mkdir(os.getcwd()+'/outputs/haloes/')

        #make a directory for the logs
        if not os.path.exists(os.getcwd()+'/logs/'):
            os.mkdir(os.getcwd()+'/logs/')
        if not os.path.exists(os.getcwd()+'/logs/haloes/'):
            os.mkdir(os.getcwd()+'/logs/haloes/')



        for snapshot in snapshot_list:
            print("Analysing haloes in snapshot = ", snapshot.snapshot_idx, ' --  z = ', snapshot.redshift)
            print("Time: ", datetime.now())

            if not os.path.exists(os.getcwd()+f'/outputs/haloes/snap_{str(snapshot.snapshot_idx).zfill(3)}/'):
                try:
                    os.mkdir(os.getcwd()+f'/outputs/haloes/snap_{str(snapshot.snapshot_idx).zfill(3)}/')
                except:
                    pass
            else:
                for fname in os.listdir(os.getcwd()+f'/outputs/haloes/snap_{str(snapshot.snapshot_idx).zfill(3)}/'):
                    if os.path.exists(os.getcwd()+f'/outputs/haloes/snap_{str(snapshot.snapshot_idx).zfill(3)}/'+fname):
                        try:
                            os.remove(os.getcwd()+f'/outputs/haloes/snap_{str(snapshot.snapshot_idx).zfill(3)}/'+fname)
                        except:
                            pass

            if not os.path.exists(os.getcwd()+f'/logs/haloes/snap_{str(snapshot.snapshot_idx).zfill(3)}/'):
                try:
                    os.mkdir(os.getcwd()+f'/logs/haloes/snap_{str(snapshot.snapshot_idx).zfill(3)}/')
                except:
                    pass
            else:
                for fname in os.listdir(os.getcwd()+f'/logs/haloes/snap_{str(snapshot.snapshot_idx).zfill(3)}/'):
                    if os.path.exists(os.getcwd()+f'/logs/haloes/snap_{str(snapshot.snapshot_idx).zfill(3)}/'+fname):
                        try:
                            os.remove(os.getcwd()+f'/logs/haloes/snap_{str(snapshot.snapshot_idx).zfill(3)}/'+fname)
                        except:
                            pass

            #load in the KD tree
            if os.path.exists(f'outputs/kdtrees/kdtree_{str(snapshot.snapshot_idx).zfill(3)}.pkl'):
                with open(f'outputs/kdtrees/kdtree_{str(snapshot.snapshot_idx).zfill(3)}.pkl','rb') as kdfile:
                    kdtree_snap=pickle.load(kdfile)
                    
                print(f'Loaded KD tree for all processes in snapshot {snapshot.snapshot_idx}')

            #split the snapshots into chunks for multiprocessing
            procs=[]
            for iproc in range(numproc):
                proc = multiprocessing.Process(target=basic_halofinder, args=(snapshot,kdtree_snap,iproc,numproc,delta,useminpot,verbose))
                procs.append(proc)
                proc.start()

            #complete the processes
            for proc in procs:
                proc.join()
            time.sleep(1)

        print()
        print(f'----> Halo finding for {len(self.snapshots)} snaps complete in {time.time()-t0stack:.2f} seconds.')

        
    # Method to analyse galaxies in all snapshots using multiprocessing
    def analyse_galaxies(self,snapshotidxs=None,numproc=1,shells_kpc=None,useminpot=False,rfac_offset=0.1,verbose=False):
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
        if not os.path.exists(os.getcwd()+'/outputs/'):
            os.mkdir(os.getcwd()+'/outputs/')
        
        if not os.path.exists(os.getcwd()+'/outputs/galaxies/'):
            os.mkdir(os.getcwd()+'/outputs/galaxies/')
        else:
            for snapdir in os.listdir(os.getcwd()+'/outputs/galaxies/'):
                if os.path.exists(os.getcwd()+'/outputs/galaxies/'+snapdir):
                    for fname in os.listdir(os.getcwd()+'/outputs/galaxies/'+snapdir):
                        if os.path.exists(os.getcwd()+'/outputs/galaxies/'+snapdir+'/'+fname):
                            os.remove(os.getcwd()+'/outputs/galaxies/'+snapdir+'/'+fname)

        #make a directory for the logs
        if not os.path.exists(os.getcwd()+'/logs/'):
            os.mkdir(os.getcwd()+'/logs/')
        if not os.path.exists(os.getcwd()+'/logs/galaxies/'):
            os.mkdir(os.getcwd()+'/logs/galaxies/')
        else:
            for snapdir in os.listdir(os.getcwd()+'/logs/galaxies/'):
                if os.path.exists(os.getcwd()+'/logs/galaxies/'+snapdir):
                    for fname in os.listdir(os.getcwd()+'/logs/galaxies/'+snapdir):
                        if os.path.exists(os.getcwd()+'/logs/galaxies/'+snapdir+'/'+fname):
                            os.remove(os.getcwd()+'/logs/galaxies/'+snapdir+'/'+fname)
        haloes=self.haloes

        for snapshot in snapshot_list:
            procs=[]
            snapshotidx=snapshot.snapshot_idx
            print("Analysing galaxies in snapshot: ", snapshotidx)
            print("Time: ", datetime.now())

            if not os.path.exists(os.getcwd()+f'/outputs/galaxies/snap_{str(snapshotidx).zfill(3)}/'):
                try:
                    os.mkdir(os.getcwd()+f'/outputs/galaxies/snap_{str(snapshotidx).zfill(3)}/')
                except:
                    pass
            
            if not os.path.exists(os.getcwd()+f'/logs/galaxies/snap_{str(snapshotidx).zfill(3)}/'):
                try:
                    os.mkdir(os.getcwd()+f'/logs/galaxies/snap_{str(snapshotidx).zfill(3)}/')
                except:
                    pass

            #load in the KD tree
            if os.path.exists(f'outputs/kdtrees/kdtree_{str(snapshot.snapshot_idx).zfill(3)}.pkl'):
                with open(f'outputs/kdtrees/kdtree_{str(snapshot.snapshot_idx).zfill(3)}.pkl','rb') as kdfile:
                    kdtree_snap=pickle.load(kdfile)
                    
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
    def gen_sim_animation(self,numproc=1,fps=10,type='baryons',frame=None,galaxies=pd.DataFrame(),useminpot=False,staralpha=1,subsample=1,clims=None):
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

        gen_sim_animation(self,numproc=numproc,fps=fps,type=type,frame=frame,galaxies=galaxies,useminpot=useminpot,staralpha=staralpha,subsample=subsample,clims=clims)

        

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

