#       _                                   _              _               _               _  _              
#  ___ (_) _ __ ___     __ _  _ __    __ _ | | _   _  ___ (_) ___   _ __  (_) _ __    ___ | |(_) _ __    ___ 
# / __|| || '_ ` _ \   / _` || '_ \  / _` || || | | |/ __|| |/ __| | '_ \ | || '_ \  / _ \| || || '_ \  / _ \
# \__ \| || | | | | | | (_| || | | || (_| || || |_| |\__ \| |\__ \ | |_) || || |_) ||  __/| || || | | ||  __/
# |___/|_||_| |_| |_|  \__,_||_| |_| \__,_||_| \__, ||___/|_||___/ | .__/ |_|| .__/  \___||_||_||_| |_| \___|
#                                              |___/               |_|       |_|                             


# halofinder.py
# This file contains the function and tools to find haloes in a snapshot.

import os
import time
import logging
import pickle
from datetime import datetime
import numpy as np
import pandas as pd
import astropy.units as apy_units

# This function is used to find haloes in a snapshot.
def basic_halofinder(snapshot,kdtree=None,iproc=0,numproc=1,delta=200,mcut=5,useminpot=False,verbose=False):

    """
    Basic halo finder for idealised/small hydro runs. Uses BH locations to find halo centres,
    then finds the minimum potential of the DM particles within a given radius of the BH.
    Returns a pandas dataframe with the properties of the halos found in the snapshot.

    Parameters
    ----------
    snapshot : snapshot object (or similar)
        Particle data object containing the snapshot data.
    delta : float
        Overdensity criteria for the halo finder.  
        Numeric value for overdensity.
    useminpot : bool
        If True, use the minimum potential of the star particles as the halo centre.
    verbose : bool
        If True, print the progress of the halo finding.

    Returns
    ----------
    haloes : pandas dataframe
        Pandas dataframe containing the properties of the halos found in the snapshot.
        
    Output fields:
    ----------
    ID : int
        Halo ID number.
    x : float
        x position of the halo based on black hole location.
    y : float
        y position of the halo based on black hole location.
    z : float
        z position of the halo based on black hole location.
    xminpot : float
        x position of the halo based on (stellar) minimum potential.
    yminpot : float
        y position of the halo based on (stellar) minimum potential.
    zminpot : float
        z position of the halo based on (stellar) minimum potential. 
    vx : float
        x velocity of the halo based on mean velocity of central DM particles.
    vy : float
        y velocity of the halo based on mean velocity of central DM particles.
    vz : float
        z velocity of the halo based on mean velocity of central DM particles.
    Halo_M_{delta}c : float
        Mass of the halo for the specified overdensity.
    Halo_R_{delta}c : float
        Radius of the halo for the specified overdensity.


    """

    t0=time.time()
    cosmo=snapshot.cosmology

    #set up the logging
    logging_folder=f'{os.getcwd()}/postprocessing/logs/haloes/snap_{str(snapshot.snapshot_idx).zfill(3)}/'
    if not os.path.exists(logging_folder):
        os.makedirs(logging_folder)
    logging_name=logging_folder+f'iproc_{str(iproc).zfill(3)}.log'
    if os.path.exists(logging_name):
        try:
            os.remove(logging_name)
        except:
            pass

    logging.basicConfig(filename=logging_name, level=logging.INFO)

    # set up the output
    output_folder=f'{os.getcwd()}/postprocessing/haloes/snap_{str(snapshot.snapshot_idx).zfill(3)}/'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_name=output_folder+f'iproc_{str(iproc).zfill(3)}.hdf5'
    if os.path.exists(output_name):
        try:
            os.remove(output_name)
        except:
            pass


    kdtree_snap=kdtree

    logging.info(f'')
    logging.info(f'************{datetime.now()}************')
    logging.info(f'')

    logging.info(f'===================================================================================================================')
    logging.info(f'Finding haloes in snapshot {snapshot.snapshot_file}, process {iproc}...')
    logging.info(f'===================================================================================================================')
    logging.info(f'')


    if verbose:
        print(f'===========================================================================================')
        print(f'Finding haloes in snapshot {snapshot.snapshot_file}, process {iproc} ...')
        print(f'===========================================================================================')
        print()

    t0=time.time()
    #create a pandas dataframe to store the halo properties
    columns=['Time','isnap','ID','x','y','z','BH_Mass','xminpot','yminpot','zminpot','vx','vy','vz',f'Halo_M_Crit{delta}',f'Halo_R_Crit{delta}']
    
    #initialize the dataframe with the number of BHs
    try:
        bhlocs=snapshot.get_particle_data(keys=['Coordinates','Velocities','Masses','ParticleIDs'],types=5)
        bhlocs=bhlocs.loc[bhlocs['Masses']>10**mcut,:]
    except:
        logging.info(f'No BHs found in snapshot {snapshot.snapshot_file}. Exiting...')
        if verbose:
            print(f'No BHs found in snapshot {snapshot.snapshot_file}. Exiting...')
        return pd.DataFrame(columns=columns)
    
    bhlocs.sort_values(by='Masses',ascending=False,inplace=True)
    bhlocs.reset_index(drop=True,inplace=True)
    bhlocs=bhlocs.loc[bhlocs['Masses']>10**mcut,:]

    #subsample the BHs based on the number of processes
    bhlocs=bhlocs.iloc[iproc::numproc,:]
    bhlocs.reset_index(drop=True,inplace=True)
    numbh=bhlocs.shape[0]
    halo_output={column:np.zeros(bhlocs['Masses'].shape[0])+np.nan for column in columns}
                
    logging.info(f'There are **{numbh}** BHs to use for finding haloes in snapshot {snapshot.snapshot_file}.')
    logging.info(f'')

    #for each "galaxy"
    for ibh in range(numbh):
        logging.info(f'Considering BH {ibh+1}/{numbh} [{(ibh+1)/numbh*100:.1f}% done with snap] (ID={bhlocs["ParticleIDs"].values[ibh]})...')

        if verbose:
            print(f'Considering BH {ibh+1}/{numbh} [{(ibh+1)/numbh*100:.1f}% done with snap] (ID={bhlocs["ParticleIDs"].values[ibh]})...')

        ibh_row=bhlocs.iloc[ibh]

        #save the snapshot number and time
        halo_output['Time'][ibh]=snapshot.time
        halo_output['isnap'][ibh]=snapshot.snapshot_idx
        halo_output['ID'][ibh]=int(ibh_row['ParticleIDs'])
        halo_output['x'][ibh]=ibh_row['Coordinates_x']
        halo_output['y'][ibh]=ibh_row['Coordinates_y']
        halo_output['z'][ibh]=ibh_row['Coordinates_z']
        halo_output['BH_Mass'][ibh]=ibh_row['Masses']
        
        t0_stars=time.time()
        #find star particles within 2 kpc of the bh
        centralstar = snapshot.get_particle_data(keys=['Coordinates','Velocities','Masses','Potential'],types=4,center=np.array([ibh_row['Coordinates_x'],ibh_row['Coordinates_y'],ibh_row['Coordinates_z']])*apy_units.kpc,radius=2*apy_units.kpc,return_rrel=False, kdtree=kdtree_snap)
        #if no potential data, use the bh location
        starspresent=centralstar.shape[0]

        #check if potential data is present
        if starspresent:
            if 'Potential' in centralstar.columns:
                potentialpresent=True
            else:
                potentialpresent=False

        if verbose:
            print(f"Took {time.time()-t0_stars:.2f} seconds to load {starspresent} star particles within 2 kpc of the BH.")
            
        if starspresent and potentialpresent:
            #find the 1000 star particles with the lowest potential energy using a boolean mask
            centralstar.sort_values(by='Potential',ascending=True,inplace=True);centralstar.reset_index(drop=True,inplace=True)
            centralstar = centralstar.iloc[:1000];centralstar.reset_index(drop=True,inplace=True)
            #find the center of potential of these particles weighted by their mass
            poscop = np.average(centralstar.loc[:,['Coordinates_x','Coordinates_y','Coordinates_z']].values,weights=centralstar['Masses'].values,axis=0)
            #find the average velocity of these particles weighted by their mass
            velcop = np.average(centralstar.loc[:,['Velocities_x','Velocities_y','Velocities_z']].values,weights=centralstar['Masses'].values,axis=0)

        else:
            poscop = np.array([ibh_row['Coordinates_x'],ibh_row['Coordinates_y'],ibh_row['Coordinates_z']])
            if centralstar.shape[0]==0 and verbose:
                print('No star particles found within 2 kpc of the BH. Using BH velocity as halo vel.')
                velcop = np.array([ibh_row['Velocities_x'],ibh_row['Velocities_y'],ibh_row['Velocities_z']])
            else:
                try:
                    velcop = np.average(centralstar.loc[:,['Velocities_x','Velocities_y','Velocities_z']].values,weights=centralstar['Masses'].values,axis=0)
                except:
                    velcop = np.array([0,0,0])+np.nan
                    
        #use the mass of the 2 kpc star particles to generate an upper limit estimate of the halo virial radius for loading particles
        if starspresent and np.nansum(np.isfinite(velcop)):
            mstar_2kpc=np.nansum(centralstar['Masses'].values)
            mhalo_est=mstar_2kpc*1e3
            rhalo_est=(mhalo_est/(delta*cosmo.critical_density(z=snapshot.redshift).to(apy_units.Msun/apy_units.kpc**3).value*4/3*np.pi))**(1/3)*2
        else:
            mhalo_est=0
            rhalo_est=0
                    
        logging.info(f'Estimated halo mass = {mhalo_est:.2e} Msun and radius = {rhalo_est:.2f} kpc.')

        #save the positions and velocities
        halo_output['xminpot'][ibh]=(poscop[0])
        halo_output['yminpot'][ibh]=(poscop[1])
        halo_output['zminpot'][ibh]=(poscop[2])
        halo_output['vx'][ibh]=velcop[0]
        halo_output['vy'][ibh]=velcop[1]
        halo_output['vz'][ibh]=velcop[2]

        #find the overdensity radius and mass
        critdens=delta*cosmo.critical_density(z=snapshot.redshift).to(apy_units.Msun/apy_units.kpc**3).value
        
        #define which center to use
        if useminpot:
            center=apy_units.Quantity([halo_output[f'{x}minpot'][ibh] for x in 'xyz'],unit='kpc')
        else:
            center=apy_units.Quantity([ibh_row['Coordinates_x'],ibh_row['Coordinates_y'],ibh_row['Coordinates_z']],unit='kpc')

        #get particle data within 1000 kpc of the center and sort by radius
        t0_counter=time.time()
        pdata_m200=snapshot.get_particle_data(keys=['Coordinates','Masses'],types=[0,1,4,5],center=center,radius=rhalo_est*apy_units.kpc,return_rrel=True,kdtree=kdtree_snap)

        if verbose:
            print(f"Took {time.time()-t0_counter:.2f} seconds to load particle data within {rhalo_est:.2f} kpc of the BH.")

        if pdata_m200.shape[0]==0:
            logging.info(f'No particles found within {rhalo_est:.2f} kpc of the BH. Skipping to next BH.')
            if verbose:
                print(f'No particles found within {rhalo_est:.2f} kpc of the BH. Skipping to next BH.')
            continue
        
        t0_counter=time.time()
        radius=pdata_m200['R'].values
        sorted_radius=np.argsort(radius)
        sorted_cummass=np.cumsum(pdata_m200['Masses'].values[sorted_radius])
        sorted_radius=radius[sorted_radius]
        sorted_volume=4/3*np.pi*(sorted_radius)**3
        sorted_cumdens=sorted_cummass/(sorted_volume)

        if verbose:
            print(f"Took {time.time()-t0_counter:.2f} seconds to sort the particle data for virial quantities.")


        iradius=len(sorted_cumdens)-np.searchsorted(sorted_cumdens[::-1],critdens)
        if not iradius<len(sorted_cumdens):
            iradius=len(sorted_cumdens)-1
    
        #save the mass and radius
        halo_output[f'Halo_M_Crit{delta}'][ibh]=sorted_cummass[iradius]
        halo_output[f'Halo_R_Crit{delta}'][ibh]=sorted_radius[iradius]


        #logging
        logging.info(f'--> Halo {bhlocs["ParticleIDs"].values[ibh]} analysed at snap {snapshot.snapshot_idx} -- actual mass = {halo_output[f"Halo_M_Crit{delta}"][ibh]:.2e} Msun and radius = {halo_output[f"Halo_R_Crit{delta}"][ibh]:.2f} kpc.')
        logging.info(f"--> Runtime: {time.time()-t0:.2f} seconds.")
        logging.info(f'')

        if verbose:
            print(f'--> Halo {bhlocs["ParticleIDs"].values[ibh]} analysed at snap {snapshot.snapshot_idx} -- actual mass = {halo_output[f"Halo_M_Crit{delta}"][ibh]:.2e} Msun and radius = {halo_output[f"Halo_R_Crit{delta}"][ibh]:.2f} kpc.')
            print(f"--> Runtime: {time.time()-t0:.2f} seconds.")
            print()

    logging.info(f'----> Halo finding complete for {snapshot.snapshot_file.split("/")[-1]} in {time.time()-t0:.2f} seconds.')
    logging.info(f'')

    if verbose:
        print(f'----> Halo finding complete for {snapshot.snapshot_file.split("/")[-1]} in {time.time()-t0:.2f} seconds.')
        print()
    
    
    #convert ID and isnap to integers
    halo_output['ID']=halo_output['ID'].astype(int)
    halo_output['isnap']=halo_output['isnap'].astype(int)

    #return the dataframe
    haloes=pd.DataFrame(halo_output)
    haloes.to_hdf(output_name,key='chunk')

    return haloes
