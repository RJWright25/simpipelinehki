#       _                                   _              _               _               _  _              
#  ___ (_) _ __ ___     __ _  _ __    __ _ | | _   _  ___ (_) ___   _ __  (_) _ __    ___ | |(_) _ __    ___ 
# / __|| || '_ ` _ \   / _` || '_ \  / _` || || | | |/ __|| |/ __| | '_ \ | || '_ \  / _ \| || || '_ \  / _ \
# \__ \| || | | | | | | (_| || | | || (_| || || |_| |\__ \| |\__ \ | |_) || || |_) ||  __/| || || | | ||  __/
# |___/|_||_| |_| |_|  \__,_||_| |_| \__,_||_| \__, ||___/|_||___/ | .__/ |_|| .__/  \___||_||_||_| |_| \___|
#                                              |___/               |_|       |_|                             

# galaxyanalysis.py
# This file contains the function and tools to analyse the properties of galaxies in a snapshot.

import os
import time
import numpy as np
import pandas as pd
import astropy.units as apy_units
import astropy.constants as apy_const

# This function is used to calculate the properties of a galaxy in a snapshot, given the properties of the halo.
def galaxy_analysis(snapshot,haloes,shells_kpc=None,useminpot=False,rfac_offset=0.1,verbose=True):

    """
    Calculate the properties of a galaxy in a snapshot, given the properties of the halo.

    Parameters
    ----------
    snapshot : snapshot object 
        snapshot object containing metadata and methods to get particle data.
    haloes : pandas dataframe
        Dataframe containing the properties of the halos found in the snapshot.
    shells_kpc : list of floats
        List of radii in kpc to calculate the properties of the galaxies.
    useminpot : bool
        If True, use the minimum potential of the star particles as the halo centre.
    rfac_offset : float
        Fractional value of sphere/shell radius to identify relevant particles.

    Returns
    ----------
    galaxies : pandas dataframe
        Dataframe containing the properties of the galaxies found in the snapshot.

    Output fields:
    ----------
    Time : float
        Time of the snapshot.
    ID : int
        Halo ID number.
    x : float
        x position of the galaxy based on black hole location.
    y : float
        y position of the galaxy based on black hole location.
    z : float
        z position of the galaxy based on black hole location.
    xminpot : float
        x position of the galaxy based on (DM) minimum potential. -- will warn if > 2kpc from BH.
    yminpot : float
        y position of the galaxy based on (DM) minimum potential. -- will warn if > 2kpc from BH.
    zminpot : float
        z position of the galaxy based on (DM) minimum potential. -- will warn if > 2kpc from BH.
    vx : float
        x velocity of the galaxy based on mean velocity of central DM particles.
    vy : float
        y velocity of the galaxy based on mean velocity of central DM particles.
    vz : float
        z velocity of the galaxy based on mean velocity of central DM particles.
    Halo_M_{delta}c : float
        Mass of the galaxy for the specified overdensity.
    Halo_R_{delta}c : float
        Radius of the galaxy for the specified overdensity.

    ==================================
    
    restar_sphere: float
        Stellar effective half-mass radius (calculated within 100 kpc spherical aperture). 
    regas_sphere: float
        Gas effective half-mass radius (calculated within 100 kpc spherical aperture).

    ==================================
    sphere quantities: in the below, xxxx is the radius in kpc, R200c (e.g. 0p20r200) or restar (e.g. 1p00restar)

    xxxx_sphere_star_tot: 
        Mass of the galaxy within xxxx sphere.
    xxxx_sphere_gas_tot:
        Star formation rate of the galaxy within xxxx sphere.
    xxxx_sphere_gas_sfr:
        Star formation rate of the galaxy within xxxx sphere.
    xxxx_sphere_gas_T:
        Temperature of the galaxy within xxxx sphere.
    xxxx_sphere_gas_Z:
        Metallicity of the galaxy within xxxx sphere.
    
    ==================================
    shell quantities: in the below, xxxx is the radius in kpc, R200c (e.g. 0p20r200) or restar

    xxxx_shell_star_sfsd:
        Star formation surface density of the galaxy at xxxx shell.
    xxxx_shell_gas_nH:
        Number density of the gas at xxxx shell.
    xxxx_shell_gas_T:
        Temperature of the gas at xxxx shell.
    xxxx_shell_gas_Z:
        Metallicity of the gas at xxxx shell.
    xxxx_shell_gasoutflow_all_mdot:
        Mass outflow rate of the gas at xxxx shell.
    xxxx_shell_gasoutflow_all_Z:
        Metallicity of all outflowing gas at xxxx shell.
    xxxx_shell_gasoutflow_all_T:
        Temperature of all outflowing gas at xxxx shell.
    xxxx_shell_gasoutflow_vcut_mdot:
        Mass outflow rate of the gas at xxxx shell with v>vcut.
    xxxx_shell_gasoutflow_vcut_Z:
        Metallicity of outflowing gas at xxxx shell with v>vcut.
    xxxx_shell_gasoutflow_vcut_T:
        Temperature of outflowing gas at xxxx shell with v>vcut.
    xxxx_shell_gasinflow_all_mdot:
        Mass inflow rate of the gas at xxxx shell.
    xxxx_shell_gasinflow_all_Z:
        Metallicity of inflowing gas at xxxx shell.
    xxxx_shell_gasinflow_all_T:
        Temperature of inflowing gas at xxxx shell.

    """ 
    
    if verbose:
        print(f'===========================================================================================')
        print(f'Characterising galaxies in snapshot {snapshot.snapshot_file}...')
        print(f'===========================================================================================')
        print()

    t0=time.time()

    #initialise the output 
    galaxy_output_all=[]

    #select the haloes in this snapshot
    isnap=snapshot.snapshot_idx
    haloes=haloes.loc[haloes['isnap'].values==isnap,:]
    haloes.reset_index(inplace=True,drop=True)
    
    if not haloes.shape[0]:
        print(f'----> No haloes tracked in snapshot {snapshot.snapshot_file.split("/")[-1]}.')
    
    #deal with the shells
    if shells_kpc is None: 
        shells_kpc=[0.01,0.05,0.1,0.5,1] #default shells
    shells_kpc_str=[]
    for ishellkpc in shells_kpc:
        if ishellkpc>=1: shells_kpc_str.append(f'{str(int(ishellkpc)).zfill(3)}kpc')
        else:shells_kpc_str.append(f'0p{str(int(ishellkpc*100)).zfill(2)}kpc')
    shells_kpc={shell_kpc_str:shell_kpc for shell_kpc_str,shell_kpc in zip(shells_kpc_str,shells_kpc)}

    #loop over the haloes
    for ihalo,halo in haloes.iterrows():
        if verbose:
            print(f'Considering galaxy {ihalo+1}/{haloes.shape[0]} (ID={int(halo["ID"])})...')

        #calculate cap for rhalf if the halo is close to another halo
        r_to_closest_halo=1e5
        rhalf_cap=0.05*halo['Halo_R_Crit200']
        if haloes.shape[0]>1:
            r_to_closest_halo=np.sqrt((halo['x']-haloes['x'].values)**2+(halo['y']-haloes['y'].values)**2+(halo['z']-haloes['z'].values)**2)
            r_to_closest_halo=sorted(r_to_closest_halo)[1]
        if r_to_closest_halo<0.05*halo['Halo_R_Crit200']/(2/3):
            rhalf_cap=r_to_closest_halo*2/3
            
        #if this shrinks the rhalf_cap to less than 1 kpc, set it to 1 kpc
        if rhalf_cap<1:
            rhalf_cap=1

        #center on the halo and 1.1R200 radius
        if useminpot:
            center=np.array([halo['xminpot'],halo['yminpot'],halo['zminpot']])*snapshot.units["Coordinates"]
        else:
            center=np.array([halo['x'],halo['y'],halo['z']])*snapshot.units["Coordinates"]

        #get the particle data and sort by radius
        galaxy=snapshot.get_particle_data(keys=['Coordinates','Velocities','Masses','Potential','StarFormationRate','Temperature','Metallicity','nH'],types=[0,4],center=center,radius=(1+rfac_offset*1.5)*halo['Halo_R_Crit200']*apy_units.kpc)
        galaxy.sort_values(by='R',ascending=True,inplace=True)
        galaxy.reset_index(inplace=True,drop=True)
        
        #masks
        maskgas=galaxy['ParticleTypes'].values==0
        maskstar=galaxy['ParticleTypes'].values==4

        #radii
        radii=galaxy['R'].values
        masses=galaxy['Masses'].values

        #create a dict to store the galaxy properties
        galaxy_output={}
        for column in list(halo.keys()):
            galaxy_output[column]=halo[column]

        #calculate the effective half-mass radius of the stars, gas, etc
        #select particles within rhalf_cap
        central=galaxy['R'].values<rhalf_cap
        gas=galaxy.loc[np.logical_and(maskgas,central),:]
        stars=galaxy.loc[np.logical_and(maskstar,central),:]
        doanalysis=True

        #calculate the effective half-mass radius of the stars and gas 
        for phase,phasestr in zip([gas,stars],['gas','star']):
            if phase.shape[0]:
                iradii=phase['R'].values
                cummass=np.cumsum(phase['Masses'].values)
                galaxy_output[f're{phasestr}_sphere']= iradii[np.searchsorted(cummass,0.5*cummass[-1])]
            else:
                doanalysis=False
                galaxy_output[f're{phasestr}_sphere']=np.nan
                
        if doanalysis:
            if verbose:
                print(f'Effective half-mass radii of stars and gas calculated for galaxy {halo["ID"]}.')
            #fix coordinates with halo
            galaxy['Coordinates_xrel']=galaxy['Coordinates_x'].values-center[0].value
            galaxy['Coordinates_yrel']=galaxy['Coordinates_y'].values-center[1].value
            galaxy['Coordinates_zrel']=galaxy['Coordinates_z'].values-center[2].value

            #fix the vrad using halo com
            halo_vcom=halo[['vx','vy','vz']].values
            galaxy['Velocities_x']-=halo_vcom[0];galaxy['Velocities_y']-=halo_vcom[1];galaxy['Velocities_z']-=halo_vcom[2]

            #find radial velocity
            galaxy['Velocities_r']=np.sum(galaxy.loc[:,[f'Velocities_{x}' for x in 'xyz']].values*galaxy.loc[:,[f'Coordinates_{x}rel' for x in 'xyz']].values,axis=1)/(np.sqrt(np.sum(galaxy.loc[:,[f'Coordinates_{x}rel' for x in 'xyz']].values**2,axis=1)))
            vrad=galaxy['Velocities_r'].values*apy_units.km/apy_units.s
            vrad_flow=vrad.to('kpc/yr')
            vrad_mask=vrad.to('km/s')
            v200=np.sqrt(2*halo['Halo_M_Crit200']*apy_units.Msun*(apy_const.G.to('kpc**3 yr**-2 Msun**-1'))/(halo['Halo_R_Crit200']*apy_units.kpc)).to('km/s')

            #deal with the shells
            shells_R200={'0p10r200':0.1*halo['Halo_R_Crit200'],'1p00r200':1*halo['Halo_R_Crit200']}
            shells_restar={'1p00restar':1*galaxy_output['restar_sphere'],'2p00restar':2*galaxy_output['restar_sphere']}
            shells_all={**shells_R200,**shells_restar,**shells_kpc}

            #calculate the properties of the galaxy within the shells
            for shell_str,shell_rad in shells_all.items():
                maxrad_idx=np.searchsorted(radii,shell_rad)
                maxrad_mask=np.ones(galaxy.shape[0])
                maxrad_mask[maxrad_idx:]=0
                maxrad_mask=maxrad_mask.astype(bool)
                shell_star_mask=np.logical_and(maskstar,maxrad_mask)
                shell_gas_mask=np.logical_and(maskgas,maxrad_mask)
                shell_gas_mass=masses[shell_gas_mask]
                shell_gas_hot=galaxy.loc[shell_gas_mask,'Temperature'].values>3e4
                shell_gas_cold=galaxy.loc[shell_gas_mask,'Temperature'].values<3e4

                shell_stars=galaxy.loc[shell_star_mask,:]
                shell_gas=galaxy.loc[shell_gas_mask,:]

                if shell_stars.shape[0]:
                    galaxy_output[f'{shell_str}_sphere_star_tot']=np.nansum(masses[shell_star_mask])
                    galaxy_output[f'{shell_str}_sphere_star_Z']=np.nansum(shell_stars['Metallicity_00'].values*masses[shell_star_mask])/galaxy_output[f'{shell_str}_sphere_star_tot']
                    
                else:
                    galaxy_output[f'{shell_str}_sphere_star_tot']=np.nan
                    galaxy_output[f'{shell_str}_sphere_star_Z']=np.nan

                if shell_gas.shape[0]:
                    galaxy_output[f'{shell_str}_sphere_gas_tot']=np.nansum(shell_gas_mass)
                    galaxy_output[f'{shell_str}_sphere_gas_tot']=np.nansum(shell_gas_mass*shell_gas_hot)
                    galaxy_output[f'{shell_str}_sphere_gas_cold']=np.nansum(shell_gas_mass*shell_gas_cold)
                    galaxy_output[f'{shell_str}_sphere_gas_sfr']=np.nansum(shell_gas['StarFormationRate'].values)
                    galaxy_output[f'{shell_str}_sphere_gas_Z']=np.nansum(shell_gas['Metallicity_00'].values*shell_gas_mass)/galaxy_output[f'{shell_str}_sphere_gas_tot']

                else:
                    galaxy_output[f'{shell_str}_sphere_gas_tot']=np.nan
                    galaxy_output[f'{shell_str}_sphere_gas_sfr']=np.nan
                    galaxy_output[f'{shell_str}_sphere_gas_Z']=np.nan

            #calculate the properties of the galaxy AT the shell radius
            for shell_str,shell_rad in shells_all.items():
                minrad_idx=np.searchsorted(galaxy['R'].values,shell_rad*(1-rfac_offset))
                rad_mask=np.ones(galaxy.shape[0])
                rad_mask[:minrad_idx]=0
                rad_mask[maxrad_idx:]=0
                rad_mask=rad_mask.astype(bool)
                
                shell_star_mask=np.logical_and(maskstar,rad_mask)
                shell_gas_mask=np.logical_and(maskgas,rad_mask)
                shell_stars=galaxy.loc[shell_star_mask,:];shell_star_mask=np.where(shell_star_mask)
                shell_gas=galaxy.loc[shell_gas_mask,:];shell_gas_mask=np.where(shell_gas_mask)
                shell_gas_mass=masses[shell_gas_mask]

                minrad=galaxy['R'].values[minrad_idx]
                maxrad=galaxy['R'].values[maxrad_idx-1]
                dA=4*np.pi*(maxrad**2-minrad**2)

                if shell_stars.shape[0]:
                    galaxy_output[f'{shell_str}_shell_star_totsd']=np.nansum(masses[shell_star_mask])/dA
                    galaxy_output[f'{shell_str}_shell_star_Z']=np.nansum(shell_stars['Metallicity_00'].values*masses[shell_star_mask])/galaxy_output[f'{shell_str}_shell_star_totsd']

                else:
                    galaxy_output[f'{shell_str}_shell_star_totsd']=0
                    galaxy_output[f'{shell_str}_shell_star_Z']=np.nan

                if shell_gas.shape[0]:
                    galaxy_output[f'{shell_str}_shell_gas_totsd']=np.nansum(shell_gas_mass)/dA
                    galaxy_output[f'{shell_str}_shell_gas_sfrsd']=np.nansum(shell_gas['StarFormationRate'].values)/dA                    
                    galaxy_output[f'{shell_str}_shell_gas_nH']=np.average(shell_gas['nH'].values,weights=shell_gas_mass)
                    galaxy_output[f'{shell_str}_shell_gas_T']=np.average(shell_gas['Temperature'].values,weights=shell_gas_mass)
                    galaxy_output[f'{shell_str}_shell_gas_Z']=np.nansum(shell_gas['Metallicity_00'].values*shell_gas_mass)/galaxy_output[f'{shell_str}_shell_gas_totsd']
                else:
                    galaxy_output[f'{shell_str}_shell_gas_totsd']=0
                    galaxy_output[f'{shell_str}_shell_gas_sfrsd']=0                
                    galaxy_output[f'{shell_str}_shell_gas_nH']=0
                    galaxy_output[f'{shell_str}_shell_gas_T']=np.nan
                    galaxy_output[f'{shell_str}_shell_gas_Z']=np.nan

                #calculate the properties of the gas inflow and outflow at the shell radius
                inflow_mask=vrad_mask[shell_gas_mask]<=0
                outflow_mask=vrad_mask[shell_gas_mask]>0
                outflow_vesc_mask=vrad_mask[shell_gas_mask]>0.25*v200
                vrad_converted=vrad_flow[shell_gas_mask]
                dr=shell_rad*(rfac_offset+1)-shell_rad*(1-rfac_offset)

                for mask, flow_str in zip([inflow_mask,outflow_mask,outflow_vesc_mask],['inflow_all','outflow_all','outflow_vesc']):
                    if 'inflow' in flow_str:sign=-1
                    else:sign=1

                    if np.nansum(mask):
                        galaxy_output[f'{shell_str}_shell_gas{flow_str}_mdot']=1/dr*np.sum(shell_gas_mass[mask]*(vrad_converted[mask]))*sign
                        galaxy_output[f'{shell_str}_shell_gas{flow_str}_T']=np.average(shell_gas['Temperature'].values[mask],weights=shell_gas_mass[mask])
                        galaxy_output[f'{shell_str}_shell_gas{flow_str}_Z']=np.nansum(shell_gas['Metallicity_00'].values[mask]*shell_gas_mass[mask])/np.nansum(shell_gas_mass[mask])

                    else:
                        galaxy_output[f'{shell_str}_shell_gas{flow_str}_mdot']=0
                        galaxy_output[f'{shell_str}_shell_gas{flow_str}_T']=np.nan
                        galaxy_output[f'{shell_str}_shell_gas{flow_str}_Z']=np.nan

            #append the galaxy properties to the output
            for column in galaxy_output:
                galaxy_output[column]=np.float64(galaxy_output[column])

            galaxy_output=pd.Series(galaxy_output).to_frame().T
            galaxy_output_all.append(galaxy_output)

        else:
            if verbose:
                print(f'No baryonic particles to analyse in galaxy {int(halo["ID"])}.')

            galaxy_output_all.append(pd.Series(halo).to_frame().T)
            continue

    #concatenate the outputs for each halo
    if galaxy_output_all:
        galaxies=pd.concat(galaxy_output_all)
        galaxies.reset_index(drop=True,inplace=True)
    else:
        print(f'No galaxies found in snapshot {snapshot.snapshot_file.split("/")[-1]}.')
        galaxies=pd.DataFrame()
    
    print(f'----> Galaxy characterisation for {snapshot.snapshot_file.split("/")[-1]} complete in {time.time()-t0:.2f} seconds.')
    if verbose:
        print()

    return galaxies


# This function is used to analyse several galaxies in a snapshot when using multiprocessing.
def stack_galaxies_worker(snaplist,haloes,iproc,shells_kpc=None,useminpot=False,rfac_offset=0.1,verbose=False):
    
    """
    Analyse galaxies in several snapshots, for use with multiprocessing.

    Parameters
    ----------
    snaplist : list of snapshot objects
        List of snapshot objects containing metadata and methods to get particle data.
    haloes : pandas dataframe
        Dataframe containing the properties of the halos found in the snapshot.
    iproc : int
        Index of the process (used for naming the output file).
    shells_kpc : list of floats
        List of radii in kpc to calculate the properties of the galaxies.
    useminpot : bool
        If True, use the minimum potential of the star particles as the halo centre.
    rfac_offset : float
        Fractional value of sphere/shell radius to identify relevant particles.

    Returns
    ----------
    None (writes the output to a file).

    """

    #initialise the output
    isnap_outputs=[]
    #loop over the snapshots
    for snapshot in snaplist:
        isnap_gals=galaxy_analysis(snapshot=snapshot,haloes=haloes,shells_kpc=shells_kpc,useminpot=useminpot,rfac_offset=rfac_offset,verbose=verbose)
        isnap_outputs.append(isnap_gals)
    
    #concatenate the outputs
    isnap_gals=pd.concat(isnap_outputs)
    fname=os.getcwd()+f'/tmpgalx/chunk_{str(iproc).zfill(3)}.hdf5'

    #write the output to a file
    if os.path.exists(fname):
        os.remove(fname)
    isnap_gals.to_hdf(fname,key='chunk')