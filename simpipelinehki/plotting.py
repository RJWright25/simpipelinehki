#       _                                   _              _               _               _  _              
#  ___ (_) _ __ ___     __ _  _ __    __ _ | | _   _  ___ (_) ___   _ __  (_) _ __    ___ | |(_) _ __    ___ 
# / __|| || '_ ` _ \   / _` || '_ \  / _` || || | | |/ __|| |/ __| | '_ \ | || '_ \  / _ \| || || '_ \  / _ \
# \__ \| || | | | | | | (_| || | | || (_| || || |_| |\__ \| |\__ \ | |_) || || |_) ||  __/| || || | | ||  __/
# |___/|_||_| |_| |_|  \__,_||_| |_| \__,_||_| \__, ||___/|_||___/ | .__/ |_|| .__/  \___||_||_||_| |_| \___|
#                                              |___/               |_|       |_|                             


# plotting.py
# This file contains a few functions to plot key results from the code.

import os
import time
import numpy as np
import pandas as pd
import multiprocessing
import matplotlib
import matplotlib.pyplot as plt
import sphviewer
import moviepy.video.io.ImageSequenceClip
from .tools import split_list

# Default matplotlib settings and color selections
plt.style.use('https://raw.githubusercontent.com/RJWright25/analysis/master/mplparams.txt')
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
dpi=1000

# Define cmaps for gas and stars
cname_star='#FDF9E8'
cmapname_gas='magma'
cmap_gas = plt.get_cmap(cmapname_gas, 256)
cmaplist_gas = cmap_gas(np.linspace(0, 1, 256))
for ival,cmapval in enumerate(cmaplist_gas):
    hsv=matplotlib.colors.rgb_to_hsv(cmapval[:3])
    cmaplist_gas[ival,:3] = matplotlib.colors.hsv_to_rgb(hsv)
    cmaplist_gas[ival,-1] = (ival+1)/256
cmap_gas = matplotlib.colors.ListedColormap(cmaplist_gas)


############ TIME SERIES DATA ############

# This function is used to plot the evolution of the properties of a galaxy specified by its ID.
def plot_glxevol(simulation,id=None):
    """
    Plots the evolution of the properties of a galaxy specified by its ID.

    Parameters
    ----------
    simulation : simulation object
        Simulation object containing the data to be plotted.
        NB: The simulation object must contain the following data:
        - haloes: pandas dataframe containing the properties of the halos found in the snapshot.
        - galaxies: pandas dataframe containing the properties of the galaxies found in the snapshot.
        - bhdetails: dictionary of pandas dataframes containing the properties of the black holes found in the snapshot.
    id : int
        ID of the galaxy to be plotted.

    Returns
    ----------
    fig : matplotlib figure
        The figure object containing the plot.
    axes : matplotlib axes
        The axes object containing the plot.
    """
    
    galaxies=simulation.galaxies
    bhdetails=simulation.bhdetails
    
    #make sure the dataframes are sorted
    galaxies.sort_values(by='Time',ascending=True,inplace=True)
    galaxies.reset_index(drop=True,inplace=True)

    #if no id is given, take the first halo
    if not id:
        #figure out which halo is the primary
        ids=galaxies['ID'].unique()
        ids_shape={id:galaxies.loc[galaxies['ID'].values==id,:].shape[0] for id in ids}
        id=[id for id in ids if ids_shape[id]==np.max(list(ids_shape.values()))][0]

    galaxy_masked=galaxies.loc[galaxies['ID'].values==id,:]
    bhdetails_masked=bhdetails[id]

    #kernels
    snapkernel=np.ones(1)/1;bhkernel=np.ones(10)/10

    #times
    snaptime=np.convolve(galaxy_masked['Time'].values,snapkernel,mode='valid')
    bhtime=np.convolve(bhdetails_masked['Time'].values,bhkernel,mode='valid')

    #mass
    mass_star=np.convolve(galaxy_masked['2p00restar_sphere_star_tot'].values,snapkernel,mode='valid')
    mass_bh=np.convolve(bhdetails_masked['bh_M'].values,bhkernel,mode='valid')

    #sfr and outflow/inflow
    sfr=np.convolve(galaxy_masked['2p00restar_sphere_gas_sfr'].values,snapkernel,mode='valid')
    inflow=np.convolve(galaxy_masked['2p00restar_shell_gasinflow_all_mdot'].values,snapkernel,mode='valid')
    outflow=np.convolve(galaxy_masked['2p00restar_shell_gasoutflow_all_mdot'].values,snapkernel,mode='valid')
    
    #figure
    fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(6,2.5),gridspec_kw={'left':0.15,'right':0.95,'bottom':0.1,'top':0.95,'hspace':0.2,'wspace':0.3})
    fig.set_dpi(dpi)
    for ax in axes:
        ax.grid(True,which='major',alpha=1)

    #mass
    axes[0].plot(snaptime,mass_star,c='k',lw=2.5,alpha=0.75)
    axes[0].plot(snaptime,mass_star,c='goldenrod',lw=1.5, label=r'$M_{\star}$ $(2\times R_{\rm eff})$')
    axes[0].plot(bhtime,mass_bh,c='grey',lw=1,alpha=0.5, label=r'$M_{\rm BH}$'+f' ({int(id)})')

    axes[0].set_xlabel(r'$t\, {\rm [Gyr]}$')
    axes[0].set_xlim(snaptime[0],snaptime[-1])
    axes[0].set_ylabel(r'$M\, [{\rm M}_{\odot}]$')
    axes[0].set_yscale('log')
    axes[0].legend(loc='lower center')

    #sfr
    axes[1].plot(snaptime,sfr,c='k',lw=2.5,alpha=0.75)
    axes[1].plot(snaptime,sfr,c='grey',lw=1.5, label=r'SFR $(2\times R_{\rm eff})$')

    axes[1].plot(snaptime,inflow,c='k',lw=2.5,alpha=0.75)
    axes[1].plot(snaptime,inflow,c='C0',lw=1.5, label=r'$\dot{M}_{\rm in} \, (2\times R_{\rm eff})$')

    axes[1].plot(snaptime,outflow,c='k',lw=2.5,alpha=0.75)
    axes[1].plot(snaptime,outflow,c='C1',lw=1.5, label=r'$\dot{M}_{\rm out} \, (2\times R_{\rm eff})$')


    axes[1].set_xlabel(r'$t\, {\rm [Gyr]}$')
    axes[1].set_xlim(snaptime[0],snaptime[-1])
    axes[1].set_ylabel(r'$\dot{M}$ [${\rm M}_{\odot}\,{\rm yr}^{-1}$]')
    axes[1].set_yscale('log')
    axes[1].legend(loc='lower center')

    if not os.path.exists(os.getcwd()+'/plots/'):
        os.mkdir(os.getcwd()+'/plots/')

    fig.set_dpi(dpi)
    plt.savefig(os.getcwd()+'/plots/'+f'glxevol_{int(id)}.png',bbox_inches='tight',dpi=dpi)

    return fig,axes


# This function is used to plot the separation and relative velocity of two galaxies specified by their IDs.
def plot_glxsep(simulation,ids=None,bh_subsample=10):
    
    """
    Plots the separation and relative velocity of two galaxies specified by their IDs.

    Parameters
    ----------
    simulation : simulation object
        Simulation object containing the data to be plotted.
        NB: The simulation object must contain the following data:
        - haloes: pandas dataframe containing the properties of the halos found in the snapshot.
        - galaxies: pandas dataframe containing the properties of the galaxies found in the snapshot.
        - bhdetails: dictionary of pandas dataframes containing the properties of the black holes found in the snapshot.
    ids: list
        List of IDs of the galaxies to be plotted.

    Returns
    ----------
    fig : matplotlib figure
        The figure object containing the plot.
    axes : matplotlib axes
        The axes object containing the plot.

    """

    galaxies=simulation.galaxies
    bhdetails=simulation.bhdetails
    
    #make sure the dataframes are sorted
    galaxies.sort_values(by='Time',ascending=True,inplace=True)
    galaxies.reset_index(drop=True,inplace=True)

    #if no ids are given, take the first two haloes
    if not ids:
        haloids=galaxies['ID'].unique()[:2]
        id1=haloids[0];id2=haloids[1]
    else:
        id1=ids[0];id2=ids[1]
        haloids=ids
    
    #mask galaxies and bhdetails
    galaxies_masked={id:galaxies.loc[galaxies['ID'].values==id,:] for id in haloids}
    bhdetails_masked={id:bhdetails[id].loc[::bh_subsample,:] for id in haloids}

    #figure out when both galaxies exist
    galaxies_masked={id:galaxies_masked[id].loc[galaxies_masked[id]['isnap'].values>=np.nanmax([np.nanmin(galaxies_masked[id]['isnap'].values) for id in haloids]),:] for id in haloids}
    galaxies_masked={id:galaxies_masked[id].loc[galaxies_masked[id]['isnap'].values<=np.nanmin([np.nanmax(galaxies_masked[id]['isnap'].values) for id in haloids]),:] for id in haloids}

    #times
    snaptime=galaxies_masked[id1]['Time'].values
    isnaps=np.int32(galaxies_masked[id1]['isnap'].values)
    redshifts=[simulation.snapshots[isnap].redshift for isnap in isnaps]
    scalefacs=np.array([1/(redshift+1) for redshift in redshifts])

    #separation
    xsep=galaxies_masked[haloids[0]]['x'].values-galaxies_masked[haloids[1]]['x'].values
    ysep=galaxies_masked[haloids[0]]['y'].values-galaxies_masked[haloids[1]]['y'].values
    zsep=galaxies_masked[haloids[0]]['z'].values-galaxies_masked[haloids[1]]['z'].values
    sep=np.sqrt(xsep**2+ysep**2+zsep**2)

    #relative velocity
    vxsep=(galaxies_masked[haloids[0]]['vx'].values-galaxies_masked[haloids[1]]['vx'].values)*np.sqrt(scalefacs)
    vysep=(galaxies_masked[haloids[0]]['vy'].values-galaxies_masked[haloids[1]]['vy'].values)*np.sqrt(scalefacs)
    vzsep=(galaxies_masked[haloids[0]]['vz'].values-galaxies_masked[haloids[1]]['vz'].values)*np.sqrt(scalefacs)
    vrel=np.sqrt(vxsep**2+vysep**2+vzsep**2)

    #bh separation
    #which bh lives the longest
    bhids=[int(bh) for bh in bhdetails_masked.keys()]
    bhids_shape={bh:bhdetails_masked[bh].shape[0] for bh in bhids}
    bhid_remnant=[bh for bh in bhids if bhids_shape[bh]==np.max(list(bhids_shape.values()))][0]
    bhid_sec=[bh for bh in bhids if bh!=bhid_remnant][0]

    #match time-step from secondary to primary
    time_sec=bhdetails_masked[bhid_sec]['Time'].values
    time_rem=bhdetails_masked[bhid_remnant]['Time'].values
    sep_bh=np.zeros(time_sec.shape[0])
    vel_bh=np.zeros(time_sec.shape[0])
    
    #for each time in the secondary, find the idx of the closest time in the primary and get sep/vel at that idx
    for itime,time in enumerate(time_sec):
        idx_prim=np.argmin(np.abs(time_rem-time))
        xyz_sec=bhdetails_masked[bhid_sec].loc[:,['Coordinates_x','Coordinates_y','Coordinates_z']].values[itime,:]
        xyz_rem=bhdetails_masked[bhid_remnant].loc[:,['Coordinates_x','Coordinates_y','Coordinates_z']].values[idx_prim,:]
        vel_sec=bhdetails_masked[bhid_sec].loc[:,['V_x','V_y','V_z']].values[itime,:]/bhdetails_masked[bhid_sec]['ScaleFactor'].values[itime]
        vel_rem=bhdetails_masked[bhid_remnant].loc[:,['V_x','V_y','V_z']].values[idx_prim,:]/bhdetails_masked[bhid_remnant]['ScaleFactor'].values[idx_prim]
        vel_bh[itime]=np.sqrt(np.sum((vel_sec-vel_rem)**2))
        sep_bh[itime]=np.sqrt(np.sum((xyz_sec-xyz_rem)**2))

    #r200 and restar
    r200_0=galaxies_masked[haloids[0]]['Halo_R_Crit200'].values
    # r200_1=galaxies_masked[haloids[1]]['Halo_R_Crit200'].values[:idx_merger]

    #find tlims
    tlims=(snaptime[0]-0.1,snaptime[-1]+0.1)

    #figure
    fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(6,2.5),gridspec_kw={'left':0.15,'right':0.95,'bottom':0.1,'top':0.95,'hspace':0.2,'wspace':0.3})
    fig.set_dpi(dpi)
    for ax in axes:
        ax.grid(True,which='major',alpha=1)

    #separation
    axes[0].plot(snaptime,sep,c='k',lw=2.5)
    axes[0].plot(snaptime,sep,c='grey',lw=1.5, label=r'Halo separation')
    
    axes[0].plot(time_sec,sep_bh,c='grey',lw=1,alpha=0.5,label=r'BH separation')

    axes[0].plot(snaptime,r200_0,c='k',lw=2.5)
    axes[0].plot(snaptime,r200_0,c='maroon',lw=1.5,label=r'Primary $R_{\rm 200c}$')
    axes[0].legend(loc='lower center')

    axes[0].set_xlabel(r'$t\, {\rm [Gyr]}$')
    axes[0].set_xlim(tlims)
    axes[0].set_ylabel(r'Separation [kpc]')
    axes[0].set_yscale('log')
    axes[0].set_ylim(10**-1.5,10**3)

    #relative vel
    smoothn=1;kernel=np.ones(smoothn)/smoothn
    axes[1].plot(np.convolve(snaptime,kernel,mode='valid'),np.convolve(vrel,kernel,mode='valid'),c='k',lw=2.5)
    axes[1].plot(np.convolve(snaptime,kernel,mode='valid'),np.convolve(vrel,kernel,mode='valid'),c='grey',lw=1.5, label=r'Halo $v_{\rm rel}$')

    axes[1].plot(time_sec,vel_bh,c='grey',lw=1,alpha=0.5,label=r'BH $v_{\rm rel}$')

    axes[1].set_xlabel(r'$t\, {\rm [Gyr]}$')
    axes[1].set_xlim(tlims)
    axes[1].set_ylabel(r'Relative velocity [${\rm km}\,{\rm s}^{-1}$]')
    axes[1].set_yscale('log')
    axes[1].legend(loc='lower center')

    if not os.path.exists(os.getcwd()+'/plots/'):
        os.mkdir(os.getcwd()+'/plots/')

    fig.set_dpi(dpi)
    plt.savefig(os.getcwd()+'/plots/'+f'glxsep_{int(id1)}_{int(id2)}.png',bbox_inches='tight',dpi=dpi)

    return fig,axes



############ RENDERING A SIMULATION ############


def render_snap(snapshot,type='baryons',frame=None,galaxies=None,center=None,useminpot=False,subsample=1,verbose=False):
    """
    Render a snapshot of the simulation.

    Parameters
    ----------
    snapshot : snapshot object
        Snapshot object to render.
    type : str
        Type of rendering to perform. Options are 'baryons' and 'dm'.
    galaxies : pandas dataframe
        Dataframe containing the properties of the galaxies found in the snapshots (optional).
    useminpot : bool
        If True, use the minimum potential of the star particles as the halo centre.
    verbose : bool
        If True, print the progress of the rendering.
    """

    #set up the rendering parameters based on type
    if type=='baryons':
        ptypes=[0,4,1];radstr='Halo_R_Crit200';rfac=1
        cmap=cmap_gas
        ls_sphere='--'
    elif type=='dm':
        ptypes=[1,0,4];radstr='Halo_R_Crit200';rfac=1
        cmap='viridis'
        ls_sphere='--'
    else:
        print('Type not recognized. Options are "baryons" and "dm".')
        return

    censtr=''
    if useminpot:censtr='minpot'

    #get galaxies if not provided
    if not np.any(galaxies):
        try:
            galaxies=snapshot.galaxies
        except:
            galaxies=pd.DataFrame()

    pdata=snapshot.get_particle_data(keys=['Coordinates','Masses'], types=ptypes, center=None, radius=None,subsample=subsample)

    #find center based on particle positions
    if not center:
        center=np.sum(pdata.loc[:,[f'Coordinates_{x}' for x in 'xyz']].values*pdata['Masses'].values[:,np.newaxis],axis=0)/np.sum(pdata['Masses'].values)
        pdata['Coordinates_x']-=center[0]
        pdata['Coordinates_y']-=center[1]
        pdata['Coordinates_z']-=center[2]

    #find frame and center based on particle positions
    if not frame:
        max_x=np.nanpercentile(pdata['Coordinates_x'].values,99)
        min_x=np.nanpercentile(pdata['Coordinates_x'].values,1)
        frame=(max_x-min_x)/2

        #if cosmo, half this
        if snapshot.cosmorun:
            frame/=2

    sph_fluidmask=pdata['ParticleTypes'].values==ptypes[0]
    sph_particles=sphviewer.Particles(pdata.loc[sph_fluidmask,[f'Coordinates_{x}' for x in 'xyz']].values,
                                      pdata.loc[sph_fluidmask,'Masses'].values,nb=32)
    
    sph_camera = sphviewer.Camera(r='infinity', t=0, p=0, roll=0, xsize=1500, ysize=1500,
                                                x=0, y=0, z=0,
                                                extent=[-frame,frame,-frame,frame])
    sph_scene = sphviewer.Scene(sph_particles,sph_camera)
    sph_render = sphviewer.Render(sph_scene)
    sph_extent = sph_render.get_extent()
    sph_img=sph_render.get_image()
    
    #make figure and plot
    fig,ax=plt.subplots(1,1,figsize=(5,5),gridspec_kw={'left':0.1,'right':0.99,'bottom':0.1,'top':0.99})
    ax.set_facecolor('k')
    ax.grid(which='both',alpha=0)
    ax.imshow(sph_img,extent=sph_extent,origin='lower',cmap=cmap,norm=matplotlib.colors.LogNorm(),zorder=1)
    
    #add stars if necessary
    if type=='baryons':
        stars=pdata.loc[pdata['ParticleTypes'].values==4,:]
        ax.scatter(stars.loc[:,'Coordinates_x'].values,stars.loc[:,'Coordinates_y'].values,c=cname_star,alpha=0.03,s=0.05,lw=0,zorder=2)

    #add galaxy positions
    if galaxies.shape[0]:
        isnap_galaxies=galaxies.loc[galaxies['isnap'].values==snapshot.snapshot_idx,:]
        if isnap_galaxies.shape[0]:
            mstar_mask=isnap_galaxies['1p00restar_sphere_star_tot'].values>=0
            centrals=np.logical_and.reduce([isnap_galaxies['Central'].values==1,mstar_mask])
            sats=np.logical_and.reduce([isnap_galaxies['Central'].values==0,mstar_mask])
            remnants=np.logical_and.reduce([isnap_galaxies['RemnantCentral'].values==1,mstar_mask])

            if np.nansum(centrals):
                for igal,gal in isnap_galaxies.loc[centrals,:].iterrows():
                    ax.scatter(gal[f'x{censtr}']-center[0],gal[f'y{censtr}']-center[1],s=1,c='w',zorder=3)
                    ax.scatter(gal[f'x{censtr}']-center[0],gal[f'y{censtr}']-center[1],s=0.5,c='k',zorder=3)
                    ax.add_artist(plt.Circle(radius=gal[radstr]*rfac,xy=[gal[f'x{censtr}']-center[0],gal[f'y{censtr}']-center[1]],color='w',lw=0.5,ls=ls_sphere,fill=False,zorder=3))

            if np.nansum(sats):
                for igal,gal in isnap_galaxies.loc[sats,:].iterrows():
                    ax.scatter(gal[f'x{censtr}']-center[0],gal[f'y{censtr}']-center[1],s=0.5,c='w',zorder=3)
                    ax.scatter(gal[f'x{censtr}']-center[0],gal[f'y{censtr}']-center[1],s=0.25,c='grey',zorder=3)
            
            if np.nansum(remnants):
                for ibh,bh in isnap_galaxies.loc[remnants,:].iterrows():
                    ax.add_artist(plt.Circle(radius=bh[radstr]*rfac,xy=[bh[f'x{censtr}']-center[0],bh[f'y{censtr}']-center[1]],color='w',lw=1,ls='-',fill=False,zorder=3))
                    ax.add_artist(plt.Circle(radius=bh[radstr]*rfac,xy=[bh[f'x{censtr}']-center[0],bh[f'y{censtr}']-center[1]],color='red',lw=0.5,ls='-',fill=False,zorder=3))

                    
    ax.set_xlim(-frame,frame)
    ax.set_ylim(-frame,frame)

    ax.text(0.55,0.01,'$x$ [kpc]',transform=fig.transFigure,ha='center',va='bottom')
    ax.text(0.01,0.55,'$y$ [kpc]',transform=fig.transFigure,ha='left',va='center',rotation=90)
    ax.text(x=0.05,y=0.95,s=r'$t='+f'{snapshot.time:.3f}$ Gyr',transform=ax.transAxes,ha='left',va='top',color='w')
    #if a cosmological sim, also add the redshift
    if snapshot.cosmorun:
        ax.text(x=0.95,y=0.95,s=r'$z='+f'{snapshot.redshift:.3f}$',transform=ax.transAxes,ha='right',va='top',color='w')
    
    fig.set_dpi(dpi)

    return fig,ax


def render_sim_worker(snaplist,type='baryons',frame=None,galaxies=None,useminpot=False,subsample=1,verbose=False):
    
    """
    Worker function to make an animation of the simulation for a given set of snaps.

    Parameters
    ----------
    snaplist : list
        List of snapshot objects (or similar) to use in the animation.
    type : str
        Type of rendering to perform. Options are 'baryons' and 'dm'.
    frame : float
        Size of the frame to use in the rendering.
    galaxies : pandas dataframe
        Dataframe containing the properties of the galaxies found in the snapshots (optional).
    useminpot : bool
        If True, use the minimum potential of the star particles as the halo centre.
    verbose : bool
        If True, print the progress of the animation.
    
    """
    
    for snapshot in snaplist:
        if verbose:
            print(f"Rendering snap {snapshot.snapshot_idx}...")
        fig,_=render_snap(snapshot,type=type,frame=frame,galaxies=galaxies,useminpot=useminpot,subsample=subsample,verbose=verbose)
        fig.savefig(f'plots/render_sim_{type}/snap_{str(snapshot.snapshot_idx).zfill(3)}.png',dpi=dpi)
        plt.close(fig)


def gen_sim_animation(simulation,numproc=1,fps=10,type='baryons',frame=None,galaxies=None,useminpot=False,subsample=1,verbose=False):
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

    #make a directory for the outputs; if it already exists, remove the files
    image_folder=f'{os.getcwd()}/plots/render_sim_{type}/'
    if not os.path.exists(f'{os.getcwd()}/plots/'):
        os.mkdir(f'{os.getcwd()}/plots/')
    if not os.path.exists(image_folder):
        os.mkdir(image_folder)
    else:
        for fname in os.listdir(image_folder):
            if os.path.exists(image_folder+fname):
                os.remove(image_folder+fname)
    
    #split the snapshots into chunks for multiprocessing
    snapshot_list=simulation.snapshots
    snapshots_chunks=split_list(snapshot_list,numproc)

    #start the processes
    procs=[]
    for iproc in range(numproc):
        time.sleep(0.1)
        snapshots_ichunk=snapshots_chunks[iproc]
        if verbose:
            print(f'Process {iproc} getting snaps: ', [snapshot.snapshot_idx for snapshot in snapshots_ichunk])
        proc = multiprocessing.Process(target=render_sim_worker, args=(snapshots_ichunk,type,frame,galaxies,useminpot,subsample,verbose))
        procs.append(proc)
        proc.start()

    # complete the processes
    for proc in procs:
        proc.join()
    time.sleep(2)

    #load in snapshots, make a movie
    image_files = sorted([os.path.join(image_folder,img)
                for img in os.listdir(image_folder)
                if img.endswith(".png")])
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
    clip.write_videofile(f'{image_folder}/animation_{type}.mp4')


############ RENDERING A MERGER ############



# This function is used to create an animation of the interaction between two galaxies specified by their IDs.
def render_merger_worker(snaplist,galaxies,ids=None,useminpot=False,verbose=False):
    
    """
    Worker function to make an animation of the interaction between two galaxies specified by their IDs.

    Parameters
    ----------
    snaplist : list
        List of snapshot objects (or similar) to use in the animation.
    galaxies : pandas dataframe
        Dataframe containing the properties of the galaxies found in the snapshots.
    ids : list
        List of galaxy IDs to use in the animation.
    useminpot : bool
        If True, use the minimum potential of the star particles as the halo centre.
    verbose : bool
        If True, print the progress of the animation.

    Returns
    ----------
    None (writes the output to a file).
    
    """

    #make an animation of stars & gas
    if not ids:
        haloids_unique=galaxies['ID'].unique()[::-1][:2]
    else:
        haloids_unique=ids

    #id remaining
    remnantid=haloids_unique[np.nanargmax([galaxies.loc[galaxies['ID'].values==haloid,:].shape[0] for haloid in haloids_unique])]
    secondid=haloids_unique[np.nanargmin([galaxies.loc[galaxies['ID'].values==haloid,:].shape[0] for haloid in haloids_unique])]

    for snapshot in snaplist:
        isnap=snapshot.snapshot_idx
        isnap_gals=galaxies.loc[galaxies['isnap'].values==isnap,:]
        isnap_primary=isnap_gals.loc[isnap_gals['ID'].values==remnantid,:];isnap_primary.reset_index(drop=True,inplace=True)
        isnap_secondary=isnap_gals.loc[isnap_gals['ID'].values==secondid,:];isnap_secondary.reset_index(drop=True,inplace=True)

        if isnap_primary.shape[0]==1 and isnap_secondary.shape[0]==1:
            merged=False
        else:
            merged=True
        
        #get xysep
        poskey=''
        if useminpot:
            poskey='minpot'
        if not merged:
            x1=isnap_primary[f'x{poskey}'].values[0];x2=isnap_secondary[f'x{poskey}'].values[0]
            y1=isnap_primary[f'y{poskey}'].values[0];y2=isnap_secondary[f'y{poskey}'].values[0]
            z1=isnap_primary[f'z{poskey}'].values[0];z2=isnap_secondary[f'z{poskey}'].values[0]
            rad1=isnap_primary['restar_sphere'].values[0]*4;rad2=isnap_secondary['restar_sphere'].values[0]*4
            xysep=np.sqrt((x1-x2)**2+(y1-y2)**2)
            center=np.array([(x1+x2)/2,(y1+y2)/2,(z1+z2)/2])
        else:
            x1=isnap_primary[f'x{poskey}'].values[0]
            y1=isnap_primary[f'y{poskey}'].values[0]
            z1=isnap_primary[f'z{poskey}'].values[0]
            x2=np.nan;y2=np.nan;z2=np.nan
            rad1=isnap_gals['restar_sphere'].values[0]*4;rad2=np.nan
            xysep=0
            center=np.array([x1,y1,z1])

        fig,ax=plt.subplots(1,1,figsize=(5,5),gridspec_kw={'left':0.1,'right':0.99,'bottom':0.1,'top':0.99})
        ax.set_facecolor('k')
        ax.grid(which='both',alpha=0)

        if verbose:
            print(f"Rendering snap {isnap}...")

        frame=np.nanmax([xysep*1,25])

        pdata=snapshot.get_particle_data(keys=['Coordinates','Masses'], types=[0,4], center=None, radius=None,subsample=1)
        stars=pdata.loc[pdata['ParticleTypes'].values==4,:]
        gas=pdata.loc[pdata['ParticleTypes'].values==0,:]

        #sph rendering
        sph_particles=sphviewer.Particles(gas.loc[:,[f'Coordinates_{x}' for x in 'xyz']].values-center,gas['Masses'].values,nb=32)
        sph_camera = sphviewer.Camera(r='infinity', t=0, p=0, roll=0, xsize=1500, ysize=1500,
                                                    x=0, y=0, z=0,
                                                    extent=[-frame,frame,-frame,frame])
        sph_scene=sphviewer.Scene(sph_particles,sph_camera)
        sph_render = sphviewer.Render(sph_scene)
        sph_extent = sph_render.get_extent()
        sph_img=sph_render.get_image()
        sph_img[sph_img==0]

        # ax.fill_between([-200,200],[-200,-200],[200,200],color='k',alpha=1,zorder=0)
        ax.imshow(sph_img,extent=sph_extent,origin='lower',cmap=cmap_gas,norm=matplotlib.colors.LogNorm(1e4,1e9),zorder=1)
        ax.scatter(stars.loc[:,'Coordinates_x'].values-center[0],stars.loc[:,'Coordinates_y'].values-center[1],c=cname_star,alpha=0.03,s=0.05,lw=0,zorder=2)

        #plot the galaxies
        ax.scatter(x1-center[0],y1-center[1],s=2,c=f'w',zorder=2)
        ax.scatter(x1-center[0],y1-center[1],s=1,c=f'k',zorder=2)
        ax.add_artist(plt.Circle(radius=rad1,xy=[x1-center[0],y1-center[1]],color=f'w',lw=0.5,ls='--',fill=False,zorder=2))
        
        if not merged:
            ax.scatter(x2-center[0],y2-center[1],s=2,c=f'w',zorder=2)
            ax.scatter(x2-center[0],y2-center[1],s=1,c=f'k',zorder=2)
            ax.add_artist(plt.Circle(radius=rad2,xy=[x2-center[0],y2-center[1]],color=f'w',lw=0.5,ls='--',fill=False,zorder=2))
        else:
            ax.text(x=0.5,y=0.95,s='Merged',transform=ax.transAxes,ha='center',va='top',color='w')
        
        ax.set_xlim(-frame,frame)
        ax.set_ylim(-frame,frame)
        ax.text(0.55,0.01,'$x$ [kpc]',transform=fig.transFigure,ha='center',va='bottom')
        ax.text(0.01,0.55,'$y$ [kpc]',transform=fig.transFigure,ha='left',va='center',rotation=90)
        ax.text(x=0.05,y=0.95,s=r'$t='+f'{snapshot.time:.3f}$ Gyr',transform=ax.transAxes,ha='left',va='top',color='w')
        #if a cosmological sim, also add the redshift
        if snapshot.cosmorun:
            ax.text(x=0.95,y=0.95,s=r'$z='+f'{snapshot.redshift:.3f}$',transform=ax.transAxes,ha='right',va='top',color='w')        
        plt.savefig(f'plots/render_merger_{int(ids[0])}_{int(ids[1])}/snap_{str(isnap).zfill(3)}.png',dpi=dpi)
        plt.close()


# Method to render a merger 
def gen_merger_animation(simulation,numproc=1,fps=10,ids=None,useminpot=False,verbose=False):

    """
    Render an animation of the interaction between two galaxies specified by their IDs.

    Parameters
    ----------
    simulation : simulation object
        Simulation object containing the data to be plotted.
        NB: The simulation object must contain the following data:
        - haloes: pandas dataframe containing the properties of the halos found in the snapshot.
        - galaxies: pandas dataframe containing the properties of the galaxies found in the snapshot.
        - bhdetails: dictionary of pandas dataframes containing the properties of the black holes found in the snapshot.
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

    if not np.any(ids):
        haloids_unique=simulation.galaxies['ID'].unique()[:2]
        ids=sorted([int(haloid) for haloid in haloids_unique])

    image_folder=f'{os.getcwd()}/plots/render_merger_{int(ids[0])}_{int(ids[1])}/'
    if not os.path.exists(os.getcwd()+'/plots/'):
        os.mkdir(os.getcwd()+'/plots/')
    if not os.path.exists(image_folder):
        os.mkdir(image_folder)
    else:
        for fname in os.listdir(image_folder):
            if os.path.exists(image_folder+fname):
                os.remove(image_folder+fname)

    snapshot_list=simulation.snapshots

    #find snapshots with both galaxies
    galaxies=simulation.galaxies
    isnaps_halo1=galaxies.loc[galaxies['ID'].values==ids[0],'isnap'].values
    isnaps_halo2=galaxies.loc[galaxies['ID'].values==ids[1],'isnap'].values

    print(f'Galaxy {ids[0]} found in snapshots: ',isnaps_halo1)
    print(f'Galaxy {ids[1]} found in snapshots: ',isnaps_halo2)

    #find the common snapshots
    common_snaps=np.intersect1d(isnaps_halo1,isnaps_halo2)
    snapshot_list=[snapshot for snapshot in snapshot_list if snapshot.snapshot_idx in common_snaps]
    #add 5 snapshots after
    isnap_last=common_snaps[-1]
    for i in range(1,5):
        snapshot_list.append(simulation.snapshots[int(isnap_last+i)])

    #split for computation
    snapshots_chunks=split_list(snapshot_list,numproc)

    procs=[]
    for iproc in range(numproc):
        time.sleep(0.1)
        snapshots_ichunk=snapshots_chunks[iproc]
        if verbose:
            print(f'Process {iproc} getting snaps: ', [snapshot.snapshot_idx for snapshot in snapshots_ichunk])            
        proc = multiprocessing.Process(target=render_merger_worker, args=(snapshots_ichunk,simulation.galaxies,ids,useminpot,verbose))
        procs.append(proc)
        proc.start()

    # complete the processes
    for proc in procs:
        proc.join()
    time.sleep(2)

    image_files = sorted([os.path.join(image_folder,img)
                for img in os.listdir(image_folder)
                if img.endswith(".png")])
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
    clip.write_videofile(f'plots/render_merger_{int(ids[0])}_{int(ids[1])}/animation.mp4')




############ KETJU BINARIES ############

def plot_bhbinarypars(simulation,binaries=None):
    """
    Plot the binary parameters of the black holes in the simulation.
    
    Parameters
    ----------
    simulation : simulation object
        Simulation object containing the data to be plotted.
        NB: The simulation object must contain the following data:
        - ketjubinaries: dictionary of pandas dataframes containing the properties of the black hole binaries.

    binaries : list
        List of binary ID tuples to plot. If None, plot all binaries.

    Returns
    ----------
    None (writes the output to a file).

    """

    if not binaries:
        binaries=list(simulation.ketjubinaries.keys())

    for binary in binaries:
        binarypars=simulation.ketjubinaries[binary]

        # plot a and e
        fig,axes=plt.subplots(2,1,figsize=(5.5,4),gridspec_kw={'left':0.1,'right':0.97,'bottom':0.1,'top':0.9,'hspace':0,'wspace':0.3})

        for ax in axes:
            ax.grid(True,which='major',alpha=1)

        axes[0].plot(binarypars['t'],binarypars['a_R'],c='k',lw=2.5)
        axes[0].plot(binarypars['t'],binarypars['a_R'],c='grey',lw=1.5)
        axes[0].set_ylabel(r'$a$ [pc]')
        axes[0].set_yscale('log')
        axes[0].set_ylim(10**-3.5,10**2)
        axes[0].set_xticklabels([])

        axes[1].plot(binarypars['t'],binarypars['e_t'],c='k',lw=2.5)
        axes[1].plot(binarypars['t'],binarypars['e_t'],c='grey',lw=1.5)
        axes[1].set_ylabel(r'$e$')
        axes[1].set_ylim(0,1.1)

        tlims=(binarypars['t'][0]-1e6,binarypars['t'][-1]+1e6)
        axes[0].set_xlim(tlims)
        axes[1].set_xlim(tlims)
        axes[1].set_xlabel(r'$t$ [Gyr]')

        if not os.path.exists(os.getcwd()+'/plots/'):
            os.mkdir(os.getcwd()+'/plots/')

        fig.suptitle(f'Binary: ID1: {binary[0]} and ID2: {binary[1]}\n {"/".join(simulation.snapshots[0].snapshot_file.split("/")[:-1])}',fontsize=8)
        
        fig.set_dpi(dpi)
        plt.savefig(f'plots/ketjubinary_{binary[0]}_{binary[1]}.png',dpi=dpi)
        plt.show()
    



