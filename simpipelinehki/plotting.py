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
import scipy
import multiprocessing
import matplotlib
import matplotlib.pyplot as plt
# import sphviewer
import moviepy.video.io.ImageSequenceClip
from .tools import split_list

# Default matplotlib settings and color selections
dpi=1000

# Define cmaps for gas and stars
cmapname_gas='magma'
cmap_gas = plt.get_cmap(cmapname_gas, 256)
cmaplist_gas = cmap_gas(np.linspace(0, 1, 256))
for ival,cmapval in enumerate(cmaplist_gas):
    hsv=matplotlib.colors.rgb_to_hsv(cmapval[:3])
    cmaplist_gas[ival,:3] = matplotlib.colors.hsv_to_rgb(hsv)
    cmaplist_gas[ival,-1] = (ival+1)/256
cmap_gas = matplotlib.colors.ListedColormap(cmaplist_gas)

cname_star='#FDF9E8'
#make a colormap for the stars with this color
cmaplist_star = np.zeros((256,4))
cmaplist_star[:,-1]=np.linspace(0,1,256)
cmaplist_star[:,:3]=matplotlib.colors.hex2color(cname_star)
cmap_star = matplotlib.colors.ListedColormap(cmaplist_star)

############ TIME SERIES DATA ############

# This function is used to plot the evolution of the properties of a galaxy specified by its ID.
def plot_glxevol(simulation,radstr='2p00restar',id=None):
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

    try:
        bhdetails=simulation.bhdetails
        plotbh=True
    except:
        plotbh=False
    
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
    if plotbh:
        bhdetails_masked=bhdetails[id]

    #kernels
    snapkernel=np.ones(1)/1;bhkernel=np.ones(10)/10

    #times
    snaptime=np.convolve(galaxy_masked['Time'].values,snapkernel,mode='valid')
    if plotbh:
        bhtime=np.convolve(bhdetails_masked['Time'].values,bhkernel,mode='valid')

    #mass
    mass_star=np.convolve(galaxy_masked[f'{radstr}_sphere_star_tot'].values,snapkernel,mode='valid')
    mass_gas=np.convolve(galaxy_masked[f'{radstr}_sphere_gas_tot'].values,snapkernel,mode='valid')
    if plotbh:
        mass_bh=np.convolve(bhdetails_masked['bh_M'].values,bhkernel,mode='valid')

    #sfr and outflow/inflow
    sfr=np.convolve(galaxy_masked[f'{radstr}_sphere_gas_sfr'].values,snapkernel,mode='valid')
    inflow=np.convolve(galaxy_masked[f'{radstr}_shell_gasinflow_all_mdot'].values,snapkernel,mode='valid')
    outflow=np.convolve(galaxy_masked[f'{radstr}_shell_gasoutflow_all_mdot'].values,snapkernel,mode='valid')
    
    #figure
    fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(6,2.5),gridspec_kw={'left':0.15,'right':0.95,'bottom':0.1,'top':0.95,'hspace':0.2,'wspace':0.35})
    fig.set_dpi(dpi)
    for ax in axes:
        ax.grid(True,which='major',alpha=1)

    #mass
    axes[0].plot(snaptime,mass_star,c='k',lw=2.5,alpha=0.75)
    axes[0].plot(snaptime,mass_star,c='goldenrod',lw=1.5, label=r'$M_{\star}$'+f'({radstr})')
    axes[0].plot(snaptime,mass_gas,c='k',lw=2.5,alpha=0.75)
    axes[0].plot(snaptime,mass_gas,c='C0',lw=1.5, label=r'$M_{\rm gas}$'+f'({radstr})')

    if plotbh:
        axes[0].plot(bhtime,mass_bh,c='grey',lw=2.5,alpha=1)
        axes[0].plot(bhtime,mass_bh,c='k',lw=1.5,alpha=1, label=r'$M_{\rm BH}$'+f' ({int(id)})')

    axes[0].set_xlabel(r'$t\, {\rm [Gyr]}$')
    axes[0].set_xlim(snaptime[0],snaptime[-1])
    axes[0].set_ylabel(r'$M\, [{\rm M}_{\odot}]$')
    axes[0].set_yscale('log')
    axes[0].legend(loc='lower center',prop={'size':6})

    #sfr
    axes[1].plot(snaptime,sfr,c='k',lw=2.5,alpha=0.75)
    axes[1].plot(snaptime,sfr,c='grey',lw=1.5, label=r'SFR'+f'({radstr})')

    axes[1].plot(snaptime,inflow,c='k',lw=2.5,alpha=0.75)
    axes[1].plot(snaptime,inflow,c='C0',lw=1.5, label=r'$\dot{M}_{\rm in}$' +f'({radstr})')

    axes[1].plot(snaptime,outflow,c='k',lw=2.5,alpha=0.75)
    axes[1].plot(snaptime,outflow,c='C1',lw=1.5, label=r'$\dot{M}_{\rm out}$' +f'({radstr})')


    axes[1].set_xlabel(r'$t\, {\rm [Gyr]}$')
    axes[1].set_xlim(snaptime[0],snaptime[-1])
    axes[1].set_ylabel(r'$\dot{M}$ [${\rm M}_{\odot}\,{\rm yr}^{-1}$]')
    axes[1].set_yscale('log')
    axes[1].legend(loc='lower center',prop={'size':6})

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
    bhid_prim=[bh for bh in bhids if bhids_shape[bh]==np.max(list(bhids_shape.values()))][0]
    bhid_sec=[bh for bh in bhids if bh!=bhid_prim][0]

    print(bhid_prim)
    print(bhid_sec)

    #match time-step from secondary to primary
    time_sec=bhdetails_masked[bhid_sec]['Time'].values
    time_prim=bhdetails_masked[bhid_prim]['Time'].values
    sep_bh=np.zeros(time_sec.shape[0])
    vel_bh=np.zeros(time_sec.shape[0])

    #for each time in the secondary, find the idx of the closest time in the primary and get sep/vel at that idx
    for itime,time in enumerate(time_sec):
        idx_prim=np.argmin(np.abs(time_prim-time))
        xyz_sec=bhdetails_masked[bhid_sec].loc[:,['Coordinates_x','Coordinates_y','Coordinates_z']].values[itime,:]
        xyz_rem=bhdetails_masked[bhid_prim].loc[:,['Coordinates_x','Coordinates_y','Coordinates_z']].values[idx_prim,:]
        vel_sec=bhdetails_masked[bhid_sec].loc[:,['V_x','V_y','V_z']].values[itime,:]/bhdetails_masked[bhid_sec]['ScaleFactor'].values[itime]
        vel_rem=bhdetails_masked[bhid_prim].loc[:,['V_x','V_y','V_z']].values[idx_prim,:]/bhdetails_masked[bhid_prim]['ScaleFactor'].values[idx_prim]
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
    # axes[0].set_ylim(10**-5,10**3)

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
def render_snap(snapshot,type='baryons',frame=None,center=None,staralpha=0.03,clims=None):
    """
    Render a snapshot of the simulation.

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

    #set up the rendering parameters based on type
    if type=='baryons':
        ptypes=[0,4,1,5]
        cmap=cmap_gas
    elif type=='stars':
        ptypes=[4,5]
        cmap=cmap_star
    elif type=='dm':
        ptypes=[1,0,4,5]
        cmap='viridis'
    else:
        print('Type not recognized. Options are "baryons", "stars" and "dm".')
        return
    

    
    #find frame and center based on particle positions
    if frame is None:
        #box size
        max_x=snapshot.boxsize/2
        min_x=-snapshot.boxsize/2
        frame=(max_x-min_x)/2
        if snapshot.cosmorun:
            frame/=2

    if frame =='dynamic':
        #dynamic frame size based on bh positions
        pdata=snapshot.get_particle_data(keys=['Coordinates','Masses'], types=[5])

        center=np.array([0,0,0])  #center is at the origin
        if pdata.shape[0]>=1:
            center= np.mean(pdata.loc[:,[f'Coordinates_x','Coordinates_y','Coordinates_z']].values,axis=0)
        
        if pdata.shape[0]<=1:
            print('No black holes found in the snapshot. Using default frame size.')
            frame=20
        else:
            #find the maximum distance of the black holes from the center
            max_distance=np.max(np.sqrt(np.sum((pdata.loc[:,[f'Coordinates_{x}' for x in 'xyz']].values-center)**2,axis=1)))
            frame=max_distance*1+20

    staralpha=staralpha*(frame/100)**-1

    #get particle data
    pdata=snapshot.get_particle_data(keys=['Coordinates','Masses'], types=ptypes)

    #make 2d histogram
    ptype_mask=pdata['ParticleTypes'].values==ptypes[0]
    coordinates=pdata.loc[ptype_mask,[f'Coordinates_{x}' for x in 'xyz']].values-center
    masses=pdata.loc[ptype_mask,'Masses'].values

    bins=np.linspace(-frame,frame,1025)
    binarea=(bins[1]-bins[0])**2

    #2d histogram
    surface_density=scipy.stats.binned_statistic_2d(coordinates[:,0],coordinates[:,1],masses,bins=[bins]*2,statistic='sum')[0]/binarea

    # old code for sphviewer
    # sph_particles=sphviewer.Particles(pdata.loc[sph_fluidmask,[f'Coordinates_{x}' for x in 'xyz']].values,
    #                                   pdata.loc[sph_fluidmask,'Masses'].values,nb=8)
    
    # sph_camera = sphviewer.Camera(r='infinity', t=0, p=0, roll=0, xsize=1500, ysize=1500,
    #                                             x=0, y=0, z=0,
    #                                             extent=[-frame,frame,-frame,frame])
    # sph_scene = sphviewer.Scene(sph_particles,sph_camera)
    # sph_render = sphviewer.Render(sph_scene)
    # sph_extent = sph_render.get_extent()
    # sph_img=sph_render.get_image()

    if clims:
        norm=matplotlib.colors.LogNorm(*clims)
    else:
        norm=matplotlib.colors.LogNorm()
        
    #make figure and plot
    fig,ax=plt.subplots(1,1,figsize=(5,5),gridspec_kw={'left':0.1,'right':0.99,'bottom':0.1,'top':0.99})
    ax.set_facecolor('k')
    ax.grid(which='both',alpha=0)
    ax.imshow(surface_density.T,extent=[-frame,frame,-frame,frame],origin='lower',cmap=cmap,norm=norm,zorder=1)

    #add stars to baryonic plot
    if type=='baryons':
        stars=pdata.loc[pdata['ParticleTypes'].values==4,:]
        ax.scatter(stars.loc[:,'Coordinates_x'].values-center[0],stars.loc[:,'Coordinates_y'].values-center[1],c=cname_star,alpha=staralpha,s=0.05,lw=0,zorder=2)

    #plot position of BHs if baryonic particles are being plotted
    blackholes=pdata.loc[pdata['ParticleTypes'].values==5,:]
    blackholes[f'Coordinates_x']=blackholes['Coordinates_x'].values-center[0]
    blackholes[f'Coordinates_y']=blackholes['Coordinates_y'].values-center[1]
    blackholes[f'Coordinates_z']=blackholes['Coordinates_z'].values-center[2]
    if blackholes.shape[0]:
        for ibh,bh in blackholes.iterrows():
            ax.add_artist(plt.Circle(radius=0.15,xy=[bh[f'Coordinates_x'],bh[f'Coordinates_y']],color='w',lw=1,ls='-',fill=True,zorder=5))
            ax.add_artist(plt.Circle(radius=0.10,xy=[bh[f'Coordinates_x'],bh[f'Coordinates_y']],color='k',lw=0.5,ls='-',fill=True,zorder=5))
    if 'mwm31' in snapshot.snapshot_file:
        bhloc1=blackholes.loc[blackholes['Masses'].values==np.nanmax(blackholes['Masses'].values),:]
        bhloc2=blackholes.loc[blackholes['Masses'].values==np.nanmin(blackholes['Masses'].values),:]

        separation= np.sqrt((bhloc1['Coordinates_x'].values-bhloc2['Coordinates_x'].values)**2 + (bhloc1['Coordinates_y'].values-bhloc2['Coordinates_y'].values)**2)
        if blackholes.shape[0]==2 and separation>0.2:
            m31=blackholes.loc[blackholes['Masses'].values==np.nanmax(blackholes['Masses'].values),:]
            mw=blackholes.loc[blackholes['Masses'].values==np.nanmin(blackholes['Masses'].values),:]
            ax.add_artist(plt.Circle(radius=9,xy=[m31['Coordinates_x'],m31['Coordinates_y']],fc='none',ec='lightgrey',lw=0.751,ls='--',fill=True,zorder=5,alpha=0.5))
            ax.add_artist(plt.Circle(radius=6,xy=[mw['Coordinates_x'],mw['Coordinates_y']],fc='none',ec='lightgrey',lw=0.751,ls='--',fill=True,zorder=5,alpha=0.5))
            ax.text(x=m31['Coordinates_x'],y=m31['Coordinates_y']+9+(frame/20)*0.05,s='M31',color='lightgrey',ha='center',va='bottom',zorder=6,alpha=0.5)
            ax.text(x=mw['Coordinates_x'],y=mw['Coordinates_y']+6+(frame/20)*0.05,s='MW',color='lightgrey',ha='center',va='bottom',zorder=6,alpha=0.5)
        else:
            bh=blackholes.iloc[0,:]
            ax.add_artist(plt.Circle(radius=9,xy=[bh[f'Coordinates_x'],bh[f'Coordinates_y']],fc='none',ec='lightgrey',lw=0.751,ls='--',fill=True,zorder=5,alpha=0.5))
            ax.text(x=bh[f'Coordinates_x'],y=bh[f'Coordinates_y']+9+(frame/20)*0.05,s='Milkdromeda',color='lightgrey',ha='center',va='bottom',zorder=6,alpha=0.5)


    ax.set_xlim(-frame,frame)
    ax.set_ylim(-frame,frame)

    ax.text(0.55,0.01,'$x$ [kpc]',transform=fig.transFigure,ha='center',va='bottom')
    ax.text(0.01,0.55,'$y$ [kpc]',transform=fig.transFigure,ha='left',va='center',rotation=90)
    ax.text(x=0.05,y=0.95,s=r'$t='+f'{snapshot.time:.3f}$ Gyr',transform=ax.transAxes,ha='left',va='top',color='w')
    #if a cosmological sim, also add the redshift
    if snapshot.cosmorun:
        ax.text(x=0.95,y=0.95,s=r'$z='+f'{snapshot.redshift:.3f}$',transform=ax.transAxes,ha='right',va='top',color='w')
    else:
        if snapshot.snapshot_idx: 
            ax.text(x=0.95,y=0.95,s=r'Snap '+f'{snapshot.snapshot_idx:.0f}',transform=ax.transAxes,ha='right',va='top',color='w')

    fig.set_dpi(dpi)

    return fig,ax


def render_sim_worker(snaplist,type='baryons',frame=None,staralpha=1,clims=None):
    
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
        print(f"Rendering snap {snapshot.snapshot_idx}...")
        fig,_=render_snap(snapshot,type=type,frame=frame,staralpha=staralpha,clims=clims)
        foutput= snapshot.snapshot_file.split('output')[0]+'/animations/render_sim_'+type+'/'
        fig.savefig(f'{foutput}/snap_{str(snapshot.snapshot_idx).zfill(3)}.png',dpi=dpi)
        plt.close(fig)


def gen_sim_animation(simulation,path=None,numproc=1,fps=10,type='baryons',frame=None,staralpha=0.03,clims=None):
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
    verbose: bool
        If True, print the progress of the rendering.

    """

    #make a directory for the outputs; if it already exists, remove the files
    if path is None:
        image_folder=simulation.snapshot_flist[0].split('output')[0]+'/animations/render_sim_'+type+'/'
        if not os.path.exists(f'{image_folder}'):
            os.mkdir(f'{image_folder}')
        else:
            for fname in os.listdir(image_folder):
                if os.path.exists(image_folder+fname):
                    os.remove(image_folder+fname)
    
    #split the snapshots into chunks for multiprocessing
    snapshot_list=simulation.snapshot_list
    snapshots_chunks=split_list(snapshot_list,numproc)

    #start the processes
    procs=[]
    for iproc in range(numproc):
        time.sleep(0.1)
        snapshots_ichunk=snapshots_chunks[iproc]
        proc = multiprocessing.Process(target=render_sim_worker, args=(snapshots_ichunk,type,frame,staralpha,clims))
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
def render_merger_worker(snaplist,galaxies,ids=None,staralpha=0.03,clims=None,useminpot=False,verbose=False):
    
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
        haloids_unique=h5py.File(snaplist[0].snapshot_file,'r')['PartType5']['ParticleIDs'][:][:2]
    else:
        haloids_unique=ids

    #id remaining
    remnantid=haloids_unique[np.nanargmax([galaxies.loc[galaxies['ID'].values==haloid,:].shape[0] for haloid in haloids_unique])]
    #pick the other id
    secondid=haloids_unique[1- np.nanargmax([galaxies.loc[galaxies['ID'].values==haloid,:].shape[0] for haloid in haloids_unique])]

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

        pdata=snapshot.get_particle_data(keys=['Coordinates','Masses'], types=[0,4,5], center=None, radius=None)
        stars=pdata.loc[pdata['ParticleTypes'].values==4,:]
        gas=pdata.loc[pdata['ParticleTypes'].values==0,:]

        #gas 2d histogram
        bins=np.linspace(-frame,frame,513)
        binarea=(bins[1]-bins[0])**2
        surface_density=scipy.stats.binned_statistic_2d(gas['Coordinates_x'].values,gas['Coordinates_y'].values,gas['Masses'].values,bins=[bins]*2,statistic='sum')[0]/binarea

        #plot the gas
        ax.pcolormesh(bins,bins,surface_density.T,cmap=cmap_gas,norm=matplotlib.colors.LogNorm(),zorder=1)

        # sph_particles=sphviewer.Particles(gas.loc[:,[f'Coordinates_{x}' for x in 'xyz']].values-center,gas['Masses'].values,nb=32)
        # sph_camera = sphviewer.Camera(r='infinity', t=0, p=0, roll=0, xsize=1500, ysize=1500,
        #                                             x=0, y=0, z=0,
        #                                             extent=[-frame,frame,-frame,frame])
        # sph_scene=sphviewer.Scene(sph_particles,sph_camera)
        # sph_render = sphviewer.Render(sph_scene)
        # sph_extent = sph_render.get_extent()
        # sph_img=sph_render.get_image()
        # sph_img[sph_img==0]

        ax.fill_between([-200,200],[-200,-200],[200,200],color='k',alpha=1,zorder=0)
        if clims:
            norm=matplotlib.colors.LogNorm(*clims)
        else:
            norm=matplotlib.colors.LogNorm()
        # ax.imshow(sph_img,extent=sph_extent,origin='lower',cmap=cmap_gas,norm=norm,zorder=1)

        ax.scatter(stars.loc[:,'Coordinates_x'].values-center[0],stars.loc[:,'Coordinates_y'].values-center[1],c=cname_star,alpha=staralpha,s=0.05,lw=0,zorder=2)

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
def gen_merger_animation(simulation,numproc=1,ids=None,fps=10,staralpha=10,clims=None,useminpot=False,verbose=False):

    """
    Render an animation of the interaction between two galaxies specified by their IDs.

    Parameters
    ----------
    simulation : simulation object
        Simulation object containing the data to be plotted.
        NB: The simulation object *must* contain the following data:
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
        try:
            snapshot_list.append(simulation.snapshots[int(isnap_last+i)])
        except:
            pass


    #split for computation
    snapshots_chunks=split_list(snapshot_list,numproc)

    procs=[]
    for iproc in range(numproc):
        time.sleep(0.1)
        snapshots_ichunk=snapshots_chunks[iproc]
        if verbose:
            print(f'Process {iproc} getting snaps: ', [snapshot.snapshot_idx for snapshot in snapshots_ichunk])            
        proc = multiprocessing.Process(target=render_merger_worker, args=(snapshots_ichunk,simulation.galaxies,ids,staralpha,clims,useminpot,verbose))
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

        binarypars['t']=(binarypars['t']-binarypars['t'][0])/1e9

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

        tlims=(binarypars['t'][0]-0.01,binarypars['t'][-1]+0.01)
        axes[0].set_xlim(tlims)
        axes[1].set_xlim(tlims)
        axes[1].set_xlabel(r'$t$ [Gyr]')

        if not os.path.exists(os.getcwd()+'/plots/'):
            os.mkdir(os.getcwd()+'/plots/')

        fig.suptitle(f'Binary -- ID1: {binary[0]} and ID2: {binary[1]}',fontsize=8)
        
        fig.set_dpi(dpi)
        plt.savefig(f'plots/ketjubinary_{binary[0]}_{binary[1]}.png',dpi=dpi)
        plt.show()
    
    

