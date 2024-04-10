#       _                                   _              _               _               _  _              
#  ___ (_) _ __ ___     __ _  _ __    __ _ | | _   _  ___ (_) ___   _ __  (_) _ __    ___ | |(_) _ __    ___ 
# / __|| || '_ ` _ \   / _` || '_ \  / _` || || | | |/ __|| |/ __| | '_ \ | || '_ \  / _ \| || || '_ \  / _ \
# \__ \| || | | | | | | (_| || | | || (_| || || |_| |\__ \| |\__ \ | |_) || || |_) ||  __/| || || | | ||  __/
# |___/|_||_| |_| |_|  \__,_||_| |_| \__,_||_| \__, ||___/|_||___/ | .__/ |_|| .__/  \___||_||_||_| |_| \___|
#                                              |___/               |_|       |_|                             


# groupfinder.py
# This file contains the function and tools to group galaxies in a snapshot.

import numpy as np
import pandas as pd

# This function is used to group galaxies in the same snapshot.
def basic_groupfinder(galaxies,verbose=False):
    
    # Sort by stellar mass to preference more massive galaxies as "central"
    galaxies.sort_values(by='1p00restar_sphere_star_tot',ascending=False,inplace=True)
    galaxies.reset_index(drop=True,inplace=True)
    isnaps=galaxies['isnap'].values

    for isnap in np.unique(isnaps):
        if verbose:
            print(f'Grouping galaxies in snapshot {int(isnap)}...')

        isnap_mask=galaxies['isnap'].values==isnap

        #initialise the output
        galaxies.loc[isnap_mask,'GroupID']=-1
        galaxies.loc[isnap_mask,'Central']=1
        galaxies.loc[isnap_mask,'RemnantFlag']=0
        galaxies.loc[isnap_mask,'RemnantPartner']=0        
        galaxies.loc[isnap_mask,'RemnantCentral']=-1
        galaxies.loc[isnap_mask,'RemnantSep']=-1
        galaxies.loc[isnap_mask,'CentralDist']=-1

        #galaxy properties
        galaxies_x=galaxies['x'].values
        galaxies_y=galaxies['y'].values
        galaxies_z=galaxies['z'].values
        galaxies_r200=galaxies['Halo_R_Crit200'].values
        galaxies_IDs=galaxies['ID'].values

        #loop over the galaxies ordered by stellar mass -- find whether any other haloes overlap their R200c
        iigal=-1
        for igal,gal in galaxies.loc[isnap_mask,:].iterrows():
            iigal+=1
            if verbose:
                print(f'Post-processing galaxy {iigal+1}/{galaxies.loc[isnap_mask,:].shape[0]} (ID={int(gal["ID"])})... in snap {isnap}')
            igal_ID=gal['ID']

            #skip galaxy if already grouped
            if gal['GroupID']>0:
                print(f'Skipping galaxy {igal_ID}: already grouped')
                continue

            #distances
            distances=np.sqrt((galaxies_x-galaxies_x[igal])**2+(galaxies_y-galaxies_y[igal])**2+(galaxies_z-galaxies_z[igal])**2)

            #### GROUP FINDING ####
            #find the potential group members as galaxies with overlapping r200c
            group_id=isnap*1e4+iigal
            group_mask=np.logical_and(isnap_mask,distances<1*galaxies_r200[igal])
            if np.nansum(group_mask):

                #assign group ID and find central galaxy
                central_status=np.zeros(np.nansum(group_mask));central_status[0]=1
                if np.nansum(group_mask):
                    galaxies.loc[group_mask,'GroupID']=group_id
                    galaxies.loc[group_mask,'Central']=central_status

            #### PARTNER FINDING ####
            #check if already a remnant
            if galaxies.loc[igal,'RemnantFlag']==1:
                continue
            else:
                #find the potential remnant partners as galaxies within 1kpc
                partner_mask=np.logical_and(distances<1,distances>=1e-4)
                partner_mask=np.logical_and(partner_mask,isnap_mask)

                if np.nansum(partner_mask):
                    galaxies.loc[partner_mask,'RemnantFlag']=1;galaxies.loc[igal,'RemnantFlag']=1
                    galaxies.loc[partner_mask,'RemnantPartner']=igal_ID;galaxies.loc[igal,'RemnantPartner']=galaxies_IDs[partner_mask][0]
                    galaxies.loc[partner_mask,'RemnantCentral']=0;galaxies.loc[igal,'RemnantPartner']=0

                    #find central as the galaxy with the largest BH mass
                    igal_bhmass=gal['BH_Mass'];igal_partner_bhmass=galaxies['BH_Mass'].values[partner_mask][0]
                    if igal_bhmass>igal_partner_bhmass:
                        galaxies.loc[igal,'RemnantCentral']=1
                    else:
                        galaxies.loc[partner_mask,'RemnantCentral']=1
                    #remnant sep
                    galaxies.loc[igal,'RemnantSep']=distances[partner_mask][0]
                    galaxies.loc[partner_mask,'RemnantSep']=distances[partner_mask][0]

        #get halo-centric distance for all satellites
        group_ids=np.unique(galaxies['GroupID'].values[isnap_mask])
        for group_id in group_ids:
            if group_id>0:
                group_mask=galaxies['GroupID'].values==group_id
                if np.nansum(group_mask):
                    galaxies_xyz=np.array([galaxies['x'].values[group_mask],galaxies['y'].values[group_mask],galaxies['z'].values[group_mask]]).T
                    galaxies.loc[group_mask,'CentralDist']=np.sqrt(np.sum((galaxies_xyz-galaxies_xyz[0,:])**2,axis=1))

    return galaxies
            