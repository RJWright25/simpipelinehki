#       _                                   _              _               _               _  _              
#  ___ (_) _ __ ___     __ _  _ __    __ _ | | _   _  ___ (_) ___   _ __  (_) _ __    ___ | |(_) _ __    ___ 
# / __|| || '_ ` _ \   / _` || '_ \  / _` || || | | |/ __|| |/ __| | '_ \ | || '_ \  / _ \| || || '_ \  / _ \
# \__ \| || | | | | | | (_| || | | || (_| || || |_| |\__ \| |\__ \ | |_) || || |_) ||  __/| || || | | ||  __/
# |___/|_||_| |_| |_|  \__,_||_| |_| \__,_||_| \__, ||___/|_||___/ | .__/ |_|| .__/  \___||_||_||_| |_| \___|
#                                              |___/               |_|       |_|                             


# bhprocessing.py
# This file contains the functions to process the black hole data from KETJU simulations.

import os
import numpy as np
import pandas as pd

# This function is used to postprocess the blackhole details files. (credit: Shihong Liao reprocess.py)
def postprocess_bhdata(path=None,outpath='postprocessing/blackhole_details_post_processing'):
    """
    Postprocesses the black hole details files. This is from Shihong Liao's reprocess.py file.

    Parameters:
    -----------
    simulation: simulation object
        The simulation object for which the black hole details are to be postprocessed.
    path: str
        The path to the directory containing the black hole details files.
    """

    # if no path, get from simulation
    if not path:
        print('No path given. Exiting...')
        return None

    # Specify file path and target BH ids
    fileNum = 0
    fileName = f"{path}/blackhole_details/blackhole_details_{fileNum}.txt"

    while(os.path.isfile(fileName)):
        fileNum += 1
        fileName = f"{path}/blackhole_details/blackhole_details_{fileNum}.txt" 
    print('Total files found:', fileNum)

    BHDetails = {}
    # Load files
    for file_index in list(range(fileNum))[:]:
        if file_index % 10 == 0:
            print('Processing file:', file_index+1, '/', fileNum)

        fileName = f"{path}/blackhole_details/blackhole_details_{file_index}.txt"
        data = pd.read_csv(fileName,usecols=list(range(11)),delim_whitespace=True,header=None)
        # data[0] contains "BH=ID". Find the unique BH IDs in this file:
        # replace column with BHIDs
        data.loc[:,0] = data[0].str.extract('BH=(\d+)').values.flatten()
        data.dropna(axis=0,how='any',inplace=True) #
        data.loc[:,0] = data.loc[:,0].astype(np.int64)
        data.sort_values(by=[0],inplace=True,ignore_index=True)
        data.reset_index(drop=True,inplace=True)

        BHIDsInFile = [int(BHID) for BHID in data.loc[:,0].values if np.isfinite(np.float32(BHID))]
        BHIDsInFile = np.unique(BHIDsInFile)
        BHNum= len(BHIDsInFile)
        for ibh in range(BHNum):
            BHID=BHIDsInFile[ibh]
            
            firstidx=np.searchsorted(data.loc[:,0].values,BHID)
            if ibh==BHNum-1:
                lastidx=data.shape[0]
            else:
                lastidx=np.searchsorted(data.loc[:,0].values,BHIDsInFile[ibh+1])            

            select_data = data.iloc[firstidx:lastidx,:]

            #check that the id in the first and last row is the same
            if select_data.shape[0]>1:
                if not select_data.values[0,0]==select_data.values[select_data.shape[0]-1,0]:
                    print('Error: BHID mismatch')
                    print(select_data.loc[0,0],select_data.loc[select_data.shape[0]-1,0])
                    continue

            # print('BHID:', BHID, 'Number of rows:', select_data.shape[0])
       
            if not f'{BHID}' in BHDetails:
                BHDetails[f'{BHID}'] = select_data
            else:
                BHDetails[f"{BHID}"] = [BHDetails[f"{BHID}"],select_data]
                BHDetails[f"{BHID}"] = pd.concat(BHDetails[f"{BHID}"],ignore_index=True)
        
    #check first value of each column to see if it is a nan
    BHIDs = np.array(list(BHDetails.keys()))
    BHIDs = np.array([int(BHIDs[ibh]) for ibh in range(BHNum)])
    for ibh in range(len(BHIDs)):
        BHDetails[f"{BHIDs[ibh]}"]=BHDetails[f"{BHIDs[ibh]}"].dropna(axis=1,how='all')

    # Get the number of BHs
    BHNum = len(BHIDs)
    print('BH number = ', BHNum)

    #Remove string columns
    for ibh in range(BHNum):
        BHDetails[f"{BHIDs[ibh]}"] = BHDetails[f"{BHIDs[ibh]}"].drop(columns=[0])

    #print number of columns
    numcol=BHDetails[str(BHIDs[0])].shape[1]
    print('Number of columns:',BHDetails[str(BHIDs[0])].shape[1])
    
    # Add column names
    for ibh in range(BHNum):
        BHDetails[f"{BHIDs[ibh]}"].columns = ['Time', 'bh_M', 'bh_Mdot', 'rho', 'cs', 'gas_Vrel_tot', 'Coordinates_x', 'Coordinates_y', 'Coordinates_z', 'V_x', 'V_y', 'V_z', 'gas_Vrel_x', 'gas_Vrel_y', 'gas_Vrel_z', 'Flag_binary', 'companion_ID', 'bh_hsml'][:numcol]
    
    #ensure all are floats
    for ibh in range(BHNum):
        BHDetails[f"{BHIDs[ibh]}"]=BHDetails[f"{BHIDs[ibh]}"].astype(float)
        
    # Sort according to time
    for ibh in range(BHNum):
        BHDetails[f"{BHIDs[ibh]}"] = BHDetails[str((BHIDs[ibh]))].sort_values(by=['Time'])
        BHDetails[f"{BHIDs[ibh]}"].reset_index(inplace=True,drop=True)
        
    # Save files
    if not os.path.exists(f'{outpath}'):
        os.mkdir(f'{outpath}')
    else:
        # Remove all files in the directory
        files = os.listdir(f'{outpath}')
        for file in files:
            os.remove(f'{outpath}/{file}')

    for ibh in range(BHNum):
        fname = f'{outpath}/BH_{BHIDs[ibh]}.txt'
        BHDetails[str(BHIDs[ibh])].to_csv(fname, sep=' ', index=False, header=False)

    return BHDetails


# This function is used to read the black hole details from a file
def read_bhdata(simulation=None,path=None,bhids=None,subsample=1):
    """
    Reads the black hole details from a file.

    Parameters:
    -----------
    simulation: simulation object
        The simulation object for which the black hole details are to be read.
    path: str
        The path to the directory containing the black hole details files.
    bhids: lsit
        The IDs of the black hole to read (may not be all bhs).
    subsample: int
        The subsampling factor to use when reading the data.
    
    Returns:
    -----------
    bhdata : dict
        A dictionary containing a pandas dataframe of the data for each black hole.
        Keys are the black hole IDs.

    """
    if not simulation and not path:
        print('No simulation object found. Exiting...')
        return None
    
    if not simulation:
        hubble=0.71
    else:
        hubble=simulation.hubble


    if not path:
        path='postprocessing/blackhole_details_post_processing/'
        if os.path.exists(path):
            print(f'Using path {path} to read black hole details...')
        else:
            print('No path found. Exiting...')
            return None
    
    #find all the files in the directory
    bhfiles=np.array([path+'/'+fname for fname in os.listdir(path) if 'BH' in fname])

    #cull the list if not all BHs are requested
    if bhids:
        bhfiles=np.array([fname for fname in bhfiles if int(fname.split('/BH_')[-1].split('.txt')[0]) in bhids])
    else:
        bhids=np.array([int(fname.split('/BH_')[-1].split('.txt')[0]) for fname in bhfiles])

    #sort by bhid
    bhids=bhids[np.argsort(bhids)]
    bhfiles=bhfiles[np.argsort(bhids)]

    #columns
    fpath=path+f'/BH_{str(int(bhids[0]))}.txt'
    bhdata_ibh=pd.DataFrame(np.loadtxt(fpath,dtype=str)[::subsample,:].astype(float))
    bhdata_ibh.dropna(axis=1,how='all',inplace=True)
    numcol=bhdata_ibh.shape[-1]
    columns=np.array(['Time','bh_M','bh_Mdot','rho','cs','gas_Vrel_tot','Coordinates_x','Coordinates_y','Coordinates_z','V_x','V_y','V_z','gas_Vrel_x','gas_Vrel_y','gas_Vrel_z','Flag_binary','companion_ID','bh_hsml'])
    columns=columns[:numcol]
    
    #initialize the output
    bhdata={}

    #read the data
    for bhid in bhids:
        print(f'Reading black hole details for BHID = {bhid}...')
        fpath=path+f'/BH_{str(int(bhid))}.txt'
        bhdata_ibh=pd.DataFrame(np.loadtxt(fpath,dtype=str)[::subsample,:].astype(float))
        bhdata_ibh.dropna(axis=1,how='all',inplace=True)

        #assign columns
        bhdata_ibh.columns=columns
        bhdata_ibh['BH_ID']=np.ones(bhdata_ibh.shape[0])*int(bhid)

        if simulation==None:
            bhdata_ibh.loc[:,'ScaleFactor']=1
        else:
            if 'cosmo' in simulation.snapshot_type:
                bhdata_ibh['ScaleFactor']=bhdata_ibh['Time'].values
            else:
                bhdata_ibh['ScaleFactor']=np.ones(bhdata_ibh.shape[0])

        #convert to physical units
        bhdata_ibh['bh_M']=bhdata_ibh['bh_M']*1e10/hubble
        bhdata_ibh['bh_Mdot']=bhdata_ibh['bh_Mdot']#don't think this needs to be converted
        for key in [f'Coordinates_{x}' for x in 'xyz']:
            bhdata_ibh[key]=bhdata_ibh[key]*bhdata_ibh['ScaleFactor'].values/hubble

        if not simulation==None:
            #if cosmo sim, convert to universe age 
            if 'cosmo' in simulation.snapshot_type:
                redshifts=1/bhdata_ibh['ScaleFactor'].values-1
                bhdata_ibh['Time']=simulation.cosmology.age(redshifts).value

        #add to the output
        bhdata[bhid]=bhdata_ibh
        
    return bhdata



def read_ketjubhdata(simulation,path=None):
    """
    Load the black hole details and from a ketju file using the ketjugw package.

    Parameters:
    -----------
    simulation: simulation object
        The simulation object for which the black hole details are to be read.
    path: str
        The path to the ketju file.
    
    Returns:
    -----------
    ketjubhs : dict of IDs 
        The black hole data as ketjugw.particle objects.
    ketjubinaries : dict of ID pairs
        The binary orbital params from ketjugw.binary objects.

    """

    try:
        import ketjugw
        from ketjugw import load_hdf5
        from ketjugw import find_binaries
    except:
        print('ketjugw package not found.')
        return None
    
    if not path:
        path=simulation.snapshots[-1].snapshot_file.split('/')[:-1]
        path='/'.join(path)+'/ketju_bhs.hdf5'
        
    ketjubhs = load_hdf5(path)
    rawbinaries = find_binaries(ketjubhs,remove_unbound_gaps=True)
    ketjubinaries={}
    for bhids, bbh in rawbinaries.items():
        pars = ketjugw.orbital_parameters(*bbh)
        if pars['t'].shape[0]>10:
            ketjubinaries[bhids] = pars

        pars['t'] = pars['t']/ketjugw.units.yr
        pars['a_R'] = pars['a_R']/ketjugw.units.pc

    return ketjubhs,ketjubinaries


