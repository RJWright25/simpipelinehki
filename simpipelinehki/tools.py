#       _                                   _              _               _               _  _              
#  ___ (_) _ __ ___     __ _  _ __    __ _ | | _   _  ___ (_) ___   _ __  (_) _ __    ___ | |(_) _ __    ___ 
# / __|| || '_ ` _ \   / _` || '_ \  / _` || || | | |/ __|| |/ __| | '_ \ | || '_ \  / _ \| || || '_ \  / _ \
# \__ \| || | | | | | | (_| || | | || (_| || || |_| |\__ \| |\__ \ | |_) || || |_) ||  __/| || || | | ||  __/
# |___/|_||_| |_| |_|  \__,_||_| |_| \__,_||_| \__, ||___/|_||___/ | .__/ |_|| .__/  \___||_||_||_| |_| \___|
#                                              |___/               |_|       |_|                             


# tools.py
# This file contains some useful functions that are used in the main functions.

import os
import numpy as np
import pandas as pd

# This function is used to split a list into nproc parts.
def split_list(lst,nproc):
    """
    Splits a list into nproc parts.
    
    Parameters:
    -----------
    lst: list
        The list to be split.
    nproc: int
        The number of processes to split the list into.

    Returns:
    -----------
    list (of lists)
        A list of nproc lists, each containing a subset of the original list.
    
    """
    return [lst[i::nproc] for i in range(nproc)]

def read_hdf_chunks(directory):
    """
    
    Read the HDF5 files in the directory and concatenate them into a single DataFrame.

    Parameters:
    -----------
    directory: str
        The directory containing the HDF5 files.

    Returns:
    -----------
    DataFrame
        A DataFrame containing the concatenated data from the HDF5 files.

    
    """

    # Get the list of snapshot directories.
    snapdirs = [snapdir for snapdir in os.listdir(directory) if 'snap' in snapdir]

    # Get the list of files in the first snapshot directory.
    fnames=[]
    for snapdir in snapdirs:
        for fname in os.listdir(f'{directory}/{snapdir}'):
            if file.endswith('.hdf5'):
                fnames.append(f'{directory}/{snapdir}/{fname}')
        

    dfs=[pd.read_hdf(fname, key='chunk') for fname in fnames]
    df=pd.concat(dfs)

    #sort by time
    df.sort_values('Time',inplace=True)
    df.reset_index(drop=True,inplace=True)

    return df