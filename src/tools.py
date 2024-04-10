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


