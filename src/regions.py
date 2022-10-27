import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# Scaling constants
MICRONS_PER_PIXEL = 2000. / 1024. # Cirrus resolution
VERT_SCALE = (1024./496.) # preprocessing rescales b-scans from 1024 x 200 to 496 x 768

# mask generated from `grid_cirrus.m`
mask = pd.read_csv('sector_masks.csv', header=None).values

# values for each region in the sector mask
sector_values = {
    'T_OD': [10,11],
    'N_OS': [10,11],
    'N_OD': [15, 14],
    'T_OS': [15, 14],
    'S': [16,9],
    'I': [13,12],
    'G': [9, 10, 11, 12, 13, 14, 15, 16],
}

def calc_sector_mean(sector_key, thickness_vals):
    '''
    Computes the mean sector RNFL thickness for a single sector.
    '''
    # gets sector mask
    sector_mask = np.zeros(mask.shape)
    for sector_val in sector_values[sector_key]:
        sector_mask = np.logical_or(sector_mask, mask==sector_val)

    # get sector thickness values
    thickness_vals = thickness_vals.copy()
    thickness_vals[~sector_mask] = np.nan
    return np.nanmean(thickness_vals)
    
def calc_sectors(path):
    '''
    Computes mean sector RNFL thickness for all sectors.
    '''
    OS_in_name = 'OS' in Path(path).name[:9]
    eye = 'OS' if OS_in_name else 'OD'
    thickness_map = pd.read_csv(path, index_col=0).values

    sector_thicknesses = {}
    for sector in ['S', 'I', 'T', 'N', 'G']:
        sector_key = f'{sector}_{eye}' if sector in ['T', 'N'] else sector
        sector_thicknesses[sector] = calc_sector_mean(sector_key, thickness_map)

    return sector_thicknesses

def get_sector_mean_table(thickness_map_dir):
    '''
    Computes the mean thickness for all thickness maps in the directory indicated by `thickness_map_dir`.

    Return DataFrame with each file/scan as rows, sector means as columns.
    '''
    thickness_map_paths = pd.Series([str(i) for i in Path(thickness_map_dir).glob('*thickness.csv')])
    sectors_df = pd.DataFrame({
        fp: calc_sectors(fp) for fp in tqdm(thickness_map_paths)
    }).T
    sectors_df = sectors_df * VERT_SCALE * MICRONS_PER_PIXEL
    return sectors_df
