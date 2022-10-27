from pathlib import Path
import pandas as pd
import cv2
import numpy as np
from p_tqdm import p_map
import matplotlib.pyplot as plt

MASK_RGB = np.array([133, 248, 208])
DIST_THRESHOLD = 5000
EXPECTED_SHAPE = (384, 496)

def bool_mask(img_arr):
    dist = (img_arr - MASK_RGB).sum(axis=2)**2
    mask = (dist < DIST_THRESHOLD)
    return mask

def compute_top_layer(img_arr):
    rows, columns = np.where(bool_mask(img_arr))
    top_layer_idxs = pd.DataFrame({'row': rows, 'column': columns}).groupby('column').row.min()
    return top_layer_idxs

def get_RNFL_thickness(ILM_mask_path):
    ILM_mask = ILM_mask_path
    RNFL_mask = ILM_mask.replace('ILM_masks', 'RNFL_masks')
    RNFL_mask, ILM_mask = cv2.imread(RNFL_mask), cv2.imread(ILM_mask)
    RNFL_mask, ILM_mask = cv2.resize(RNFL_mask, EXPECTED_SHAPE), cv2.resize(ILM_mask, EXPECTED_SHAPE)

    RNFL_layer = compute_top_layer(RNFL_mask).dropna()
    ILM_layer = compute_top_layer(ILM_mask).dropna()
    
    diff = RNFL_layer - ILM_layer
    diff = diff[diff > 0] # drop where RNFL above ILM

    return diff.mean()

def plot_with_layer(rnfl_path, axes=None):
    ilm_path = rnfl_path.replace('RNFL_masks', 'ILM_masks')
    img_path = rnfl_path.replace('RNFL_masks', 'CirrusPNGs')
    name = Path(img_path).parent.name + '_' + Path(img_path).name
    
    rnfl_img = cv2.imread(rnfl_path)
    ilm_img = cv2.imread(ilm_path)
    img = cv2.imread(img_path)
    
    rnfl_img, ilm_img = cv2.resize(rnfl_img, EXPECTED_SHAPE), cv2.resize(ilm_img, EXPECTED_SHAPE)
    rnfl_layer, ilm_layer = compute_top_layer(rnfl_img), compute_top_layer(ilm_img)
    rnfl_img, ilm_img = cv2.cvtColor(rnfl_img, cv2.COLOR_BGR2RGB), cv2.cvtColor(ilm_img, cv2.COLOR_BGR2RGB)
    
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img)
    axes[0].scatter(ilm_layer.index, ilm_layer, label='ILM', color='c', s=1)
    axes[0].scatter(rnfl_layer.index, rnfl_layer, label='RNFL', color='r', s=1)
    axes[0].set_title('Raw')
    axes[0].set_ylabel(name)
    axes[0].legend()

    axes[1].imshow(ilm_img)
    axes[1].set_title('ILM Mask')
    axes[2].imshow(rnfl_img)
    axes[2].set_title('RNFL Mask')
    if axes is None:
        fig.savefig('figure.png', bbox_inches='tight', facecolor='w')

def main():
    ILM_masks = pd.Series([str(p) for p in Path('ILM_masks/').rglob('*.jpg')])
    ILM_masks = ILM_masks[~ILM_masks.str.contains('ipynb_checkpoints')]
    
    csv_exists = Path('RNFL_thickness.csv').exists()
    
    if csv_exists:
        processed_items = pd.read_csv(
            'RNFL_thickness.csv',
            names=['filepath', 'thickness'],
            skiprows=1
        ).set_index('filepath').thickness

        # remove processed items
        ILM_masks = ILM_masks[~ILM_masks.isin(processed_items.index)]
    
    RNFL_thickness_vals = p_map(get_RNFL_thickness, ILM_masks)
    RNFL_thickness_vals = pd.Series(RNFL_thickness_vals, index=ILM_masks)
    RNFL_thickness_vals.to_csv('RNFL_thickness.csv', header=(not csv_exists), mode='a')
    
    
if __name__=='__main__':
    main()