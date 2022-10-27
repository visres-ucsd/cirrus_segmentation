from pathlib import Path
import pandas as pd
import cv2
import numpy as np
from p_tqdm import p_map
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage.measure import label
from skimage import morphology
# from Loess import Loess
import argparse
import scipy


MASK_RGB = np.array([133, 248, 208])
DIST_THRESHOLD = 5000
EXPECTED_SHAPE = (384, 496)

def bool_mask(img_arr):
    dist = (img_arr - MASK_RGB).sum(axis=2)**2
    mask = (dist < DIST_THRESHOLD)
    return mask

def getLargestCC(segmentation):
    '''Returns the largest connected component in a boolean segmentation'''
    labels = label(segmentation)
    largestCC = labels == np.argmax(np.bincount(labels.flat, weights=segmentation.flat))
    return largestCC

def imclose(img_arr, kernel_size):
    # kernel_size = (10,10) # matlab is radius, cv2 is diameter: https://stackoverflow.com/q/62112076/13710014
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    closed = cv2.morphologyEx(img_arr, cv2.MORPH_CLOSE, kernel)
    return closed

# def loess_smoothing(series):
#     '''Performs LOESS smoothing on a pandas Series.'''
#     estimator = Loess(series.index, series.values)
#     window = int(series.size * .05)
#     pred = pd.Series([estimator.estimate(x, window) for x in series.index])
#     pred.index = series.index
#     return pred

def morph_operations(img_arr):
    '''Performs morphological operations on masks to remove noise.'''
    # img_arr = imclose(img_arr.astype(np.uint8), kernel_size=(5,5)) # connect close components
    img_arr = morphology.remove_small_objects(img_arr.astype(bool), min_size=4000, connectivity=2)
    img_arr = imclose(img_arr.astype(np.uint8), kernel_size=(30,30)) # connect farther components
    return img_arr

def compute_top_layer(img_arr):
    img_arr = bool_mask(img_arr)
    # img_arr = getLargestCC(img_arr)
    img_arr = morph_operations(img_arr)
    rows, columns = np.where(img_arr)
    top_layer_idxs = pd.DataFrame({'row': rows, 'column': columns}).groupby('column').row.min()
    # top_layer_idxs = loess_smoothing(top_layer_idxs)
    return top_layer_idxs

def compute_reference_layer(img):
    if isinstance(img, str):
        if not Path(img).exists():
            raise FileNotFoundError(img)
        img = get_stitched_image(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.fastNlMeansDenoising(img, h=7)
    img = cv2.GaussianBlur(img, (13,13), 0)
    img = cv2.normalize(img, None, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    sobely = cv2.Sobel(src=img, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=5)
    sobely = cv2.GaussianBlur(sobely, (13,13), 0)
    sobel_threshold = sobely >= 2
    sobel_morph = morphology.remove_small_objects(sobel_threshold, min_size=300, connectivity=2)
    sobel_morph = imclose(sobel_morph.astype(np.uint8), (10, 10))
    rows, columns = np.where(sobel_morph)
    top_layer_idxs = pd.DataFrame({'row': rows, 'column': columns}).groupby('column').row.min()
    return top_layer_idxs

def get_RNFL_thickness(ILM_mask_path):
    RNFL_mask_path = ILM_mask_path.replace('ILM', 'RNFL')
    assert Path(ILM_mask_path).exists(), 'ILM mask not found.'
    assert Path(RNFL_mask_path).exists(), 'RNFL mask not found.'

    RNFL_img = get_stitched_image(RNFL_mask_path)
    ILM_img = get_stitched_image(ILM_mask_path)
    
    RNFL_layer, ILM_layer = compute_top_layer(RNFL_img), compute_top_layer(ILM_img)
    
    RNFL_thickness = RNFL_layer - ILM_layer
    RNFL_thickness = RNFL_thickness[RNFL_thickness > 0].dropna() # drop where RNFL above ILM

    return ILM_layer, RNFL_layer, RNFL_thickness
    
def get_stitched_image(img_fp):
    if isinstance(img_fp, Path):
        img_fp = str(img_fp)
    is_left = '_L' in img_fp
    if is_left:
        left_fp = img_fp
        right_fp = img_fp.replace('_L', '_R')
    else:
        left_fp = img_fp.replace('_R', '_L')
        right_fp = img_fp
    
    right = cv2.imread(right_fp)
    left = cv2.imread(left_fp)
    
    together = np.hstack([left, np.fliplr(right)])

    return together

def process_mask_dir(ILM_mask_dir):
    L_images = [i for i in Path(ILM_mask_dir).glob('*_L.jpg')]
    layer_idxs = [int(p.name.split('_')[1]) for p in L_images]

    RNFL_layers = {}
    ILM_layers = {}
    thickness_values = {}

    # collect layer data
    for layer_idx, image_path in zip(layer_idxs, L_images):
        ILM_layer, RNFL_layer, RNFL_thickness =  get_RNFL_thickness(str(image_path))

        RNFL_layers[layer_idx] = RNFL_layer
        ILM_layers[layer_idx] = ILM_layer
        thickness_values[layer_idx] = RNFL_thickness

    # build layer dfs
    RNFL_df = pd.concat(RNFL_layers, axis=1).T.sort_index().sort_index(axis=1)
    ILM_df = pd.concat(ILM_layers, axis=1).T.sort_index().sort_index(axis=1)
    # thickness_df = pd.concat(thickness_values, axis=1).T.sort_index().sort_index(axis=1)
    thickness_df = surface_smoothing(RNFL_df) - surface_smoothing(ILM_df)
    thickness_df = thickness_df[thickness_df>=0]

    out_folder = Path('data_wd/layer_maps/')
    scan_rep = Path(ILM_mask_dir).name
    out_folder.mkdir(parents=True, exist_ok=True)

    RNFL_df.to_csv(out_folder.joinpath(f'{scan_rep}_RNFL_location.csv'))
    ILM_df.to_csv(out_folder.joinpath(f'{scan_rep}_ILM_location.csv'))
    thickness_df.to_csv(out_folder.joinpath(f'{scan_rep}_RNFL_thickness.csv'))

def surface_smoothing(layer_df):
    layer_df = scipy.signal.medfilt2d(layer_df, 3)
    layer_df = cv2.resize(layer_df, (200,200))
    layer_df = pd.DataFrame(layer_df)
    return layer_df

def main():
    '''
    Performs post-processing on all scans indicated by their ILM mask.
    For each scan, all ILM and RNFL masks should be available in `data_wd/ILM_masks/` or 
    `data_wd/RNFL_masks/` respectively.
    
    Writes outputs to `data_wd/layer_maps/`.
    '''
    parser = argparse.ArgumentParser(description='Cirrus Segmentation post-processing')
    parser.add_argument('--mask_dir_list', default=None)
    args = parser.parse_args()

    if args.mask_dir_list is None:
        ILM_mask_dirs = pd.Series([p for p in Path('data_wd/ILM_masks/').glob('*/*')])
    else:
        ILM_mask_dirs = pd.read_csv(args.mask_dir_list, names=['dirs']).dirs
    
    p_map(process_mask_dir, ILM_mask_dirs)
    # RNFL_masks = pd.Series([p for p in Path('RNFL_masks/').rglob('*.jpg')])
    # ILM_masks = pd.Series([str(p) for p in Path('ILM_masks/').rglob('*.jpg')])
    # ILM_masks = pd.Series([str(p) for p in Path('sample_scans/').rglob('*.jpg')])
    # ILM_masks = ILM_masks.str.replace('sample_scans', 'ILM_masks')
    # ILM_masks = ILM_masks[~ILM_masks.str.contains('ipynb_checkpoints')]
    
    # csv_exists = Path('RNFL_thickness.csv').exists()
    
    # if csv_exists:
    #     processed_items = pd.read_csv(
    #         'RNFL_thickness.csv',
    #         names=['filepath', 'thickness'],
    #         skiprows=1
    #     ).set_index('filepath').thickness

    #     # remove processed items
    #     ILM_masks = ILM_masks[~ILM_masks.isin(processed_items.index)]
    
    # RNFL_thickness_vals = p_map(get_RNFL_thickness, ILM_masks)
    # RNFL_thickness_vals = pd.Series(RNFL_thickness_vals, index=ILM_masks)
    # use_header = not csv_exists
    # mode = 'a' if csv_exists else 'w'
    # RNFL_thickness_vals.to_csv('RNFL_thickness.csv', header=use_header, mode=mode)
    
    
if __name__=='__main__':
    main()