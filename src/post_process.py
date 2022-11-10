from pathlib import Path
import pandas as pd
import cv2
import numpy as np
from p_tqdm import p_map
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage.measure import label
from skimage import morphology
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

def morph_operations(img_arr):
    '''Performs morphological operations on masks to remove noise.'''
    # img_arr = imclose(img_arr.astype(np.uint8), kernel_size=(5,5)) # connect close components
    img_arr = morphology.remove_small_objects(img_arr.astype(bool), min_size=4000, connectivity=2)
    img_arr = imclose(img_arr.astype(np.uint8), kernel_size=(30,30)) # connect farther components
    return img_arr

def compute_top_layer(img_arr):
    img_arr = bool_mask(img_arr)
    img_arr = morph_operations(img_arr)
    rows, columns = np.where(img_arr)
    top_layer_idxs = pd.DataFrame({'row': rows, 'column': columns}).groupby('column').row.min()
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

def load_img_data(img_path):
    return np.fromfile(img_path, dtype=np.uint8).reshape(200, 1024, 200)

def get_projection_image(img_data):
    projection_image = img_data.mean(axis=1)
    return projection_image

def get_slab_image(img_data, surface_data_path):
    # copy array
    slab_array = np.full(rnfl_surface.shape, np.nan)

    # loop through x, y coords
    for x in range(0,200):
        for y in range(0,200):
            # at each x, y set values outside of slab to nan
            ilm_idx = ilm_surface.values[x,y]
            if np.isnan(ilm_idx):
                continue

            # slice slab and take mean
            z_idxs = np.arange(0,1024)
            inside_slab = np.logical_and((z_idxs >= ilm_idx), (z_idxs <= ilm_idx + slab_width))
            slab_array[x, y] = test_img[x, ~inside_slab, y].mean()
    return slab_array

EN_FACE_PIX_RESOLUTION = (200/6.) # 200px/6mm
BASE_DIAM = 3.46 * EN_FACE_PIX_RESOLUTION
OUTER_RADIUS = (BASE_DIAM + (.1 * EN_FACE_PIX_RESOLUTION)) / 2
INNER_RADIUS = (BASE_DIAM - (.1 * EN_FACE_PIX_RESOLUTION)) / 2

def derived_circle_scan(vol_data, ilm_surface, rnfl_surface, onh_center, mode):
    onh_center = np.array(onh_center)

    point = np.array([100,100])
    coords = np.stack(np.meshgrid(range(200), range(200)), axis=2)
    coord_diff = coords - point
    coord_dist = np.sqrt(np.sum((coord_diff)**2, axis=2))
    coord_mask = np.logical_and(INNER_RADIUS < coord_dist, coord_dist < OUTER_RADIUS)
    # plt.figure(figsize=(10,10))
    # plt.imshow(coord_mask)

    # diameter 115px * pi (also converts to degrees easily)
    anglular_resolution = 360
    coor_atan = np.arctan2(coord_diff[:,:,0], coord_diff[:,:,1])
    angle_bins = np.linspace(coor_atan.min(), coor_atan.max(), anglular_resolution)
    angles_binned = np.digitize(coor_atan, bins=angle_bins).astype(float) - 1
    bin_labels = np.unique(angles_binned)
    angles_binned[~coord_mask] = np.nan
    # plt.figure(figsize=(10,10))
    # plt.imshow(angles_binned)

    # order bin_labels in TSNIT order
    if mode=='OD':
        bin_labels = np.flip((bin_labels + 90) % 360)
    elif mode=='OS':
        bin_labels = (bin_labels - 90) % 360

    # debug print out of angular mappings, should follow:
    # OS: 270 --> 360/0 --> 90 --> 180 --> 270
    # OD: 90 --> 0/360 --> 270 --> 180 --> 90
    print(pd.Series({k: v for k, v in enumerate(bin_labels)})[[0, 90, 180, 270, 359]])

    # Take average across 
    der_circle_scan = []
    der_ilm_surface = {}
    der_rnfl_surface = {}
    for bin_idx, bin_id in enumerate(bin_labels):
        bin_coords = [i for i in zip(*np.where(angles_binned==bin_id))]
        bin_lum_values = np.array([vol_data[y, :, x] for y, x in bin_coords])
        der_circle_scan.append(bin_lum_values.mean(axis=0))

        ilm_values = np.array([ilm_surface[y, x] for y, x in bin_coords])
        der_ilm_surface[bin_idx] = np.nanmean(ilm_values)

        rnfl_values = np.array([rnfl_surface[y, x] for y, x in bin_coords])
        der_rnfl_surface[bin_idx] = np.nanmean(rnfl_values)

    der_circle_scan = np.flip(np.array(der_circle_scan).T, axis=0)
    der_circle_scan = cv2.resize(der_circle_scan, (3*1024, 1024))
    der_ilm_surface = pd.Series(der_ilm_surface).dropna()
    der_rnfl_surface = pd.Series(der_rnfl_surface).dropna()

    der_ilm_surface.index = der_ilm_surface.index * der_circle_scan.shape[1] / bin_labels.size
    der_rnfl_surface.index = der_rnfl_surface.index * der_circle_scan.shape[1] / bin_labels.size
    
    plt.figure(figsize=(10,5))
    plt.imshow(der_circle_scan, cmap='gray', aspect='auto')

    ilm_x_res = np.linspace(0, der_circle_scan.shape[1], der_ilm_surface.size)
    der_ilm_surface.plot(linewidth=1, color='cyan')
    der_rnfl_surface.plot(linewidth=1, color='r')

    # set ticks to scale
    xticks = np.array([0, 90, 180, 270, 360])
    xticks_pos = xticks * (der_circle_scan.shape[1] / 360)
    yticks = np.array([0, 1, 2, 3, 4, 5, 6])
    yticks_pos = yticks * (der_circle_scan.shape[0] / 6)
    plt.xticks(xticks_pos, ['T', 'S', 'N', 'I', 'T'])
    plt.yticks(yticks_pos, yticks)

    plt.ylabel('Millimeters (mm)')
    return der_circle_scan

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
    
    
if __name__=='__main__':
    main()