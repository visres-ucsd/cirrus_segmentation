from pathlib import Path
import pandas as pd
import cv2
import numpy as np
from p_tqdm import p_map
from functools import partial
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage.measure import label
from skimage import morphology
import argparse
import scipy
import base64
import json

MASK_RGB = np.array([133, 248, 208])
DIST_THRESHOLD = 5000
EXPECTED_SHAPE = (384, 496)
MICRONS_PER_PIXEL = 2000. / 1024.#3.3
VERT_SCALE = (1024./496.)
TEMP_IDX = 1

def b64str_encode(ndarr):
    return base64.b64encode(ndarr.tobytes()).decode('utf-8')

def b64_decode(b64str, dtype=np.int8, shape=None):
    decoded = np.frombuffer(base64.b64decode(b64str), dtype=dtype)
    if not shape is None:
        decoded = decoded.reshape(*shape)
    return decoded

def image_to_base64(img_arr):
    '''
    Converts image to a base64 string with JPG encoding.
    '''
    global TEMP_IDX
    img_arr = img_arr.copy()
    img_arr[np.isnan(img_arr)] = img_arr.max()
    
    temp_im_file = f'temp/img_to_str_{TEMP_IDX}.png'
    TEMP_IDX += 1 
    Path(temp_im_file).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(temp_im_file, img_arr)
    with open(temp_im_file, 'rb') as image_file:
        data = image_file.read()
    data = b64str_encode(data)
    Path(temp_im_file).unlink()
    return data

def read_img_base64(img_str, dtype=np.uint8):
    '''
    Loads an image saved as a base64 string.
    '''
    data = b64_decode(img_str, dtype=dtype)
    img = cv2.imdecode(data, flags=0)
    return img

def load_derived_json(json_path):
    with open(json_path) as json_handle:
        json_collection = json.load(json_handle)
    
    if not 'jpg_encoded' in json_collection: # assume jpg encoding
        json_collection['jpg_encoded'] = False

    if json_collection['jpg_encoded']:
        for im_key in ['derived_circle_scan', 'projection_image', 'en_face_slab_image']:
             # read base64 jpg data
            json_collection[im_key] = read_img_base64(json_collection[im_key])
    else:
        for im_key in ['projection_image', 'en_face_slab_image']:
             # read base64 numpy data
            json_collection[im_key] = b64_decode(json_collection[im_key], dtype=np.float16, shape=(200, 200))
        json_collection['derived_circle_scan'] = b64_decode(json_collection['derived_circle_scan'], dtype=np.float16, shape=(1024, 360))
    
    json_collection['rnfl_thickness_values'] = b64_decode(json_collection['rnfl_thickness_values'], dtype=np.float16, shape=(200, 200))

    # load surfaces as pandas series
    for key in ['derived_ilm_surface', 'derived_rnfl_surface']:
        json_collection[key] = pd.Series(json_collection[key])
        json_collection[key].index = json_collection[key].index.astype(float)
    
    json_collection = pd.Series(json_collection)
    return json_collection

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

def process_mask_dir(ILM_mask_dir, outdir='data_wd/layer_maps/'):
    L_images = [i for i in Path(ILM_mask_dir).glob('*_L.jpg')]
    layer_idxs = [int(p.name.split('_')[1]) for p in L_images]
    Path().glob('*_L.jpg')

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

    out_folder = Path(outdir)
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

def get_slab_image(img_data, ilm_surface_df):
    slab_width_microns = 52
    slab_width = np.ceil(slab_width_microns / MICRONS_PER_PIXEL)

    # copy array
    slab_array = np.full(ilm_surface_df.shape, np.nan)

    # loop through x, y coords
    for x in range(0,200):
        for y in range(0,200):
            # at each x, y set values outside of slab to nan
            ilm_idx = ilm_surface_df.values[x,y]
            if np.isnan(ilm_idx):
                continue

            # slice slab and take mean
            z_idxs = np.arange(0,1024)
            z_idxs = z_idxs * -1 - 1
            ilm_idx *= -1
            inside_slab = np.logical_and((z_idxs <= ilm_idx), (z_idxs >= ilm_idx - slab_width))
            z_idxs = z_idxs[inside_slab]
            # if x==0 and y==0: print(ilm_idx, inside_slab.sum()) # debug
            slab_array[x, y] = img_data[x, z_idxs, y].mean()
    return slab_array

EN_FACE_PIX_RESOLUTION = (200/6.) # 200px/6mm
BASE_DIAM = 3.46 * EN_FACE_PIX_RESOLUTION # 3.46mm in px
# BASE_DIAM = 3.5 * EN_FACE_PIX_RESOLUTION
# BASE_DIAM = 180
OUTER_RADIUS = (BASE_DIAM + (.1 * EN_FACE_PIX_RESOLUTION)) / 2
INNER_RADIUS = (BASE_DIAM - (.1 * EN_FACE_PIX_RESOLUTION)) / 2

def euc_dist(ref_point, arr, axis=-1):
    '''Euclidean distance function. Use axis to determine which axis to compute against.'''
    return np.linalg.norm(arr - ref_point, axis=axis)

DEFAULT_EN_FACE_CENTER = np.array([99, 99])

def make_derived_circle_scan(vol_data, ilm_surface, rnfl_surface, onh_center, mode):
    # print(f'{BASE_DIAM=}')
    onh_center = np.array(onh_center).round()

    # protect against onh_centers too far away from image center
    encounter_scan_boundary = (np.abs((onh_center - DEFAULT_EN_FACE_CENTER)) >= (100 - OUTER_RADIUS)).any()
    if encounter_scan_boundary:
        onh_center = DEFAULT_EN_FACE_CENTER

    coords = np.stack(np.meshgrid(range(200), range(200)), axis=2)
    coord_diff = coords - onh_center
    coord_dist = np.linalg.norm(coord_diff, axis=2)
    coord_mask = np.logical_and(INNER_RADIUS < coord_dist, coord_dist < OUTER_RADIUS)
    # plt.figure(figsize=(10,10))
    # plt.imshow(coord_mask)

    # diameter 115px * pi ~= 361
    anglular_resolution = 360 # (also converts to degrees easily)
    coor_atan = np.arctan2(coord_diff[:,:,0], coord_diff[:,:,1])
    angle_bins = np.linspace(coor_atan.min(), coor_atan.max(), anglular_resolution)
    angles_binned = np.digitize(coor_atan, bins=angle_bins).astype(float) - 1
    bin_labels = np.unique(angles_binned)
    angles_binned[~coord_mask] = np.nan
    # plt.figure(figsize=(10,10))
    # plt.imshow(angles_binned)

    # order bin_labels in TSNIT order
    if mode=='OS':
        bin_labels = np.flip((bin_labels + 90) % 360)
    elif mode=='OD':
        bin_labels = (bin_labels - 90) % 360
    # print(f'{mode}: {bin_labels[0]}, {bin_labels[90]}, {bin_labels[180]}, {bin_labels[270]}')

    # debug print out of angular mappings, should follow:
    # OS: 270 --> 360/0 --> 90 --> 180 --> 270
    # OD: 90 --> 0/360 --> 270 --> 180 --> 90
    # print(pd.Series({k: v for k, v in enumerate(bin_labels)})[[0, 90, 180, 270, 359]])

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
    
    try:
        der_circle_scan = np.flip(np.array(der_circle_scan).T, axis=0)#.astype(int)
        # der_circle_scan = cv2.resize(der_circle_scan, (3*1024, 1024))
        der_ilm_surface = pd.Series(der_ilm_surface)
        der_rnfl_surface = pd.Series(der_rnfl_surface)

        # der_ilm_surface.index = der_ilm_surface.index * der_circle_scan.shape[1] / bin_labels.size
        # der_rnfl_surface.index = der_rnfl_surface.index * der_circle_scan.shape[1] / bin_labels.size
    except cv2.error as e:
        print(f'Error building derived circle scan')
        print(f'ONH Center: {onh_center}')
        raise e
    
    # plt.figure(figsize=(10,5))
    # plt.imshow(der_circle_scan, cmap='gray', aspect='auto')

    # ilm_x_res = np.linspace(0, der_circle_scan.shape[1], der_ilm_surface.size)
    # der_ilm_surface.plot(linewidth=1, color='cyan')
    # der_rnfl_surface.plot(linewidth=1, color='r')

    # set ticks to scale
    # xticks = np.array([0, 90, 180, 270, 360])
    # xticks_pos = xticks * (der_circle_scan.shape[1] / 360)
    # yticks = np.array([0, 1, 2, 3, 4, 5, 6])
    # yticks_pos = yticks * (der_circle_scan.shape[0] / 6)
    # plt.xticks(xticks_pos, ['T', 'S', 'N', 'I', 'T'])
    # plt.yticks(yticks_pos, yticks)

    # plt.ylabel('Millimeters (mm)')
    return der_circle_scan, der_ilm_surface, der_rnfl_surface

def find_onh_center(rnfl_thickness_mat):
    rnfl_isna = np.isnan(rnfl_thickness_mat).astype(np.uint8)
    ret, labels = cv2.connectedComponents(rnfl_isna)
    center_component = labels == labels[99,99]
    center_component = imclose(center_component.astype(np.uint8), (5,5)).astype(bool)
    center_mean_y, center_mean_x = np.mean(np.where(center_component), axis=1)
    return center_mean_y, center_mean_x

def collect_json(img_path, savefig=False, jpg_encode=False):
    debug = False

    if debug:
        print('Step 1/6: Loading data...')
    pt_id, scan_type, scan_date, scan_time, scan_eye, _, _, _ = Path(img_path).name.split('_')
    scan_outname = '_'.join([pt_id+scan_eye, scan_date, scan_time])

    ilm_surface_path = Path('data_wd/layer_maps/').joinpath(scan_outname+'_ILM_location.csv')
    rnfl_surface_path = Path('data_wd/layer_maps/').joinpath(scan_outname+'_RNFL_location.csv')
    rnfl_thickness_path = Path('data_wd/layer_maps/').joinpath(scan_outname+'_RNFL_thickness.csv')
    try: # try to match a comparison spectralis circle scan
        # spectralis_raw_path = next(Path('raw-bscans').rglob(f'{pt_id[1:]}*bscan0024.tif'))
        spectralis_raw_path = None
    except StopIteration:
        spectralis_raw_path = None

    ilm_surface = pd.read_csv(ilm_surface_path, index_col=0)
    rnfl_surface = pd.read_csv(rnfl_surface_path, index_col=0)
    ilm_surface = (surface_smoothing(ilm_surface) * VERT_SCALE).round() 
    rnfl_surface = (surface_smoothing(rnfl_surface) * VERT_SCALE).round()

    rnfl_thickness_df = pd.read_csv(rnfl_thickness_path, index_col=0)
    rnfl_thickness_df = rnfl_thickness_df * VERT_SCALE * MICRONS_PER_PIXEL

    img_data = np.fromfile(img_path, dtype=np.uint8).reshape(200, 1024, 200)

    if debug:
        print('Step 2/6: Generating projection image...')
    proj_image = img_data.mean(axis=1)

    if debug:
        print('Step 3/6: Generating slab image...')
    slab_image = get_slab_image(img_data, ilm_surface)
    
    if debug:
        print('Step 4/6: Looking for onh center...')
    scan_center = find_onh_center(rnfl_thickness_df.values)
    if debug:
        print('Step 5/6: Generating derived circle image...')
    der_circle_scan, der_ilm_surface, der_rnfl_surface = make_derived_circle_scan(
        img_data,
        ilm_surface.values,
        rnfl_surface.values,
        scan_center,
        scan_eye
    )

    if debug:
        print('Step 6/6: Building JSON dictionary...')
    json_dict = {
        'img_path': str(img_path),
        'pt_id': pt_id,
        'scan_eye': scan_eye,
        'scan_type': scan_type,
        'scan_date': scan_date,
        'scan_time': scan_time,
        'scan_outname': scan_outname,
        'scan_center': list(scan_center),
        'derived_circle_scan': der_circle_scan,
        'projection_image': proj_image,
        'en_face_slab_image': slab_image,
        'derived_ilm_surface': der_ilm_surface.to_dict(),
        'derived_rnfl_surface': der_rnfl_surface.to_dict(),
        'rnfl_thickness_values': rnfl_thickness_df.values,
        'spectralis_raw_path': spectralis_raw_path,
        'jpg_encoded': jpg_encode
    }

    for key in ['derived_circle_scan', 'projection_image', 'en_face_slab_image', 'rnfl_thickness_values']:
        if jpg_encode:
            json_dict[key] = image_to_base64(json_dict[key])
        else:
            json_dict[key] = b64str_encode(json_dict[key].astype(np.float16))

    if debug:
        print('Saving json...')
    json_out_path = Path('data_wd/').joinpath('jsons', f'{scan_outname}.json')
    json_out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(json_out_path), 'w') as json_handle:
        json.dump(json_dict, json_handle)
    # return json_dict

def main():
    '''
    Performs post-processing on all scans indicated by their ILM mask.
    For each scan, all ILM and RNFL masks should be available in `data_wd/ILM_masks/` or 
    `data_wd/RNFL_masks/` respectively.
    
    Writes outputs to `data_wd/layer_maps/`.
    '''
    parser = argparse.ArgumentParser(description='Cirrus Segmentation post-processing')
    parser.add_argument('--proj_dir', default=None)
    parser.add_argument('--mask_dir_list', default=None)
    args = parser.parse_args()

    if args.proj_dir is not None:
        ILM_mask_dirs = pd.Series([p for p in Path(args.proj_dir).joinpath('data_wd', 'ILM_masks').glob('PSD*/*')])
    elif args.mask_dir_list is None:
        ILM_mask_dirs = pd.Series([p for p in Path('data_wd/ILM_masks/').glob('PSD*/*')])
    else:
        ILM_mask_dirs = pd.read_csv(args.mask_dir_list, names=['dirs']).dirs
    
    outdir=str(Path(args.proj_dir).joinpath('data_wd', 'layer_maps'))
    p_map(partial(process_mask_dir, outdir=outdir), ILM_mask_dirs)
    
    
if __name__=='__main__':
    main()