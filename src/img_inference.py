import pandas as pd
import numpy as np
import cv2
import skimage
import scipy
from pathlib import Path
import pywt

import utils
import post_process
from post_process import VERT_SCALE, MICRONS_PER_PIXEL

from keras_segmentation.models.unet import vgg_unet
from keras_segmentation.data_utils.data_loader import get_image_array, class_colors
from keras_segmentation.models.config import IMAGE_ORDERING
from keras_segmentation.predict import visualize_segmentation

from scipy.spatial import cKDTree
from skimage.restoration import denoise_wavelet
from statsmodels.nonparametric.smoothers_lowess import lowess

CLAHE = cv2.createCLAHE(clipLimit=2)

ONH_CENTER_MAP = (
    pd.read_csv(
        '20220503_ADAGES-DIGS_CirrusOCT_OpticDisc_ALL.csv',
        usecols=['PATIENT_ID', 'SITE', 'DATE_TIME',
                 'OPTICDISC_ONHCENTER_X', 'OPTICDISC_ONHCENTER_Y'],
        parse_dates=['DATE_TIME'])
    .set_index(['PATIENT_ID', 'SITE', 'DATE_TIME'])
)
ONH_CENTER_MAP.OPTICDISC_ONHCENTER_X = ONH_CENTER_MAP.OPTICDISC_ONHCENTER_X.apply(lambda x: x * 200 if x < 1 else x)
ONH_CENTER_MAP.OPTICDISC_ONHCENTER_Y = ONH_CENTER_MAP.OPTICDISC_ONHCENTER_Y.apply(lambda x: x * 200 if x < 1 else x)
ONH_CENTER_MAP.OPTICDISC_ONHCENTER_X -=  2 * (ONH_CENTER_MAP.OPTICDISC_ONHCENTER_X - 99)
ONH_CENTER_MAP = ONH_CENTER_MAP.apply(tuple, axis=1)

SAMPLE_OCT_NOISE = np.load('sample_cirrus_noise.npy')

def fill_nan_with_nearest_neighbor(arr):
    # Convert the input array to a NumPy array if it's not already
    arr = np.array(arr)
    
    # Create a mask of NaN values in the input array
    nan_mask = np.isnan(arr)
    
    # Find the indices of NaN values
    nan_indices = np.argwhere(nan_mask)
    
    # Find the indices of non-NaN values
    non_nan_indices = np.argwhere(~nan_mask)
    
    # Create a KD-tree from the non-NaN values for efficient nearest neighbor search
    tree = cKDTree(non_nan_indices)
    
    # Fill NaN values with their nearest neighbors
    for nan_index in nan_indices:
        _, nearest_index = tree.query(nan_index)
        arr[nan_index[0], nan_index[1]] = arr[non_nan_indices[nearest_index][0], non_nan_indices[nearest_index][1]]
    
    return arr

def register_bscans(img_data):
    offsets = [np.array([0.,0.])]
    for i in range(199):
        bscan = img_data[i]
        next_bscan = img_data[i+1]
        shift, error, diffphase = skimage.registration.phase_cross_correlation(bscan, next_bscan, normalization=None)
        offsets.append(shift)

    offsets = np.stack(offsets).cumsum(axis=0)
    
    return offsets

def align_bscans(img_data):
    median, std = np.median(img_data), np.std(img_data)
    sample_noise = np.sort(img_data[:,-5:,:].flatten())[:180000]
    offsets = register_bscans(img_data)
    # offsets -= offsets[99]
    offsets -= np.array([offsets.max(axis=0)[0], 0])
    offsets = offsets.astype(int)
    offsets[:,1] = 0 # dont apply offest in this axis

    adjusted_image = []
    for bscan, offset in zip(img_data, offsets):
        bscan = scipy.ndimage.shift(bscan, offset)#(offset[1], offset[0]))
        if offset[0] < 0:
            # bscan[offset[0]:] = 0 #np.random.normal(median, std, size=bscan[offset[0]:].shape)
            # bscan[offset[0]:] = np.random.choice(SAMPLE_OCT_NOISE, size=bscan[offset[0]:].shape) # fill voids with noise
            bscan[offset[0]:] = np.random.choice(sample_noise, size=bscan[offset[0]:].shape) # fill voids with noise
        adjusted_image.append(bscan)

    return np.stack(adjusted_image)

def _prep_img(img_data, ref_model, use_CLAHE):
    '''
    Prepares img data for prediction.
    '''
    def prep_bscan(bscan):
        # bscan = cv2.resize(img_data[0], (768, 496))
        if use_CLAHE:
            bscan = CLAHE.apply(bscan)
        bscan = cv2.cvtColor(bscan, cv2.COLOR_GRAY2RGB)
        bscan = get_image_array(bscan, ref_model.input_width, ref_model.input_height, ordering=IMAGE_ORDERING)
        return bscan
    
    img_data_flipped = np.flip(img_data, axis=1) # right side up
    # bisect bscans into left and right
    bisect_idx = img_data_flipped.shape[2]//2
    bscans_left = img_data_flipped[:,:,:bisect_idx]
    bscans_right = img_data_flipped[:,:,bisect_idx:]
    bscans_right = np.flip(bscans_right, axis=2)

    x = np.array([prep_bscan(inp) for inp in img_data_flipped])
    x_left = np.array([prep_bscan(inp) for inp in bscans_left])
    x_right = np.array([prep_bscan(inp) for inp in bscans_right])

    return x, x_left, x_right

def _segmentation_prediction(data, model, verbose=0):
    pred = model.predict(data, verbose=verbose)
    pred = pred.argmax(axis=2).reshape(200, model.output_height, model.output_width)
    return pred

def process_segmentation(segmentation):
    length = segmentation.shape[0]
    segmentation = skimage.transform.resize(segmentation, (length, 496, 768), preserve_range=True).round()
    segmentation = np.array([post_process.morph_operations(bscan) for bscan in segmentation])
    return segmentation

def _segmentation_to_surface(segmentation):
    surface = np.flip(segmentation.argmax(axis=1).astype(np.float32))
    surface[surface==0] = np.nan
    surface = skimage.transform.resize(surface * VERT_SCALE, (200, 200), preserve_range=True)
    nan_mask = np.isnan(surface)
    nan_indices = np.argwhere(nan_mask)
    surface = fill_nan_with_nearest_neighbor(surface)
    # apply lowess smooting
    surface = np.stack([
        lowess(line, list(range(line.size)), 0.05, is_sorted=True, missing='drop', return_sorted=False)
        for line in surface
    ])
    surface = np.stack([
        lowess(line, list(range(line.size)), 0.05, is_sorted=True, missing='drop', return_sorted=False)
        for line in surface.T
    ]).T
    for nan_index in nan_indices: # replace nans
        surface[nan_index[0], nan_index[1]] = np.nan
    # surface = scipy.ndimage.median_filter(surface, size=7) # median filter
    return surface

def _split_segmentation_prediction(x_left, x_right, ilm_model, rnfl_model, verbose=0):
    # predict rnfl segmentation
    rnfl_out_left = _segmentation_prediction(x_left, rnfl_model, verbose=verbose)
    rnfl_out_right = _segmentation_prediction(x_right, rnfl_model, verbose=verbose)
    
    # predict ILM segmentation
    ilm_out_left = _segmentation_prediction(x_left, ilm_model, verbose=verbose)
    ilm_out_right = _segmentation_prediction(x_right, ilm_model, verbose=verbose)

    # join left and right segmentation masks
    ilm_out = np.concatenate([ilm_out_left, np.flip(ilm_out_right, axis=2)], axis=2)
    rnfl_out = np.concatenate([rnfl_out_left, np.flip(rnfl_out_right, axis=2)], axis=2)
    return ilm_out, rnfl_out


def segmentation_prep(img_path, align=True):
    img_data = utils.load_numpy_onh_cube_img(img_path)
    if align:
        img_data = align_bscans(img_data)
    raw_img_data = img_data.copy()
    img_data = denoise_wavelet(
        img_data,
        sigma=np.std(img_data),
        wavelet='bior4.4'
    )
    img_data = scipy.ndimage.median_filter(img_data, size=(5,1,1))
    img_data = (img_data * 255).astype(np.uint8)

    data_dict = dict(
        img_data=img_data,
        raw_img_data=raw_img_data,
        img_path=img_path
    )
    return data_dict

def inference_step_only(data_dict, ilm_model, rnfl_model, split_bscans=True, use_CLAHE=False):
    verbose = 0

    img_data = data_dict['img_data']
    raw_img_data = data_dict['raw_img_data']
    img_path = data_dict['img_path']
    
    pt_id, scan_type, scan_date, scan_time, scan_eye, _, _, _ = Path(img_path).name.split('_')
    scan_outname = '_'.join([pt_id+scan_eye, scan_date, scan_time])

    x, x_left, x_right = _prep_img(img_data, rnfl_model, use_CLAHE)

    if split_bscans:
        ilm_out, rnfl_out = _split_segmentation_prediction(x_left, x_right, ilm_model, rnfl_model, verbose=verbose)
    else:
        ilm_out = _segmentation_prediction(x, ilm_model, verbose=verbose)
        rnfl_out = _segmentation_prediction(x, rnfl_model, verbose=verbose)

    json_dict = {
        'img_path': str(img_path),
        'pt_id': pt_id,
        'scan_eye': scan_eye,
        'scan_type': scan_type,
        'scan_date': scan_date,
        'scan_time': scan_time,
        'scan_outname': scan_outname,
        'ILM_seg': ilm_out,
        'RNFL_seg': rnfl_out,
    }

    json_dict['cube_data'] = raw_img_data
    
    return pd.Series(json_dict)

def postprocess_segmentation_step_only(json_collection):
    if not isinstance(json_collection, (pd.Series, dict)):
        json_collection = load_json_collection(json_collection)

    pt_id, scan_eye, scan_time, scan_date = json_collection.pt_id, json_collection.scan_eye, json_collection.scan_time, json_collection.scan_date
    proj_image = json_collection.cube_data.mean(axis=1)

    ilm_out = json_collection.ILM_seg
    rnfl_out = json_collection.RNFL_seg

    ilm_out = process_segmentation(ilm_out)
    rnfl_out = process_segmentation(rnfl_out)

    # compute/process surface
    ilm_surface = _segmentation_to_surface(ilm_out)
    rnfl_surface = _segmentation_to_surface(rnfl_out)

    # flip to correct orientation
    ilm_surface = np.flip(ilm_surface)#, axis=1)
    rnfl_surface = np.flip(rnfl_surface)#, axis=1)

    # compute rnfl thickness
    rnfl_thickness = rnfl_surface - ilm_surface
    rnfl_thickness *= MICRONS_PER_PIXEL # convert to microns
    rnfl_thickness[rnfl_thickness<=0] = np.nan

    slab_image = post_process.get_slab_image(json_collection.cube_data, ilm_surface, slab_width_microns=52)
    scan_center_idx = (pt_id[1:], scan_eye, pd.to_datetime(scan_date+'T'+scan_time.replace('-', ':')))
    scan_center = np.array(ONH_CENTER_MAP.get(scan_center_idx, (99,99)))
    
    der_circle_scan, der_ilm_surface, der_rnfl_surface = post_process.make_derived_circle_scan(
        json_collection.cube_data, ilm_surface, rnfl_surface,
        scan_center, scan_eye
    )

    derived_circle_segmentation = pd.concat(
        {'ILM_y': der_ilm_surface, 'RNFL_y': der_rnfl_surface}
    , axis=1)

    json_collection['scan_center'] = list(scan_center)
    json_collection['derived_circle_scan'] = der_circle_scan
    json_collection['derived_circle_segmentation'] = derived_circle_segmentation
    json_collection['projection_image'] = proj_image
    json_collection['en_face_slab_image'] = slab_image
    json_collection['rnfl_thickness_values'] = rnfl_thickness
    json_collection['ILM_y'] = ilm_surface
    json_collection['RNFL_y'] = rnfl_surface
    
    return json_collection
    

def process_from_img(img_path, ilm_model, rnfl_model, align=False, split_bscans=True, use_CLAHE=True):
    verbose = 0
    pt_id, scan_type, scan_date, scan_time, scan_eye, _, _, _ = Path(img_path).name.split('_')
    scan_outname = '_'.join([pt_id+scan_eye, scan_date, scan_time])
    
    img_data = utils.load_numpy_onh_cube_img(img_path)
    proj_image = img_data.mean(axis=1)
    if align:
        img_data = align_bscans(img_data)
    raw_img_data = img_data.copy()
    img_data = denoise_wavelet(
        img_data,
        sigma=np.std(img_data),
        wavelet='bior4.4'
    )
    img_data = scipy.ndimage.median_filter(img_data, size=(5,1,1))
    img_data = (img_data * 255).astype(np.uint8)
    # img_data = scipy.ndimage.gaussian_filter(img_data, sigma=(1,0,0))
    # img_data = CLAHE.apply(img_data.reshape((200,204800))).reshape((200,1024,200))

    x, x_left, x_right = _prep_img(img_data, rnfl_model, use_CLAHE)

    if split_bscans:
        ilm_out, rnfl_out = _split_segmentation_prediction(x_left, x_right, ilm_model, rnfl_model, verbose=verbose)
    else:
        ilm_out = _segmentation_prediction(x, ilm_model, verbose=verbose)
        rnfl_out = _segmentation_prediction(x, rnfl_model, verbose=verbose)

    ilm_out = process_segmentation(ilm_out)
    rnfl_out = process_segmentation(rnfl_out)

    # print(ilm_out.shape, rnfl_out.shape)

    # compute/process surface
    ilm_surface = _segmentation_to_surface(ilm_out)
    rnfl_surface = _segmentation_to_surface(rnfl_out)

    # flip to correct orientation
    ilm_surface = np.flip(ilm_surface)#, axis=1)
    rnfl_surface = np.flip(rnfl_surface)#, axis=1)

    # compute rnfl thickness
    rnfl_thickness = rnfl_surface - ilm_surface
    rnfl_thickness *= MICRONS_PER_PIXEL # convert to microns
    rnfl_thickness[rnfl_thickness<=0] = np.nan

    
    slab_image = post_process.get_slab_image(raw_img_data, ilm_surface, slab_width_microns=52)
    # scan_center = post_process.find_onh_center(rnfl_thickness)
    scan_center = np.array(ONH_CENTER_MAP.get(
        (pt_id[1:], scan_eye, pd.to_datetime(scan_date+'T'+scan_time.replace('-', ':')))
    , (99,99)))
    
    der_circle_scan, der_ilm_surface, der_rnfl_surface = post_process.make_derived_circle_scan(
        raw_img_data, ilm_surface, rnfl_surface,
        scan_center, scan_eye
    )

    derived_circle_segmentation = pd.concat(
        {'ILM_y': der_ilm_surface, 'RNFL_y': der_rnfl_surface}
    , axis=1)

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
        'derived_circle_segmentation': derived_circle_segmentation,
        'projection_image': proj_image,
        'en_face_slab_image': slab_image,
        'rnfl_thickness_values': rnfl_thickness,
        'ILM_y': ilm_surface,
        'RNFL_y': rnfl_surface,
    }

    json_dict['cube_data'] = raw_img_data
    
    return pd.Series(json_dict)
