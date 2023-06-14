import pandas as pd
import numpy as np
import cv2
import skimage
import scipy
from pathlib import Path

import utils
import post_process
from post_process import VERT_SCALE, MICRONS_PER_PIXEL

from keras_segmentation.models.unet import vgg_unet
from keras_segmentation.data_utils.data_loader import get_image_array, class_colors
from keras_segmentation.models.config import IMAGE_ORDERING
from keras_segmentation.predict import visualize_segmentation

from statsmodels.nonparametric.smoothers_lowess import lowess

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
    offsets = register_bscans(img_data)
    # offsets -= offsets[99]
    offsets -= np.array([offsets.max(axis=0)[0], 0])
    # offsets[:,1] = 0

    adjusted_image = []
    for bscan, offset in zip(img_data, offsets):
        bscan = scipy.ndimage.shift(bscan, offset)#(offset[1], offset[0]))
        adjusted_image.append(bscan)

    return np.stack(adjusted_image)

def _prep_img(img_data, ref_model):
    '''
    Prepares img data for prediction.
    '''
    def prep_bscan(bscan):
        # bscan = cv2.resize(img_data[0], (768, 496))
        bscan = cv2.cvtColor(bscan, cv2.COLOR_GRAY2RGB)
        bscan = get_image_array(bscan, ref_model.input_width, ref_model.input_height, ordering=IMAGE_ORDERING)
        return bscan
    
    img_data_flipped = np.flip(img_data) # right side up
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
    # apply lowess
    surface = np.stack([
        lowess(line, list(range(line.size)), 0.05, is_sorted=True, missing='drop', return_sorted=False)
        for line in surface
    ])
    surface = np.stack([
        lowess(line, list(range(line.size)), 0.05, is_sorted=True, missing='drop', return_sorted=False)
        for line in surface.T
    ]).T
    return surface

def process_from_img(img_path, ilm_model, rnfl_model, align=False):
    verbose = 0
    pt_id, scan_type, scan_date, scan_time, scan_eye, _, _, _ = Path(img_path).name.split('_')
    scan_outname = '_'.join([pt_id+scan_eye, scan_date, scan_time])
    
    img_data = utils.load_numpy_onh_cube_img(img_path)
    if align:
        img_data = align_bscans(img_data)

    x, x_left, x_right = _prep_img(img_data, rnfl_model)

    # predict rnfl segmentation
    rnfl_out_left = _segmentation_prediction(x_left, rnfl_model, verbose=verbose)
    rnfl_out_right = _segmentation_prediction(x_right, rnfl_model, verbose=verbose)
    
    # predict ILM segmentation
    ilm_out_left = _segmentation_prediction(x_left, ilm_model, verbose=verbose)
    ilm_out_right = _segmentation_prediction(x_right, ilm_model, verbose=verbose)

    # join left and right segmentation masks
    ilm_out = np.concatenate([ilm_out_left, np.flip(ilm_out_right, axis=2)], axis=2)
    rnfl_out = np.concatenate([rnfl_out_left, np.flip(rnfl_out_right, axis=2)], axis=2)

    ilm_out = process_segmentation(ilm_out)
    rnfl_out = process_segmentation(rnfl_out)

    # compute/process surface
    ilm_surface = _segmentation_to_surface(ilm_out)
    rnfl_surface = _segmentation_to_surface(rnfl_out)

    # compute rnfl thickness
    rnfl_thickness = rnfl_surface - ilm_surface
    rnfl_thickness *= MICRONS_PER_PIXEL # convert to microns
    rnfl_thickness[rnfl_thickness<=0] = np.nan

    proj_image = img_data.mean(axis=1)
    slab_image = post_process.get_slab_image(img_data, ilm_surface)
    scan_center = post_process.find_onh_center(rnfl_thickness)
    
    der_circle_scan, der_ilm_surface, der_rnfl_surface = post_process.make_derived_circle_scan(
        img_data, ilm_surface, rnfl_surface,
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

    json_dict['cube_data'] = img_data
    
    return pd.Series(json_dict)
