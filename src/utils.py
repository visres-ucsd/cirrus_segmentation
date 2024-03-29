import numpy as np
import pandas as pd
import base64
import json
import cv2
import io
from pathlib import Path

def compress_array(array):
    file_obj = io.BytesIO()
    np.savez_compressed(file_obj, array=array)
    compressed_bytes = file_obj.getvalue()
    compressed_string = compressed_bytes.decode('ISO-8859-1')
    return compressed_string

def decompress_array(compressed_string):    
    compressed_bytes = compressed_string.encode('ISO-8859-1')
    file_obj = io.BytesIO(compressed_bytes)
    loaded = np.load(file_obj)
    array = loaded['array']
    return array

def save_json_dict_savez(json_dict, outfp):
    def _jsonify_obj(obj):
        if isinstance(obj, np.ndarray):
            return compress_array(obj)
        elif isinstance(obj, pd.DataFrame):
            return base64_enc_df(obj)
        else:
            return obj
        
    json_dict.apply(_jsonify_obj).to_json(outfp)

def save_json_dict(json_dict, outfp):
    def _jsonify_obj(obj):
        if isinstance(obj, np.ndarray):
            return b64encode_numpy(obj)
        elif isinstance(obj, pd.DataFrame):
            return base64_enc_df(obj)
        else:
            return obj
        
    json_dict.apply(_jsonify_obj).to_json(outfp)

def load_numpy_onh_cube_img(img_path):
    return np.fromfile(img_path, dtype=np.uint8).reshape(200, 1024, 200)

# Base64 Encoding Utils
def b64str_encode(ndarr):
    return base64.b64encode(ndarr.tobytes()).decode('utf-8')

def b64_decode(b64str, dtype=np.int8, shape=None):
    decoded = np.frombuffer(base64.b64decode(b64str), dtype=dtype)
    if not shape is None:
        decoded = decoded.reshape(*shape)
    return decoded

def b64encode_numpy(ndarr):
    '''Converts numpy array into JSON dictionary with the numerical data encoded in a base64 string.'''
    json_numpy = {
        'shape': ndarr.shape,
        'dtype': str(ndarr.dtype),
        'data': b64str_encode(ndarr),
    }
    return json_numpy

def b64decode_numpy(np_json):
    '''Loads a numpy array from JSON dictionary with the numerical data encoded in a base64 string.'''
    decoded = np.frombuffer(
        base64.b64decode(np_json['data']),
        dtype=np_json['dtype']
    ).reshape(*np_json['shape'])
    return decoded

def base64_enc_df(df):
    '''JSONifies a Pandas DataFrame with numerical data encoded in base64.'''
    json_df = {
        'data': {col: b64encode_numpy(coldata.values) for col, coldata in df.iteritems()}
    }
    json_df['index'] = df.index.to_list()
    return json_df

def base64_dec_df(df_json_dict, index_dtype=None):
    '''Reconstructs a Pandas DataFrame from JSON with 'data encoded in base64.'''
    df_json_dict = df_json_dict.copy()
    df_json_dict['data'] = {col: b64decode_numpy(coldata) for col, coldata in df_json_dict['data'].items()}
    return pd.DataFrame(**df_json_dict)

def image_to_base64(img_arr):
    '''
    Converts image to a base64 string with JPG encoding.
    '''
    img_arr = img_arr.copy()
    img_arr[np.isnan(img_arr)] = img_arr.max()
    
    temp_im_file = f'temp/img_to_str.png'
    Path(temp_im_file).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(temp_im_file, img_arr)
    with open(temp_im_file, 'rb') as image_file:
        data = image_file.read()
    data = b64str_encode(data)
    Path(temp_im_file).unlink()
    return data

def read_img_base64(data_str, dtype=np.uint8):
    '''
    Loads an png image saved as a base64 string.
    '''
    data = b64_decode(data_str, dtype=dtype)
    img = cv2.imdecode(data, flags=0)
    return img

def load_partial_json_collection(json_path, decode_cube=False):
    '''Loads a partial (unfinished) JSON collection of OCT segmentation outputs.

    Use decode_cube=True if decoding raw cube data is necessary.
    '''
    with open(json_path) as json_handle:
        json_collection = json.load(json_handle)

    for key in ['ILM_seg', 'RNFL_seg']:
        json_collection[key] = b64decode_numpy(json_collection[key])

    json_collection['cube_data'] = b64decode_numpy(json_collection['cube_data'])

    json_collection = pd.Series(json_collection)
    return json_collection

def load_json_collection(json_path):
    '''Loads a JSON collection of OCT segmentation outputs.'''
    with open(json_path) as json_handle:
        json_collection = json.load(json_handle)
    
    for key in ['rnfl_thickness_values', 'ILM_y', 'RNFL_y', 'ILM_seg', 'RNFL_seg']:
        if key in json_collection:
            json_collection[key] = b64decode_numpy(json_collection[key])

    if not 'jpg_encoded' in json_collection: # assume jpg encoding if missing
        json_collection['jpg_encoded'] = False

    if json_collection['jpg_encoded']:
        for im_key in ['derived_circle_scan', 'projection_image', 'en_face_slab_image']:
             # read base64 jpg data
            json_collection[im_key] = read_img_base64(json_collection[im_key])
    else:
        for im_key in ['projection_image', 'en_face_slab_image', 'derived_circle_scan']:
             # read base64 numpy data
            json_collection[im_key] = b64decode_numpy(json_collection[im_key])
    

    # load derived surfaces df
    json_collection['derived_circle_segmentation'] = base64_dec_df(json_collection['derived_circle_segmentation'])

    if 'cube_data' in json_collection:
        json_collection['cube_data'] = b64decode_numpy(json_collection['cube_data'])
    
    json_collection = pd.Series(json_collection)
    return json_collection