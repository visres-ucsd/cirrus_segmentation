from keras_segmentation.models.unet import vgg_unet
from keras_segmentation.data_utils.data_loader import get_image_array, class_colors
from keras_segmentation.models.config import IMAGE_ORDERING
from keras_segmentation.predict import visualize_segmentation
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os
import argparse

MODEL_TYPE = 'ILM'

def seg_scan_dir(scan_dir):
    scan = scan_dir.name
    pt_id = scan_dir.parent.name
    group = scan_dir.parent.parent.name
    out_path = os.path.join(f"{MODEL_TYPE}_masks", group, pt_id, scan)
    
    predict_multiple(
        model=model, 
        inp_dir=str(scan_dir), 
        out_dir=out_path
    )

# for scan_dir in tqdm(scan_dirs):
#     seg_scan_dir(scan_dir)

# def seg_jpg(jpg_path):
#     scan = jpg_path.parent.name
#     pt_id = jpg_path.parent.parent.name
#     group = jpg_path.parent.parent.parent.name
#     out_path = os.path.join("RNFL_masks", group, pt_id, scan, jpg_path.name)
    
#     # dout = model.predict_segmentation(inp=str(jpg_path), out_fname=out_path)
    
#     predict( 
#         model=model, 
#         inp=str(jpg_path), 
#         out_fname=out_path
#     )

# for file in tqdm(files):
#     seg_jpg(file)

def gen_out_name(jpg_path):
    '''Generates the output path for an image.'''
    scan = jpg_path.parent.name
    pt_id = jpg_path.parent.parent.name
    group = jpg_path.parent.parent.parent.name
    out_path = os.path.join(f"{MODEL_TYPE}_masks", group, pt_id, scan, jpg_path.name)
    
    return out_path

def dir_batch_predict(model, inp_dir=None):
    jpgs = [i for i in inp_dir.glob('*.jpg')]
    batch_predict(model, jpgs)
    

def batch_predict(model, jpgs, checkpoints_path=None, overlay_img=False,
                  class_names=None, show_legends=False, colors=class_colors,
                  prediction_width=None, prediction_height=None,
                  read_image_type=1):
    '''
    Predicts segmentation of a batch of images. Writes output to file.

    @param model: keras segmnetation model
    @param jpgs: list of images to predict
    @returns: None
    '''
    output_width = model.output_width
    output_height = model.output_height
    input_width = model.input_width
    input_height = model.input_height
    n_classes = model.n_classes
    
    out_paths = [gen_out_name(jpg_path) for jpg_path in jpgs]
    if Path(out_paths[0]).exists():
        return
        
    inp_example = cv2.imread(str(jpgs[0]))
    x = [get_image_array(str(inp), input_width, input_height, ordering=IMAGE_ORDERING) for inp in jpgs]
    
    pr = model.predict(np.array(x))
    pr = pr.reshape((len(jpgs), output_height,  output_width, n_classes)).argmax(axis=3)
    
    for arr, out_fname in zip(pr, out_paths):
        # seg_img = visualize_segmentation(arr, x[0], n_classes=n_classes, # previously incorrect
        seg_img = visualize_segmentation(arr, inp_example, n_classes=n_classes,
                                     colors=colors, overlay_img=overlay_img,
                                     show_legends=show_legends,
                                     class_names=class_names,
                                     prediction_width=prediction_width,
                                     prediction_height=prediction_height)
        
        # make sure out dir exits
        Path(out_fname).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(out_fname, seg_img)
    
        
def main():
    global MODEL_TYPE
    parser = argparse.ArgumentParser()
    parser.add_argument('png_dir')
    parser.add_argument('model_type')
    args = parser.parse_args()

    path = args.png_dir
    MODEL_TYPE = args.model_type

    model = vgg_unet(n_classes=2,  input_height=416, input_width=608  )
    model.load_weights(f"{MODEL_TYPE}25/{MODEL_TYPE}.00025")
    
    files = [i for i in Path(path).rglob('*.jpg')]
    scan_dirs = [i for i in Path('CirrusPNGs').glob('*/*/*')]
    
    for scan_dir in tqdm(scan_dirs):
        dir_batch_predict(model=model, inp_dir=scan_dir)
        
if __name__=='__main__':
    main()
    