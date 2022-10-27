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
import itertools

# OUT_DIR = None

# path = 'CirrusPNGs'

# names = []

# for jpg_path in tqdm(files):
    # scan = jpg_path.parent.name
    # pt_id = jpg_path.parent.parent.name
    # group = jpg_path.parent.parent.parent.name
    # out_path = os.path.join(group, pt_id, scan, jpg_path.name)
    
    # dout = model.predict_segmentation(inp=str(jpg_path), out_fname=out_path)

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

def chunked_iterable(iterable, size):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, size))
        if not chunk:
            break
        yield chunk

def gen_out_name(jpg_path, outdir):
    # scan = jpg_path.parent.name
    # pt_id = jpg_path.parent.parent.name
    # group = jpg_path.parent.parent.parent.name
    # out_path = os.path.join(f"{MODEL_TYPE}_masks", pt_id, scan, jpg_path.name)
    
    parts = list(jpg_path.parts)
    parts[0] = outdir
    out_path = os.path.join(*parts)
    
    return out_path

def dir_batch_predict(model, inp_dir=None):
    jpgs = [i for i in inp_dir.glob('*.jpg')]
    # batch1 = jpgs[:100]
    # batch2 = jpgs[100:]
    
    batch_predict(model, jpgs)
    # batch_predict(model, batch2)
    

def batch_predict(model, jpgs, outdir, checkpoints_path=None, overlay_img=False,
                  class_names=None, show_legends=False, colors=class_colors,
                  prediction_width=None, prediction_height=None,
                  read_image_type=1):
    output_width = model.output_width
    output_height = model.output_height
    input_width = model.input_width
    input_height = model.input_height
    n_classes = model.n_classes
    
    out_paths = [gen_out_name(jpg_path, outdir) for jpg_path in jpgs]
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
    global OUT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument('rnfl_model_weights')
    parser.add_argument('ilm_model_weights')
    parser.add_argument('png_dir')
    parser.add_argument('outdir')
    parser.add_argument('--nested', action='store_true')
    args = parser.parse_args()

    path = args.png_dir
    # if 'ILM' in args.model_weights.upper():
    #     OUT_DIR = os.path.join(args.outdir, 'ILM')
    # if 'RNFL' in args.model_weights.upper():
    #     OUT_DIR = os.path.join(args.outdir, 'RNFL')
    
    rnfl_model = vgg_unet(n_classes=2,  input_height=416, input_width=608)
    rnfl_model.load_weights(args.rnfl_model_weights)
    
    ilm_model = vgg_unet(n_classes=2,  input_height=416, input_width=608)
    ilm_model.load_weights(args.ilm_model_weights)
    
    ilm_outdir = os.path.join(args.outdir, 'ILM')
    rnfl_outdir = os.path.join(args.outdir, 'RNFL')
    
    if args.nested:
        scan_dirs = [i for i in Path(path).glob('*/*/*')]
        for scan_dir in tqdm(scan_dirs):
            jpgs = [i for i in scan_dir.glob('*.jpg')]
            if Path(gen_out_name(jpgs[0], rnfl_outdir)).exists(): # skip existing dirs
                continue
            else:
                dir_batch_predict(model=rnfl_model, inp_dir=scan_dir, outdir=rnfl_outdir)
                dir_batch_predict(model=ilm_model, inp_dir=scan_dir, outdir=ilm_outdir)
    else:
        files = [i for i in Path(path).rglob('*.jpg')]
        batch_size = 400
        for batch in tqdm(list(chunked_iterable(files, size=batch_size))):
            batch_predict(rnfl_model, batch, outdir=rnfl_outdir)
            batch_predict(ilm_model, batch, outdir=ilm_outdir)
    
        
if __name__=='__main__':
    main()
    