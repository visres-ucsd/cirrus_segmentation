import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from p_tqdm import p_umap
import glob
import argparse

def split_LR(cross_section_data):
    left = np.fliplr(np.flip(cross_section_data[:,:100]))
    right = np.flip(cross_section_data[:,100:])
    left, right = (Image.fromarray(img).resize((384, 496)) for img in (left, right))
    return left, right

def img_to_pngs(img_path, jpg_folder):
    '''
    Converts Cirrus 200x200 Optic Disc scan into jpgs of each b-scan. Outputs two images: Left and right.
    '''
    pt_id, _, date, time, eye, unknown_id, cube, _ = Path(img_path).name.split('_')
    session_folder = "_".join([pt_id+eye, date, time])
    jpg_folder.joinpath(pt_id, session_folder).mkdir(parents=True, exist_ok=True)
    img_data = np.fromfile(img_path, dtype=np.uint8).reshape(200, 1024, 200)
    
    for cross_section_idx, cross_section_data in enumerate(img_data):
        left, right = split_LR(cross_section_data)
        left_fp = "_".join([unknown_id] + [str(cross_section_idx+1), 'L']) + '.jpg'
        right_fp = "_".join([unknown_id] + [str(cross_section_idx+1), 'R']) + '.jpg'
        left.save(jpg_folder.joinpath(pt_id, session_folder, left_fp))
        right.save(jpg_folder.joinpath(pt_id, session_folder, right_fp))
        
def main():
    '''
    Reads cirrus 200x200 ONH .img files and outputs bscans into pngs.

    img_dir: Directory with .img files to process.
    out_dir: Directory to place output files.
    imgs_list: File listing the paths to imgs to process.
      (Optional, will be prioritized over img_dir if specified)
    '''
    parser = argparse.ArgumentParser(description='Cirrus img to png converter')
    parser.add_argument('--img_dir')
    parser.add_argument('--out_dir', default='jpgs')
    parser.add_argument('--imgs_list')
    
    args = parser.parse_args()
    
    jpg_folder = Path(args.out_dir)
    
    if args.imgs_list:
        imgs = pd.read_csv(args.imgs_list, header=None)[0]
    else:
        imgs = glob.glob(f'{args.img_dir}/*.img')
    
    # do parallel
    p_umap(img_to_pngs, imgs, [jpg_folder]*len(imgs))
    
if __name__ == '__main__':
    main()