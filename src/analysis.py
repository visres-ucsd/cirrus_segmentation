import pandas as pd
import scipy
import cv2
import os
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from pathlib import Path
from src.post_process import EXPECTED_SHAPE, get_stitched_image, compute_top_layer, morph_operations

def plot_after_2d_med_filter(path, layer, axes=None, savefig=True, savefig_path=None):
    def get_ilm_rnfl_surfaces(path):
        if 'ILM_location' in path:
            ilm_path = path
            rnfl_path = path.replace('ILM', 'RNFL')
        elif 'RNFL_location':
            ilm_path = path.replace('RNFL', 'ILM')
            rnfl_path = path
        return ilm_path, rnfl_path
    
    ilm_path, rnfl_path = get_ilm_rnfl_surfaces(path)
    # print(ilm_path, rnfl_path, sep='\n')
    ilm_layer = pd.read_csv(ilm_path, index_col=0)
    ilm_layer = scipy.signal.medfilt2d(ilm_layer)
    ilm_layer = pd.DataFrame(ilm_layer, index=range(1,201)).loc[layer]
    # rnfl_layer = pd.read_csv(rnfl_layer, index_col=0).loc[layer]
    rnfl_layer = pd.read_csv(rnfl_path, index_col=0)
    rnfl_layer = scipy.signal.medfilt2d(rnfl_layer)
    rnfl_layer = pd.DataFrame(rnfl_layer, index=range(1,201)).loc[layer]
    
    # find bscan filepath
    eye_id, date, time, modelout_type, _ = Path(path).name.split('_')
    pt_id, eye = eye_id[:-2], eye_id[-2:]
    bscan_left = str([i for i in
        Path(f'data_wd/sample_scans/{pt_id}').
        joinpath('_'.join([eye_id, date, time]))
        .glob(f'*_{layer}_L.jpg')
    ][0])

    bscan_image = get_stitched_image(bscan_left)

    if axes is None:
        fig, (ax) = plt.subplots(1, 1, figsize=(15, 8))
    ax.imshow(bscan_image)
    ax.scatter(ilm_layer.index, ilm_layer, label='ILM', color='c', s=1, alpha=0.7)
    ax.scatter(rnfl_layer.index, rnfl_layer, label='RNFL', color='r', s=1, alpha=0.7)
    ax.legend()
    if savefig:
        if savefig_path is None:
            savefig_path = f'plots/{Path(bscan_left).name}'
        print(f'Saving figure to {savefig_path}')
        fig.savefig(savefig_path, bbox_inches='tight', facecolor='w', dpi=600)
    return fig

def animate_layers(path):
    scan_id = pd.Series(path).str.extract('(PSD\d+O[DS][-_\d+]+)', expand=False)[0].strip('_')
    for layer in range(5, 200, 5):
        fig = plot_after_2d_med_filter(path, layer, savefig=True, savefig_path=f'temp/anim_layer_{layer:03d}.png')
        plt.close(fig)
    
    outfn = os.path.join('anim', scan_id+".mp4")
    ffmpeg_cmd = f'ffmpeg -framerate 5 -pattern_type glob -i "temp/anim_layer_*.png" -c:v libx264 ' \
        f'-pix_fmt yuv420p -vf pad="width=ceil(iw/2)*2:height=ceil(ih/2)*2" -y {outfn}'
    print(ffmpeg_cmd)
    os.system(ffmpeg_cmd)

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
    
    # select valid layer
    diff = rnfl_layer - ilm_layer
    diff = diff[diff > 0].dropna()
    rnfl_layer = rnfl_layer[diff.index]
    ilm_layer = ilm_layer[diff.index]
    
    if axes is None:
        fig, axes = plt.subplots(1, 1, figsize=(15, 10))
    axes[0].imshow(img)
    axes[0].scatter(ilm_layer.index, ilm_layer, label='ILM', color='c', s=1)
    axes[0].scatter(rnfl_layer.index, rnfl_layer, label='RNFL', color='r', s=1)
    axes[0].set_title('Raw')
    axes[0].set_ylabel(name)
    legend = axes[0].legend()

    axes[1].imshow(ilm_img)
    axes[1].set_title('ILM Mask')
    axes[2].imshow(rnfl_img)
    axes[2].set_title('RNFL Mask')
    if axes is None:
        fig.savefig('figure.png', bbox_inches='tight', facecolor='w')
    else:
        for handle in legend.legendHandles:
            print('setting size')
            handle._legmarker.set_markersize(6)

def plot_LR_stitched(img_path, rnfl_path, ilm_path, axes=None, savefig=True, morph=False):
    # ilm_path = rnfl_path.replace('RNFL_masks', 'ILM_masks')
    # # img_path = rnfl_path.replace('RNFL_masks', 'CirrusPNGs')
    # img_path = rnfl_path.replace('RNFL_masks', 'sample_scans')
    # # name = Path(img_path).parent.name + '_' + Path(img_path).name
    name = Path(img_path).name
    plot_name = name.replace('_R.jpg', '').replace('_L.jpg', '')
    
    rnfl_img = get_stitched_image(rnfl_path)
    ilm_img = get_stitched_image(ilm_path)
    img = get_stitched_image(img_path)
    
    # rnfl_img, ilm_img = cv2.resize(rnfl_img, (200,200)), cv2.resize(ilm_img, (200,200))
    # img = cv2.resize(img, (200,200))
    rnfl_layer, ilm_layer = compute_top_layer(rnfl_img), compute_top_layer(ilm_img)
    # reference_layer = compute_reference_layer(img)
    rnfl_img, ilm_img = bool_mask(rnfl_img), bool_mask(ilm_img)

    if morph:
        rnfl_img = morph_operations(rnfl_img)
        ilm_img = morph_operations(ilm_img)
    
    if axes is None:
        gs = gridspec.GridSpec(2,4)
        fig = plt.figure(figsize=(20,10))
        raw_ax = fig.add_subplot(gs[:2, :2])
        ilm_ax = fig.add_subplot(gs[0, 2:3])
        rnfl_ax = fig.add_subplot(gs[1, 2:3])
        
    else:
        raw_ax, ilm_ax, rnfl_ax = axes
    raw_ax.imshow(img)
    raw_ax.scatter(ilm_layer.index, ilm_layer, label='ILM', color='c', s=1, linewidths=1)
    raw_ax.scatter(rnfl_layer.index, rnfl_layer, label='RNFL', color='r', s=1, linewidths=1)
    # raw_ax.scatter(reference_layer.index, reference_layer, label='Sobel ED', color='g', s=1, linewidths=1)
    raw_ax.set_title('Raw')
    if axes is not None:
        raw_ax.set_ylabel(plot_name)
    legend = raw_ax.legend()

    for handle in legend.legendHandles:
        handle.set_sizes([10.0])

    ilm_ax.imshow(ilm_img)
    ilm_ax.set_title('ILM Mask')
    rnfl_ax.imshow(rnfl_img)
    rnfl_ax.set_title('RNFL Mask')
    if axes is None:
        raw_ax.set_title(plot_name)
        fig.tight_layout(rect=[0, 0, 1, 0.75])
        fig.subplots_adjust(wspace=.1, hspace=0)
        if savefig:
            fig.savefig(f'plots/{plot_name}', bbox_inches='tight', facecolor='w', dpi=1200)

def plot_layer(pt_id, eye, date=None, time=None, scan_id=None, layer=None, morph=True):
    date_time_none = date is None
    scan_id_none = scan_id is None
    assert not all([date_time_none, scan_id_none]), 'Must specify either date-time or scan_id'

    # find rnfl_mask_image
    rnfl_fp = Path(f'RNFL_masks/zhi_cases/{pt_id}')

    if scan_id is not None:
        pass # TODO
    else:
        if not rnfl_fp.joinpath(f'{pt_id}{eye}_{date}_{time}').exists():
            rnfl_fp = Path(f'RNFL_masks/zhi_controls/{pt_id}')
        rnfl_fp = rnfl_fp.joinpath(f'{pt_id}{eye}_{date}_{time}')
        assert rnfl_fp.exists(), f'No RNFL mask directrory: {str(rnfl_fp)}'
        rnfl_fp = rnfl_fp.glob(f'*_{layer}_*.jpg')
        rnfl_fp = next(rnfl_fp)
    
    plot_LR_stitched(str(rnfl_fp), savefig=False, morph=morph)