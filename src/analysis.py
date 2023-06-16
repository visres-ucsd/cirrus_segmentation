import pickle
import pandas as pd
import numpy as np
import scipy
import cv2
import os
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
from pathlib import Path
import json
from post_process import EXPECTED_SHAPE, VERT_SCALE, INNER_RADIUS, OUTER_RADIUS, \
    BASE_DIAM, get_stitched_image, compute_top_layer, \
    morph_operations, bool_mask, surface_smoothing, get_slab_image, find_onh_center
from utils import load_json_collection

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
        # print(f'Saving figure to {savefig_path}')
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
    # print(ffmpeg_cmd)
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
            # print('setting size')
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

def draw_derived_en_face_imgs(json_collection, savefig=False):
    if not isinstance(json_collection, (pd.Series, dict)):
        json_collection = load_json_collection(json_collection)

    # spectralis_raw_path = json_collection.spectralis_raw_path
    der_circle_scan = json_collection.derived_circle_scan
    der_ilm_surface = json_collection.derived_circle_segmentation.ILM_y
    der_rnfl_surface = json_collection.derived_circle_segmentation.RNFL_y

    # print('Initializing figure...')
    gs_rows = 2 #if spectralis_raw_path is None else 3
    gs = gridspec.GridSpec(gs_rows,3)
    fig = plt.figure(figsize=(10,10))# if spectralis_raw_path is None else 13.33))
    fig.subplots_adjust(wspace=0, hspace=0)
    circle_ax = fig.add_subplot(gs[0, :])
    proj_ax = fig.add_subplot(gs[1, 0])
    slab_ax = fig.add_subplot(gs[1, 1])
    rnfl_ax = fig.add_subplot(gs[1, 2])
    # if spectralis_raw_path is not None:
    #     spectralis_raw_ax = fig.add_subplot(gs[2, :])

    # print('Plotting data...')
    # show derived circle scan
    circle_ax.imshow(der_circle_scan, cmap='gray', aspect='auto')
    circle_ax.set_title('Derived circumpapillary Circle Scan')

    # plot ilm/rnfl surfaces
    # ilm_x_res = np.linspace(0, der_circle_scan.shape[1], der_ilm_surface.size)
    der_ilm_surface.plot(linewidth=1, color='cyan', ax=circle_ax)
    der_rnfl_surface.plot(linewidth=1, color='r', ax=circle_ax)
    
    # set ticks to scale
    xticks = np.array([0, 90, 180, 270, 359])
    xticks_pos = xticks * (der_circle_scan.shape[1] / 360)
    xtick_labels_region = ['T', 'S', 'N', 'I', 'T']
    xtick_labels_angle = [90, 0, 270, 180, 90] if json_collection.scan_eye=='OS' else [270, 0, 90, 180, 270]
    xtick_labels = [f'{region}\n({angle})' for region, angle in zip(xtick_labels_region, xtick_labels_angle)]

    yticks = np.array([0, 1, 2])
    yticks_pos = yticks * (der_circle_scan.shape[0] / 2)
    circle_ax.set_xticks(xticks_pos, xtick_labels)
    circle_ax.set_yticks(yticks_pos, [f'{x}\nmm' for x in yticks])
    minor_yticks = np.linspace(0,2,9) * (der_circle_scan.shape[0] / 2)
    circle_ax.yaxis.set_minor_locator(FixedLocator(minor_yticks))
    # circle_ax.axis('off')
    # circle_ax.get_xaxis().set_visible(False)
    # circle_ax.get_yaxis().set_visible(False)

    # load cirrus cmap
    with open('cirrus_cmap.pkl', 'rb') as handle:
        cirrus_cmap = pickle.load(handle)

    # ad-hoc en-face image flip (may need to remove once added to post-processing code)
    for key in ['projection_image', 'en_face_slab_image', 'rnfl_thickness_values']:
        json_collection[key] = np.flip(json_collection[key], axis=1)

    # recompute scan center for display
    json_collection['scan_center'][0] -= 2 * (json_collection['scan_center'][0] - 99)
    # json_collection['scan_center'] = find_onh_center(json_collection.rnfl_thickness_values)

    # proj_image_lim = np.percentile(json_collection.projection_image, [10, 90])
    # slab_lim = json_collection.en_face_slab_image.flatten()
    # slab_lim = slab_lim[~np.isnan(slab_lim)]
    # slab_lim = np.percentile(slab_lim, [5, 95])

    def np_replace_values(arr, value, new_value):
        arr = arr.copy()
        arr[arr==value] = new_value
        return arr

    # json_collection.projection_image[np.where(json_collection.en_face_slab_image==0)] = np.nan
    # json_collection.en_face_slab_image[np.where(json_collection.en_face_slab_image==0)] = np.nan

    # show projection, slab, and rnfl thickness maps
    plotting_proj_img = np_replace_values(json_collection.projection_image, 0, np.nan)
    plotting_slab_img = np_replace_values(json_collection.en_face_slab_image, 0, np.nan)
    proj_ax.imshow(plotting_proj_img, cmap='gray', aspect='equal')#, vmin=0, vmax=256)
    proj_ax.scatter(*json_collection['scan_center'], s=2, c='r')
    slab_ax.imshow(plotting_slab_img, cmap='gray', aspect='equal')#, vmin=0, vmax=256)
    rnfl_ax.imshow(json_collection.rnfl_thickness_values, aspect='equal', cmap=cirrus_cmap, vmin=0, vmax=350)

    proj_ax.set_title('Projection Image', y=-0.1)
    slab_ax.set_title('En Face Slab Image', y=-0.1)
    rnfl_ax.set_title('RNFL Thickness Map', y=-0.1)

    def cricle_xy(r,phi):
        center_x, center_y = json_collection.scan_center
        return center_x + r*np.cos(phi), center_y + r*np.sin(phi)
    
    for ax in [proj_ax, slab_ax, rnfl_ax]:
        phis=np.arange(0,6.28,0.01)
        circle_width = OUTER_RADIUS - INNER_RADIUS
        ax.plot( *cricle_xy(BASE_DIAM/2,phis), c='r',ls='-', alpha=.4)

    der_circle_scan_dim = der_circle_scan.shape[1], der_circle_scan.shape[0]
    # if spectralis_raw_path is not None:
    #     spectralis_raw = cv2.imread(str(spectralis_raw_path))
    #     spectralis_raw_ax.imshow(spectralis_raw, aspect='auto')
    #     spectralis_raw_ax.axis('off')
    #     spectralis_raw_ax.set_title('Spectralis Circle Scan')
    #     spectralis_info = spectralis_table[spectralis_table.eye_id==pt_id[1:]+scan_eye].iloc[0]
    #     spectralis_info = ' '.join([spectralis_info.ExamDate, spectralis_info.ExamTime])
    #     spectralis_raw_ax.text(10,spectralis_raw.shape[0]*.95,spectralis_info, c='r')

    for ax in (proj_ax, slab_ax, rnfl_ax):
        ax.axis('off')
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)

    fig.suptitle(json_collection.scan_outname)

    fig.tight_layout()
    if savefig: # save figure, return nothing
        if isinstance(savefig, str):
            fig_outpath = Path(savefig)
        else:
            fig_outpath = Path('data_wd').joinpath('plots', 'comb_plots', json_collection.scan_outname+'.png')
        fig_outpath.parent.mkdir(exist_ok=True, parents=True)
        fig.savefig(str(fig_outpath), bbox_inches='tight', facecolor='w', dpi=600)
        plt.close(fig)
        return
    return fig

def visualize_bscan(json_collection, bscan_idx, visualize_layers=False, smoothing=False, ax=None):
    '''For visualizing bscans from JSON collections with cube data embedded.'''
    bscan = np.flip(json_collection.cube_data, axis=1)[bscan_idx,:,:]
    # bscan = json_collection.cube_data[bscan_idx,:,:]
    # bscan = np.flip(bscan, axis=0)
    # bscan = cv2.resize(bscan, (768, 496))

    ILM_y = json_collection.ILM_y
    RNFL_y = json_collection.RNFL_y

    if smoothing: # pass ILM and RNFL through median filter
        ILM_y = scipy.signal.medfilt2d(ILM_y.astype(np.float32), 3)
        RNFL_y = scipy.signal.medfilt2d(RNFL_y.astype(np.float32), 3)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(bscan, aspect='auto', cmap='gray')
    ax.plot(ILM_y[bscan_idx], color='c', label='ILM')
    ax.plot(RNFL_y[bscan_idx], color='r', label='RNFL')
    ax.legend()
    if ax is None:
        return ax
    
def viz_multi_slab(json_collection):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    slab_52 = get_slab_image(
        json_collection.cube_data, json_collection.ILM_y,
        slab_width_microns=52, slab_RPE_buffer=0)

    slab_70 = get_slab_image(
        json_collection.cube_data, json_collection.ILM_y,
        slab_width_microns=70, slab_RPE_buffer=0)

    slab_62 = get_slab_image(
        json_collection.cube_data, json_collection.ILM_y,
        slab_width_microns=62, slab_RPE_buffer=10)

    ax1.imshow(slab_52, cmap='gray')
    ax2.imshow(slab_70, cmap='gray')
    ax3.imshow(slab_62, cmap='gray')

    for ax in (ax1, ax2, ax3):
        ax.axis('off')

    ax1.set_title('Slab image (52µm)')
    ax2.set_title('Slab image (70µm)')
    ax3.set_title('Slab image (10µm-62µm)')

    fig.suptitle(json_collection.scan_outname)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.subplots_adjust(top=0.85)

    fig_outpath = Path('data_wd').joinpath('plots', 'slab_comparisons', json_collection.scan_outname)
    fig_outpath.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(str(fig_outpath), bbox_inches='tight', facecolor='w', dpi=600)
    plt.close(fig)

    return