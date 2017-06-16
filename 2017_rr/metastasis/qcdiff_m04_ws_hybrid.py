# -*- coding: utf-8 -*-
"""
Created on Tue May  2 10:33:29 2017

@author: cp251292
Copyrignt : CEA NeuroSpin - 2014
"""
import nibabel as ni
import numpy as np
import argparse
import os.path
from glob import glob
from scipy.ndimage.morphology import binary_erosion
import seaborn as sns
import matplotlib.pyplot as plt
import json


doc = """
Command qcWS: plot distrib of standardized NAWM

python ~/gits/scripts/2017_rr/metastasis/qc_m04_ws_std.py \
       -i /neurospin/radiomics/studies/metastasis/base
"""
parser = argparse.ArgumentParser(description=doc)
parser.add_argument(
    "-v", "--verbose", dest="verbose", type=int, choices=[0, 1, 2], default=0,
    help="increase the verbosity level: 0 silent, [1, 2] verbose.")
parser.add_argument(
    "--desc", dest="desc", metavar="FILE",
    help="Json describing data")
parser.add_argument(
    "-t", "--iterations", dest="iterations", default=0, metavar="int",
    help="Num of iterations")
parser.add_argument(
    "-o", "--out", dest="outfile", metavar="FILE",
    help="Image file that will contain the resampled result")


def erode_mask(mask_image, iterations=1):
    """ Erode a binary mask file.

    Parameters
    ----------
    mask_image: Nifti image
        the mask to erode.
    iterations: int (optional, default 1)
        the number of path for the erosion.
    white_thresh: float (optional, default 1.)
        threshold to apply to mask_image.

    Returns
    -------
    erode_mask: Nifti image
        the eroded binary Nifti image.
    """
    # specific case
    if iterations == 0:
        return mask_image
    # Generate structural element
    structuring_element = np.array(
        [[[0, 0, 0],
          [0, 1, 0],
          [0, 0, 0]],
         [[0, 1, 0],
          [1, 1, 1],
          [0, 1, 0]],
         [[0, 0, 0],
          [0, 1, 0],
          [0, 0, 0]]])

    # Erode source mask
    source_data = mask_image.get_data()
    erode_data = binary_erosion(source_data, iterations=iterations,
                                structure=structuring_element)
    erode_data = erode_data.astype(source_data.dtype)
    erode_mask = ni.Nifti1Image(erode_data, mask_image.get_affine())
    
    return erode_mask

def main(spyder_inter):
    if spyder_inter:
        args = parser.parse_args(['--desc', '/home/vf140245/gits/scripts/2017_rr/metastasis/hybrid.json',
                                  '-o', '/tmp/WS_hybrid/',
                                  '-t', '3'])
    else:
        args = parser.parse_args()
    #
    white_thresh = 0.9    
    gray_thresh = 0.9
    #
    w1_base_hist = []
    w1_ws_hist = []
    g1_base_hist = []
    g1_ws_hist = []
    w2_base_hist = []
    w2_ws_hist = []
    g2_base_hist = []
    g2_ws_hist = []

    with open(args.desc) as fp:
        etude = json.load(fp)
    W1 = etude["t1correct"]
    W2 = etude["t2correct"]
    A1 = etude["t1uncorrect"]
    A2 = etude["t2uncorrect"]
    M = etude["whitemask"]
    GM = etude["graymask"]
    B =  etude["out"]

    for w1, w2, a1, a2 , m, gm, b in zip(W1, W2, A1, A2, M, GM, B):
        print "=============================================================="
        print w1, w2, a1, a2 , m, gm, b


#    for s in subjects:
#        if (not sel_s is None) and (not sel_s in s):
#            continue
#        print s
#        w1 = glob(os.path.join(args.indir, s, 'model05', '*bfc_WS_hybrid.nii.gz')) # T1 corrected image
#        w2 = glob(os.path.join(args.indir, s, 'model05', '*rAxT2_WS_hybrid.nii.gz')) # T2 corrected image
#        a1 = glob(os.path.join(args.indir, s, 'model02', '*bfc.nii.gz')) # T1 input image
#        a2 = glob(os.path.join(args.indir, s, 'model01', '*rAxT2.nii.gz')) # T2 input image
#        b1 = [i.replace('model05', 'anat').replace('_bfc_WS_hybrid', '') for i in w1] # T1 natif
#        b2 = [i.replace('model05', 'anat').replace('rAxT2_WS_hybrid', 'FLAIR') for i in w2] # T2 natif
#        m = glob(os.path.join(args.indir, s, 'model03', '*pve_2*.nii.gz'))
#        gm = glob(os.path.join(args.indir, s, 'model03', '*pve_1*.nii.gz'))
#        vx = glob(os.path.join(args.indir, s, 'model05', '*bfc_WS_hybrid_voxels.nii.gz')) # contient 3 labels : 1 vx T1, 1000 vx T2, 500 vx intersection
        
#        # load and mask
#        if len(a1)==0 or not os.path.exists(a1[0]):
#            print "         ", s, " is not properly corrected"
#            continue
#        if len(a2)==0 or not os.path.exists(a2[0]):
#            print "         ", s, " is not properly corrected"
#            continue
#        if len(w1)==0 or not os.path.exists(w1[0]):
#            print "         ", s, " is not properly corrected"
#            continue
#        if len(w2)==0 or not os.path.exists(w2[0]):
#            print "         ", s, " is not properly corrected"
#            continue
#        if len(m)==0 or not os.path.exists(m[0]):
#            print "         ", s, " is not properly corrected"
#            continue
#        if len(gm)==0 or not os.path.exists(gm[0]):
#            print "         ", s, " is not properly corrected"
#            continue
        a1_img = ni.load(a1)
        a1_data = a1_img.get_data()
        w1_img = ni.load(w1)
        w1_data = w1_img.get_data()
        a2_img = ni.load(a2)
        a2_data = a2_img.get_data()
        w2_img = ni.load(w2)
        w2_data = w2_img.get_data()
        m_img = erode_mask(ni.load(m), iterations=int(args.iterations))
        gm_img = erode_mask(ni.load(gm), iterations=1)
#        vx_img = ni.load(vx)
#        vx_img_affine = vx_img.affine
#        vx_data = vx_img.get_data()
#        vx_t1 = np.copy(vx_data)
#        vx_t1[vx_t1 > 1] = np.nan
#        vx_t2 = np.copy(vx_data)
#        vx_t2[vx_t2 < 1000] = np.nan
#        vx_inter = np.copy(vx_data)
#        vx_inter[vx_inter != 500] = np.nan
        # Process wm
        tmp1_data = ((a1_data - np.mean(a1_data))/np.std(a1_data))
        tmp1_data += np.min(tmp1_data)
        tmp1_data *= 500.
        w1_base_hist.append(tmp1_data[m_img.get_data() > white_thresh])
        w1_ws_hist.append(w1_data[m_img.get_data() > white_thresh])
        tmp2_data = ((a2_data - np.mean(a2_data))/np.std(a2_data))
        tmp2_data += np.min(tmp2_data)
        tmp2_data *= 500.
        w2_base_hist.append(tmp2_data[m_img.get_data() > white_thresh])
        w2_ws_hist.append(w2_data[m_img.get_data() > white_thresh])
        # Process gm
        g1_base_hist.append(tmp1_data[gm_img.get_data() > gray_thresh])
        g1_ws_hist.append(w1_data[gm_img.get_data() > gray_thresh])
        g2_base_hist.append(tmp2_data[gm_img.get_data() > gray_thresh])
        g2_ws_hist.append(w2_data[gm_img.get_data() > gray_thresh])
        # planche QC
#        from nilearn import plotting
#        print(b1)
#        print(b2)
#        t1_img = ni.load(b1)
#        t2_img = ni.load(b2)
#        fig = plt.figure("T1 and T2 overlays")
#        plt.subplots_adjust(hspace=0.4)
#        plt.subplot(6, 1, 1)
#        plotting.plot_roi(roi_img=m_img, bg_img=t1_img, cmap=plotting.cm.red_transparent, 
#                          title="Overlay WM on T1 anatomy", display_mode = 'z', alpha=white_thresh,
#                          cut_coords = 20)
#        plt.axis('off')
#        
#        
#        plt.subplot(6, 1, 2)
#        
#        plotting.plot_roi(roi_img=ni.Nifti1Image(vx_t1, affine=vx_img_affine), bg_img=t1_img, cmap=plotting.cm.red_transparent,
#                          title="Overlay selected voxels on T1 anatomy", display_mode = 'z', alpha=white_thresh,
#                          cut_coords = 20)
#        plt.axis('off')
#        
#        plt.subplot(6, 1, 3)
#        plotting.plot_roi(roi_img=ni.Nifti1Image(vx_inter, affine=vx_img_affine), bg_img=t1_img, cmap=plotting.cm.green_transparent, 
#                          title="Overlay voxels of intersection on T1 anatomy", display_mode = 'z', alpha=white_thresh,
#                          cut_coords = 20)
#        plt.axis('off')
#        
#        plt.subplot(6, 1, 4)
#        plotting.plot_roi(roi_img=m_img, bg_img=t2_img, cmap=plotting.cm.blue_transparent, 
#                          title="Overlay WM on T2 anatomy", display_mode = 'z', alpha=white_thresh,
#                          cut_coords = 20)
#        plt.axis('off')
#        
#        plt.subplot(6, 1, 5)
#        plotting.plot_roi(roi_img=ni.Nifti1Image(vx_t2, affine=vx_img_affine), bg_img=t2_img, cmap=plotting.cm.blue_transparent, 
#                          title="Overlay selected voxels on T2 anatomy", display_mode = 'z', alpha=white_thresh,
#                          cut_coords = 20)
#        plt.axis('off')
#
#        plt.subplot(6, 1, 6)
#        plotting.plot_roi(roi_img=ni.Nifti1Image(vx_inter, affine=vx_img_affine), bg_img=t2_img, cmap=plotting.cm.blue_transparent, 
#                          title="Overlay voxels of intersection on T2 anatomy", display_mode = 'z', alpha=white_thresh,
#                          cut_coords = 20)
#        plt.axis('off')
#        
#        file_name=args.outfile+'unitary_{}'.format(s)+'.png'
#        print(file_name)
#        fig.savefig(fname=file_name, dpi=300, format='png')
        
        #display1.savefig(args.outfile+'unitary_{}'.format(s)+'.png')
        #display1.close()
        
        #
        sns.plt.figure()
        sns.set_style('whitegrid')
        
        ### T1
        ax = sns.plt.subplot(4,2,1)
        title = 'Density in WM on T1 {}'.format(b)
        ax.set_title(title)
#        ax.set_autoscalex_on(False)
#        ax.set_xlim([0,2000])
        sns.kdeplot(np.array(w1_base_hist[-1]), bw=0.5)
        ax = sns.plt.subplot(4,2,5)
        title = 'Density in WM after WSH on T1 {}'.format(b)
        ax.set_title(title)
        ax.set_autoscalex_on(False)
        ax.set_xlim([-20, 20])
        sns.kdeplot(np.array(w1_ws_hist[-1]), bw=0.5)
        ax = sns.plt.subplot(4,2,2)
        title = 'Density in GM on T1 {}'.format(b)
        ax.set_title(title)
#        ax.set_autoscalex_on(False)
#        ax.set_xlim([-40, 20])
        sns.kdeplot(np.array(g1_base_hist[-1]), bw=0.5)
        ax = sns.plt.subplot(4,2,6)
        title = 'Density in GM after WSH on T1 {}'.format(b)
        ax.set_title(title)
        ax.set_autoscalex_on(False)
        ax.set_xlim([-40, 20])
        sns.kdeplot(np.array(g1_ws_hist[-1]), bw=0.5)
        
        ### T2
        ax = sns.plt.subplot(4,2,3)
        title = 'Density in WM on T2 {}'.format(b)
        ax.set_title(title)
#        ax.set_autoscalex_on(False)
#        ax.set_xlim([0,2000])
        sns.kdeplot(np.array(w2_base_hist[-1]), bw=0.5)
        ax = sns.plt.subplot(4,2,7)
        title = 'Density in WM after WSH on T2 {}'.format(b)
        ax.set_title(title)
        ax.set_autoscalex_on(False)
        ax.set_xlim([-20, 20])
        sns.kdeplot(np.array(w2_ws_hist[-1]), bw=0.5)
        ax = sns.plt.subplot(4,2,4)
        title = 'Density in GM on T2 {}'.format(b)
        ax.set_title(title)
#        ax.set_autoscalex_on(False)
#        ax.set_xlim([-40, 20])
        sns.kdeplot(np.array(g2_base_hist[-1]), bw=0.5)
        ax = sns.plt.subplot(4,2,8)
        title = 'Density in GM after WSH on T2 {}'.format(b)
        ax.set_title(title)
        ax.set_autoscalex_on(False)
        ax.set_xlim([-40, 20])
        sns.kdeplot(np.array(g2_ws_hist[-1]), bw=0.5)
        
        
        sns.plt.savefig(args.outfile+'dist_{}.png'.format(b), dpi=300)
        sns.plt.close()

        
    #
    sns.plt.figure()
    sns.set_style('whitegrid')
    
    ### T1
    ax = sns.plt.subplot(4,2, 1)
    ax.set_title('Densities in WM before WSH on T1')
#    ax.set_autoscalex_on(False)
#    ax.set_xlim([0,2000])
    for h in w1_base_hist:
        sns.kdeplot(np.array(h), bw=0.5)
    ax = sns.plt.subplot(4,2,5)
    ax.set_title('Densities in WM after WSH on T1')
    ax.set_autoscalex_on(False)
    ax.set_xlim([-40, 40])
    for h in w1_ws_hist:
        sns.kdeplot(np.array(h), bw=0.5)
    ax = sns.plt.subplot(4,2,2)
    ax.set_title('Densities in GM before WSH on T1')
#    ax.set_autoscalex_on(False)
#    ax.set_xlim([-500, 1500])
    for h in g1_base_hist:
        sns.kdeplot(np.array(h), bw=0.5)
    ax = sns.plt.subplot(4,2,6)
    ax.set_title('Densities in GM after WSH on T1')
    ax.set_autoscalex_on(False)
    ax.set_xlim([-60, 80])
    for h in g1_ws_hist:
        sns.kdeplot(np.array(h), bw=0.5)
    

    ### T2    
    ax = sns.plt.subplot(4,2,3)
    ax.set_title('Densities in WM before WSH on T2')
#    ax.set_autoscalex_on(False)
#    ax.set_xlim([0,2000])
    for h in w2_base_hist:
        sns.kdeplot(np.array(h), bw=0.5)
    ax = sns.plt.subplot(4,2,7)
    ax.set_title('Densities in WM after WSH on T2')
    ax.set_autoscalex_on(False)
    ax.set_xlim([-40, 40])
    for h in w2_ws_hist:
        sns.kdeplot(np.array(h), bw=0.5)
    ax = sns.plt.subplot(4,2,4)
    ax.set_title('Densities in GM before WSH on T2')
#    ax.set_autoscalex_on(False)
#    ax.set_xlim([-500, 1500])
    for h in g2_base_hist:
        sns.kdeplot(np.array(h), bw=0.5)
    ax = sns.plt.subplot(4,2,8)
    ax.set_title('Densities in GM after WSH on T2')
    ax.set_autoscalex_on(False)
    ax.set_xlim([-60, 80])
    for h in g2_ws_hist:
        sns.kdeplot(np.array(h), bw=0.5)        
        
        
    sns.plt.savefig(args.outfile+'all.png', dpi=1200)
    sns.plt.close()

    #
    return w1_ws_hist, w2_ws_hist 

if __name__ == "__main__":
    spyder_inter = True
    ws1_hist, ws2_hist = main(spyder_inter)
