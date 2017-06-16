
"""
Created on Thu Mar 23 17:25:17 2017

@author: vf140245
Copyrignt : CEA NeuroSpin - 2014
"""
import nibabel as ni
import numpy as np
import argparse
import os.path
from glob import glob
from scipy.ndimage.morphology import binary_erosion
import seaborn as sns
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
        args = parser.parse_args(['--desc', '/home/vf140245/gits/scripts/2017_rr/metastasis/t2.json',
                                  '-o', '/tmp/WS/',
                                  '-t', '3'])
    else:
        args = parser.parse_args()
    #
    white_thresh = 0.9    
    gray_thresh = 0.9

    w_base_hist = []
    w_ws_hist = []
    g_base_hist = []
    g_ws_hist = []

    with open(args.desc) as fp:
        etude = json.load(fp)
    W = etude["correct"]
    A = etude["uncorrect"]
    M = etude["whitemask"]
    GM = etude["graymask"]
    B =  etude["out"]

    for w, a, m, gm, b in zip(W, A, M, GM, B):
        print "=============================================================="
        print w, a, m, gm
#        continue
#        w = glob(os.path.join(args.indir, s, 'model04', '*bfc_WS*.nii.gz'))
#        a = glob(os.path.join(args.indir, s, 'model02', '*bfc.nii.gz'))
#        b = [i.replace('model04', 'anat').replace('_bfc_WS', '') for i in w]
#        m = glob(os.path.join(args.indir, s, 'model03', '*pve_2*.nii.gz'))
#        gm = glob(os.path.join(args.indir, s, 'model03', '*pve_1*.nii.gz'))
        # load and mask
#        if len(a)==0 or not os.path.exists(a):
#            print "         ", b, " is not properly corrected"
#            continue
#        if len(w)==0 or not os.path.exists(w):
#            print "         ", b, " is not properly corrected"
#            continue
#        if len(m)==0 or not os.path.exists(m):
#            print "         ", b, " is not properly corrected"
#            continue
#        if len(gm)==0 or not os.path.exists(gm):
#            print "         ", b, " is not properly corrected"
#            continue
        a_img = ni.load(a)
        a_data = a_img.get_data()
        w_img = ni.load(w)
        w_data = w_img.get_data()
        m_img = erode_mask(ni.load(m), iterations=int(args.iterations))
        gm_img = erode_mask(ni.load(gm), iterations=1)
        # Process wm
        tmp_data = ((a_data - np.mean(a_data))/np.std(a_data))
        tmp_data += np.min(tmp_data)
        tmp_data *= 500.
        w_base_hist.append(tmp_data[m_img.get_data() > white_thresh])
        w_ws_hist.append(w_data[m_img.get_data() > white_thresh])
        # Process gm
        g_base_hist.append(tmp_data[gm_img.get_data() > gray_thresh])
        g_ws_hist.append(w_data[gm_img.get_data() > gray_thresh])
        # planche QC
#        from nilearn import plotting
#        display = plotting.plot_anat(ni.load(b), title="Overlay wm on anatomy", 
#                         display_mode = 'z',
#                         cut_coords = 20)
#        display.add_overlay(m_img, cmap=plotting.cm.red_transparent, threshold=white_thresh)
#        display.savefig(args.outfile+'unitary_{}'.format(s)+'.png')
#        display.close()
        #
        sns.plt.figure()
        sns.set_style('whitegrid')
        ax = sns.plt.subplot(2,2,1)
        title = 'Density in wm {}'.format(b)
        ax.set_title(title)
#        ax.set_autoscalex_on(False)
#        ax.set_xlim([0,2000])
        sns.kdeplot(np.array(w_base_hist[-1]), bw=0.5)
        ax = sns.plt.subplot(2,2,3)
        title = 'Density in wc after WS  {}'.format(b)
        ax.set_title(title)
        ax.set_autoscalex_on(False)
        ax.set_xlim([-20, 20])
        sns.kdeplot(np.array(w_ws_hist[-1]), bw=0.5)
        ax = sns.plt.subplot(2,2,2)
        title = 'Density in gm {}'.format(b)
        ax.set_title(title)
#        ax.set_autoscalex_on(False)
#        ax.set_xlim([-40, 20])
        sns.kdeplot(np.array(g_base_hist[-1]), bw=0.5)
        ax = sns.plt.subplot(2,2,4)
        title = 'Density in gm after WS  {}'.format(b)
        ax.set_title(title)
        ax.set_autoscalex_on(False)
        ax.set_xlim([-40, 20])
        sns.kdeplot(np.array(g_ws_hist[-1]), bw=0.5)
        sns.plt.savefig(args.outfile+'dist_{}.png'.format(b, dpi=300))
        sns.plt.close()
        #
        
    sns.plt.figure()
    sns.set_style('whitegrid')
    ax = sns.plt.subplot(2,2, 1)
    ax.set_title('Densities in WM before WS')
#    ax.set_autoscalex_on(False)
#    ax.set_xlim([0,2000])
    for h in w_base_hist:
        sns.kdeplot(np.array(h), bw=0.5)
    ax = sns.plt.subplot(2,2,3)
    ax.set_title('Densities in WM after WS')
    ax.set_autoscalex_on(False)
    ax.set_xlim([-40, 40])
    for h in w_ws_hist:
        sns.kdeplot(np.array(h), bw=0.5)
    ax = sns.plt.subplot(2,2,2)
    ax.set_title('Densities in GM before WS')
#    ax.set_autoscalex_on(False)
#    ax.set_xlim([-500, 1500])
    for h in g_base_hist:
        sns.kdeplot(np.array(h), bw=0.5)
    ax = sns.plt.subplot(2,2,4)
    ax.set_title('Densities in GM after WS')
    ax.set_autoscalex_on(False)
    ax.set_xlim([-60, 80])
    for h in g_ws_hist:
        sns.kdeplot(np.array(h), bw=0.5)
    sns.plt.savefig(args.outfile+'all.png', dpi=1200)
    sns.plt.close()

    #
    return w_ws_hist

if __name__ == "__main__":
    spyder_inter = True
    ws_hist = main(spyder_inter)

