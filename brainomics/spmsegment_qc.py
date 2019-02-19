#!/usr/bin/env python3
# -*- coding: utf-8 -*-
epilog = """
SPM segment QC tool. Take a lists of : grey, white matter, csf probability images
and t1 images and plot t1 (--t1 provided) with grey matter density overlay with
contours where GM probability exceed other tissue density.

Also provide csv file of tissue volumes computed as sum(probability map) * voxel size in liter.

Example:
-------

# with csf map
spmsegment_qc.py --gm data/*/c1usub-*.nii.gz --wm data/*/c2usub-*.nii.gz --t1 data/*/usub-*.nii --csf  data/*/c3usub-*.nii.gz --pdf --anat --output withcsf

# without csf map
spmsegment_qc.py --gm data/*/c1usub-*.nii.gz --wm data/*/c2usub-*.nii.gz --t1 data/*/usub-*.nii --pdf --anat --output nocsf

# rsync -rltgoDuvzn ed203246@$NS:/neurospin/psy/hbn/CBIC/derivatives/spmsegment/sub-NDARAM873GAC ./data/

python -W ignore ./src/spmsegment_qc.py --gm data/*/c1usub-*.nii.gz --wm data/*/c2usub-*.nii.gz --t1 data/*/usub-*.nii --csf  data/*/c3usub-*.nii.gz --pdf --anat --output withcsf

"""
import os, sys
import os.path
import argparse
import numpy as np
import nibabel as nib  # import generate a FutureWarning
import matplotlib.pyplot as plt
from nilearn import plotting
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import pandas as pd
#

if __name__ == "__main__":
    # Set default values to parameters
    output_dir = os.path.join(os.getcwd(), "slicesdir")
    threshold = 0.01
    alpha_overlay = 0.7
    nslices = 6
    gm_filenames = wm_filenames = csf_filenames = t1_filenames = None
    cut_coords = [-50, -25, 0, 25, 50, 75]

    # parse command line options
    #parser = optparse.OptionParser(description=__doc__)
    parser = argparse.ArgumentParser(epilog=epilog)
    parser.add_argument('--gm', help='list of grey matter proba images (c1)', nargs='+', type=str)
    parser.add_argument('--wm', help='list of white matter proba map images (c2)', nargs='+', type=str)
    parser.add_argument('--csf', help='list of csf proba map images (c3) (optional)', nargs='+', type=str)
    parser.add_argument('--t1', help='list of T1w anatomical images (optional)', nargs='+', type=str)
    parser.add_argument('-o', '--output', help='Output directory', type=str)
    parser.add_argument('--thresh', help='Threshold (default %f)'% threshold,
        default=threshold, type=float)
    parser.add_argument('--alpha_overlay', help='Overlay transparency (default %f)'% alpha_overlay,
        default=alpha_overlay, type=float)
    parser.add_argument('--anat', action='store_true', help='Plot t1 without GM density overlay')
    parser.add_argument('--pdf', action='store_true', help='Produce pdf file of slices')
    parser.add_argument('--save_seg', action='store_true', help='Save segmented images of tissues.')
    parser.add_argument('--nslices', help='Number of slices (default %f)'% nslices,
        default=nslices, type=float)
    parser.add_argument('--dry', action='store_true', help='Dry run list input files')

    options = parser.parse_args()

    ## DEBUG
    """
    options.gm = ["data/sub-NDARAM873GAC/c1usub-NDARAM873GAC_acq-VNav_T1w.nii.gz"]
    options.wm = ["data/sub-NDARAM873GAC/c2usub-NDARAM873GAC_acq-VNav_T1w.nii.gz"]
    options.csf = ["data/sub-NDARAM873GAC/c3usub-NDARAM873GAC_acq-VNav_T1w.nii.gz"]
    options.t1 = ["data/sub-NDARAM873GAC/usub-NDARAM873GAC_acq-VNav_T1w.nii"]
    #
    options.gm = ["/neurospin/psy/canbind/data/derivatives/spmsegment/sub-CAM-0002/ses-01/anat/c1sub-CAM-0002_ses-01_T1w.nii"]
    options.wm = ["/neurospin/psy/canbind/data/derivatives/spmsegment/sub-CAM-0002/ses-01/anat/c2sub-CAM-0002_ses-01_T1w.nii"]
    options.csf = ["/neurospin/psy/canbind/data/derivatives/spmsegment/sub-CAM-0002/ses-01/anat/c3sub-CAM-0002_ses-01_T1w.nii"]
    options.t1 = ["/neurospin/psy/canbind/data/derivatives/spmsegment/sub-CAM-0002/ses-01/anat/sub-CAM-0002_ses-01_T1w.nii"]
    options.output = "/neurospin/psy/canbind/data/derivatives/spmsegment/QC"

    #
    options.anat = True
    options.pdf = True
    options.nslices = 6
    options.save_seg = False
    """
    ## DEBUG
    if options.gm is None:
        #print("Error: Input is missing.")
        parser.print_help()
        raise SystemExit("Error: grey matter proba images are missing")
    gm_filenames = options.gm

    if options.wm is None:
        #print("Error: Input is missing.")
        parser.print_help()
        raise SystemExit("Error: grey matter proba images are missing")
    wm_filenames = options.wm

    t1_filenames = options.t1
    csf_filenames = options.csf

    if options.output is not None:
        output_dir = options.output
    os.makedirs(output_dir, exist_ok=True)

    assert len(gm_filenames) == len(wm_filenames), "GM and WM list are not of the same length"

    df = pd.DataFrame(dict(gm=gm_filenames, wm=wm_filenames, t1=t1_filenames, csf=csf_filenames))
    df.to_csv(os.path.join(output_dir, "spmsegment_files.csv"), index=False)
    print(df.shape)
    print(df)

    if options.dry:
        raise SystemExit("Dry run")

    def filename_to_key(filename):
        import re
        rename = re.compile( r"^(.*?)\.nii\.gz$|^(.*?)\.nii$")
        return [m for m in rename.search(os.path.basename(filename)).groups() if m is not None][0]

    keys = [filename_to_key(filename) for filename in gm_filenames]
    if len(set(keys)) != len(gm_filenames):
        keys = [filename.replace('/', '_') for filename in gm_filenames]

    if options.pdf:
        pdf = PdfPages(os.path.join(output_dir, "spmsegment_slices.pdf"))

    tissues_vol = list()
    # Iterate over images
    for i in range(len(keys)):
        # i = 0
        output_prefix = os.path.join(output_dir, keys[i])
        print(keys[i])

        # Load an threshold map, load t1
        gm_filename = gm_filenames[i]
        gm_img = nib.load(gm_filename)
        voxsize = np.asarray(gm_img.get_header().get_zooms())
        voxvol = voxsize.prod()  # mm3
        gm_arr = gm_img.get_data().squeeze()
        gm_vol = gm_arr.sum() * voxvol / (10 ** 6) # l
        gm_arr[gm_arr < threshold] = 0


        wm_filename = wm_filenames[i]
        wm_img = nib.load(wm_filename)
        wm_arr = wm_img.get_data().squeeze()
        wm_vol = wm_arr.sum() * voxvol / (10 ** 6) # l
        wm_arr[wm_arr < threshold] = 0

        csf_vol = np.NaN
        if options.csf:
            csf_filename = csf_filenames[i]
            csf_img = nib.load(csf_filename)
            csf_arr = csf_img.get_data().squeeze()
            csf_vol = csf_arr.sum() * voxvol / (10 ** 6) # l
            csf_arr[csf_arr < threshold] = 0

        tissues_vol.append([keys[i], gm_vol, wm_vol, csf_vol])

        if options.t1:
            t1_filename = t1_filenames[i]
            t1_img = nib.load(t1_filename)
            t1_arr = t1_img.get_data().squeeze()
            background_img = t1_img
        else:
            background_img = nib.Nifti1Image(np.zeros(gm_arr.shape, dtype=int),
                                             gm_img.affine)
        # Segment image base of max probabilities
        gm_msk_arr = np.zeros(gm_arr.shape, dtype=int)
        wm_msk_arr = np.zeros(gm_arr.shape, dtype=int)
        csf_msk_arr = np.zeros(gm_arr.shape, dtype=int)

        if options.csf:
            gm_msk_arr[:] = (gm_arr > wm_arr) & (gm_arr > csf_arr)
            wm_msk_arr[:] = (wm_arr > gm_arr) & (wm_arr > csf_arr)
            csf_msk_arr[:] = (csf_arr > gm_arr) & (csf_arr > wm_arr)
        else:
            gm_msk_arr[:] = (gm_arr > wm_arr)
            wm_msk_arr[:] = (wm_arr > gm_arr)

        gm_msk_img = nib.Nifti1Image(gm_msk_arr, gm_img.affine)
        wm_msk_img = nib.Nifti1Image(wm_msk_arr, wm_img.affine)

        if options.save_seg:
            gm_msk_img.to_filename(os.path.join(output_dir, gm_filename))
            wm_msk_img.to_filename(os.path.join(output_dir, wm_filename))
            if options.csf:
                csf_msk_img = nib.Nifti1Image(csf_msk_arr, csf_img.affine)
                csf_msk_img.to_filename(os.path.join(output_dir, csf_filename))

        fig = plt.figure(figsize=(19.995, 11.25))
        # 16x9: 13.33,  7.5 inches or 33.867 x 19.05 cm.
        fig.suptitle(keys[i])
        fignum = 411 if options.anat else 311

        #plt.subplot(211)
        if options.anat:
            ax = fig.add_subplot(fignum)
            plotting.plot_anat(background_img, figure=fig,axes=ax, dim=-1)
            fignum += 1
        #plotting.plot_anat(background_img, dim=-1)

        ax = fig.add_subplot(fignum + 0)

        # Workaround of a bug in nilearn
        import numbers
        if isinstance(nslices, numbers.Number):
            cuts = plotting.find_cut_slices(background_img, direction='z', n_cuts=nslices, spacing='auto')
            if len(set(cuts)) != nslices:
                nslices = cut_coords

        display = plotting.plot_anat(background_img, display_mode='z', cut_coords=nslices, figure=fig,axes=ax, dim=-1)
        #display = plotting.plot_anat(background_img, display_mode='z', cut_coords=nslices)
        display.add_overlay(gm_img, alpha=alpha_overlay, cmap=plt.cm.Greens, colorbar=True)
        display.add_contours(gm_msk_img, colors='r')

        ax = fig.add_subplot(fignum + 1)
        display = plotting.plot_anat(background_img, display_mode='y', cut_coords=nslices, figure=fig,axes=ax, dim=-1)
        #display = plotting.plot_anat(background_img, display_mode='y', cut_coords=nslices, dim=-1)
        display.add_overlay(gm_img, alpha=alpha_overlay, cmap=plt.cm.Greens, colorbar=True)
        display.add_contours(gm_msk_img, colors='r')

        ax = fig.add_subplot(fignum + 2)
        display = plotting.plot_anat(background_img, display_mode='x', cut_coords=nslices, figure=fig,axes=ax, dim=-1)
        # display = plotting.plot_anat(background_img, display_mode='x', cut_coords=nslices, dim=-1)
        display.add_overlay(gm_img, alpha=alpha_overlay, cmap=plt.cm.Greens, colorbar=True)
        display.add_contours(gm_msk_img, colors='r')

        plt.subplots_adjust(wspace=0, hspace=0, top=0.9, bottom=0.1)
        #plt.subplots_adjust(left = (5/25.4)/fig.xsize, bottom = (4/25.4)/fig.ysize, right = 1 - (1/25.4)/fig.xsize, top = 1 - (3/25.4)/fig.ysize)
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        #plt.tight_layout()
        plt.savefig(output_prefix + ".png")
        if options.pdf:
            pdf.savefig()

        plt.close(fig)

    if options.pdf:
        pdf.close()

    tissues_vol = pd.DataFrame(tissues_vol, columns = ["scan_id", "GMvol_l", "WMvol_l", "CSFvol_l"])
    tissues_vol["TIV_l"] = tissues_vol[["GMvol_l", "WMvol_l", "CSFvol_l"]].sum(axis=1)
    tissues_vol["GMratio"] = tissues_vol["GMvol_l"] / tissues_vol["TIV_l"]
    tissues_vol["WMratio"] = tissues_vol["WMvol_l"] / tissues_vol["TIV_l"]
    tissues_vol["CSFratio"] = tissues_vol["CSFvol_l"] / tissues_vol["TIV_l"]
    tissues_vol.to_csv(os.path.join(output_dir, "spmsegment_volumes.csv"), index=False)

    # Plot of Volume ratios
    pdf = PdfPages(os.path.join(output_dir, "spmsegment_volumes.pdf"))
    fig = plt.figure(figsize=(13.33, 7.5))
    df = tissues_vol[["GMratio", "WMratio", "CSFratio"]].melt()
    sns.violinplot("variable", "value", data=df, inner="quartile")
    sns.swarmplot("variable", "value", color="white", data=df, size=2, alpha=0.5)
    pdf.savefig()
    plt.close(fig)

    fig = plt.figure(figsize=(13.33, 7.5))
    sns.pairplot(tissues_vol[["GMratio", "WMratio", "CSFratio"]])
    pdf.savefig()
    plt.close(fig)
    pdf.close()
