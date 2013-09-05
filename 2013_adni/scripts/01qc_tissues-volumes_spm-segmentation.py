#!/usr/bin/env python

# -*- coding: utf-8 -*-

## QC: Quality control based on tissues volumes:
##  Compute tissue volumes based on SPM segmentation
##
## Parameters:
## in_file : a text file that contains:
##  - the site
##  - the subject ID
##  - path to the grey matter images
##  - path to the white matter images
##  - path to the CSF images
##  - path to the skull images
##  - path to the outside images
##
## out_file : a file where the results will be written
##
## Then execute the R script: "01qc_tisues-volumes_brainvisa-segmentation.R" to make a
## nice plot:
## $ Rscript 01qc_tisues-volumes_brainvisa-segmentation.R in out

import sys
import nibabel

import numpy, scipy, scipy.ndimage

def tissues_volumes(gm_im, wm_im, csf_im, skull_im, outside_im):
    gm_arr = gm_im.get_data()
    wm_arr = wm_im.get_data()
    csf_arr = csf_im.get_data()
    skull_arr = skull_im.get_data()
    outside_arr = outside_im.get_data()

    # classify
    gm_bool_arr = (gm_arr > wm_arr) & (gm_arr > csf_arr) & \
                  (gm_arr > skull_arr) & (gm_arr > outside_arr)

    # pylab.matshow(gm_bool_arr[np.int(gm_arr.shape[0]/2),:,:], cmap=pylab.cm.gray)

    wm_bool_arr = (wm_arr > gm_arr) & (wm_arr > csf_arr) & \
                  (wm_arr > skull_arr) & (wm_arr > outside_arr)

    # pylab.matshow(wm_bool_arr[np.int(wm_arr.shape[0]/2),:,:], cmap=pylab.cm.gray)

    csf_bool_arr = (csf_arr > gm_arr) & (csf_arr > wm_arr) & \
                   (csf_arr > skull_arr) & (csf_arr > outside_arr)

    # pylab.matshow(csf_bool_arr[np.int(csf_arr.shape[0]/2),:,:], cmap=pylab.cm.gray)

    # union of voxels segmeted as qm, wm or csf (all tissues)
    alltissues_arr = gm_bool_arr | wm_bool_arr | csf_bool_arr
    #pylab.matshow(alltissues_arr[np.int(alltissues_arr.shape[0]/2),:,:], cmap=pylab.cm.gray)
    gm_nomask = gm_bool_arr.sum()
    wm_nomask = wm_bool_arr.sum()
    csf_nomask = csf_bool_arr.sum()

    #    print "CCs/",;sys.stdout.flush()
    # label connected components (CCs) (26 connexity):
    # 0 is the background, features are labels 1, ..., nlabels
    alltissues_arr_labels, nlabels = scipy.ndimage.label(alltissues_arr, numpy.ones((3, 3, 3)))

    # get label (CCs) size (excluding background)
    labels_sizes, labels = scipy.histogram(alltissues_arr_labels, bins=range(1,nlabels+2))

    #    print "maxCC/",;sys.stdout.flush()
    # get brain mask: CC of maximal size wich is not the background
    brain_lab = labels[labels_sizes.argmax()]

    brain_mask_arr = (alltissues_arr_labels == brain_lab)

    #    print "morpho/",;sys.stdout.flush()
    brain_mask_arr = scipy.ndimage.binary_closing(brain_mask_arr, structure=numpy.ones((3, 3, 3)))

    #    #pylab.matshow(brain_mask_arr[np.int(brain_mask_arr.shape[0]/2),:,:], cmap=pylab.cm.gray)
    #    #pylab.matshow(brain_mask_arr[:,np.int(brain_mask_arr.shape[1]/2),:], cmap=pylab.cm.gray)
    #    #pylab.matshow(brain_mask_arr[:,:,np.int(brain_mask_arr.shape[2]/2)], cmap=pylab.cm.gray)
    #
    #    print "mask/",;sys.stdout.flush()
    # apply mask
#    gm_bool_arr[brain_mask_arr == False] = False
#    wm_bool_arr[brain_mask_arr == False] = False
#    csf_bool_arr[brain_mask_arr == False] = False

    gm = gm_bool_arr[brain_mask_arr].sum()
    wm = wm_bool_arr[brain_mask_arr].sum()
    csf = csf_bool_arr[brain_mask_arr].sum()

    return [gm, wm, csf, gm_nomask, wm_nomask, csf_nomask]

if __name__ == '__main__':
    # Parse CLI
    import argparse
    parser = argparse.ArgumentParser(description="""Compute the volumes of grey and white matter of a SPM segmentation.
                          The goal is to compare it with Brainvisa in order to detect outliers.
                          The input is a text file where each line contains the site, the subject ID and the names of the segmented images (fo each tissue).
                          The output is a CSV file that contains the volumes.""")

    # Positionnal arguments
    parser.add_argument('segmented_files', nargs='?', type=argparse.FileType('r'),
                        default=sys.stdin,
                        help='File containing the filenames (default stdin)')
    parser.add_argument('outfile', nargs='?', type=argparse.FileType('w'),
                        default=sys.stdout,
                        help ='Output file (default stdout)')
    args = parser.parse_args()

    # Write output header
    OUT_HEADER = "center"+" "+"subject"+" "+"dimX"+" "+"dimY"+" "+"dimZ"+" " \
                 "voxsizeX"+" "+"voxsizeY"+" "+"voxsizeZ"+" " \
                 "nvoxGM_nomask"+" "+"nvoxWM_nomask"+" "+"nvoxCSF_nomask"+" " \
                 "nvoxGM"+" "+"nvoxWM"+" "+"nvoxCSF"
    OUT_FORMAT="{center} {subject} {dim[0]} {dim[1]} {dim[2]} " \
                "{vox_sizes[0]} {vox_sizes[1]} {vox_sizes[2]} " \
                "{nvoxGM_nomask} {nvoxWM_nomask} {nvoxCSF_nomask} " \
                "{nvoxGM} {nvoxWM} {nvoxCSF}"
    args.outfile.write(OUT_HEADER + "\n")

    # Call comparison method for each segmented file
    for line in args.segmented_files:
        center, subject, gm_filename, wm_filename, csf_filename, skull_filename, outside_filename = line.split()
        gm_image = nibabel.load(gm_filename)
        wm_image = nibabel.load(wm_filename)
        csf_image = nibabel.load(csf_filename)
        skull_image = nibabel.load(skull_filename)
        outside_image = nibabel.load(outside_filename)

        gm_image_header = gm_image.get_header()
        gm, wm, csf, gm_nomask, wm_nomask, csf_nomask = tissues_volumes(gm_image, wm_image, csf_image, skull_image, outside_image)

        # Write volumes to file
        dat = OUT_FORMAT.format(center=center, subject=subject,
                                dim=gm_image.shape,
                                vox_sizes=gm_image_header['pixdim'][1:4],
                                nvoxGM_nomask=gm_nomask, nvoxWM_nomask=wm_nomask, nvoxCSF_nomask=csf_nomask,
                                nvoxGM=gm, nvoxWM=wm, nvoxCSF=csf)
        args.outfile.write(dat+"\n")
        args.outfile.flush()

    args.outfile.close()