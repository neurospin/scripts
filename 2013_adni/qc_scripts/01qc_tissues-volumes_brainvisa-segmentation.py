#!/usr/bin/env python

# -*- coding: utf-8 -*-

## QC: Quality control based on tissues volumes:
##  Compute tissue volumes based on BrainVISA segmentation
##
## Parameters:
## in_file : a text file that contains:
##  - the site
##  - the subject ID
##  - path to the segmented images
##
## out_file : a file where the results will be written
##
## Then execute the R script: "01qc_tisues-volumes_brainvisa-segmentation.R" to make a
## nice plot:
## $ Rscript 01qc_tisues-volumes_brainvisa-segmentation.R in out

import sys
import nibabel

DEFAULT_GM_VALUE = 100
DEFAULT_WM_VALUE = 200

def tissues_volumes(gm_wm_image, gm_value = DEFAULT_GM_VALUE, wm_value = DEFAULT_WM_VALUE):
    gm_wm_arr = gm_wm_image.get_data()

    # classify
    #    print "classify/",;sys.stdout.flush()
    gm_bool_arr = (gm_wm_arr == gm_value)
    wm_bool_arr = (gm_wm_arr == wm_value)

    gm = gm_bool_arr.sum()
    wm = wm_bool_arr.sum()

    return [gm, wm]

if __name__ == '__main__':
    # Parse CLI
    import argparse
    parser = argparse.ArgumentParser(description="""Compute the volumes of grey and white matter of a BrainVisa segmentation.
                          The goal is to compare it with SPM in order to detect outliers.
                          The input is a text file where each line contains the site, the subject ID and the path of the segmented image.
                          The output is a CSV file that contains the volumes.""")
    parser.add_argument('--gm_value', type=int, default=DEFAULT_GM_VALUE,
                        help='Value for grey matter in segmented images (default %i)' % DEFAULT_GM_VALUE)
    parser.add_argument('--wm_value', type=int, default=DEFAULT_WM_VALUE,
                        help='Value for white matter in segmented images (default %i)' % DEFAULT_WM_VALUE)

    # Positionnal arguments
    parser.add_argument('segmented_files', nargs='?', type=argparse.FileType('r'),
                        default=sys.stdin,
                        help='File containing the filenames (default stdin)')
    parser.add_argument('outfile', nargs='?', type=argparse.FileType('w'),
                        default=sys.stdout,
                        help ='Output file (default stdout)')
    args = parser.parse_args()

    # Write output header
    OUT_HEADER = "center"+" "+"subject"+" "+"dimX"+" "+"dimY"+" "+"dimZ"+" "+ \
                 "voxsizeX"+" "+"voxsizeY"+" "+"voxsizeZ"+" "+ \
                 "nvoxGM"+" "+"nvoxWM"
    OUT_FORMAT="{center} {subject} {dim[0]} {dim[1]} {dim[2]} {vox_sizes[0]} {vox_sizes[1]} {vox_sizes[2]} {nvoxGM} {nvoxWM}"
    args.outfile.write(OUT_HEADER + "\n")

    # Call comparison method for each segmented file
    for line in args.segmented_files:
        center, subject, gm_wm_filename = line.split()
        gm_wm_image = nibabel.load(gm_wm_filename)
        gm_wm_image_header = gm_wm_image.get_header()
        nvoxGM, nvoxWM = tissues_volumes(gm_wm_image)

        # Write volumes to file
        dat = OUT_FORMAT.format(center=center, subject=subject,
                                dim=gm_wm_image.shape,
                                vox_sizes=gm_wm_image_header['pixdim'][1:4],
                                nvoxGM=nvoxGM, nvoxWM=nvoxWM)
        args.outfile.write(dat+"\n")
        args.outfile.flush()

    args.outfile.close()