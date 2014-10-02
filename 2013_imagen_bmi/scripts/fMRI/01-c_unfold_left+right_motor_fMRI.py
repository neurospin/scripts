# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 17:18:28 2014

@author: hl237680

Read activation maps from left + right motor tasks fMRI acquisitions, unfold
them for each subject in order to get an array where each row corresponds to
a subject and each column to the different voxels within the subject's image.

The order is read from the file giving the list of subjects who passed the
quality control on SNPs data.

INPUT:
- '/neurospin/brainomics/2013_imagen_bmi/data/Imagen_mainSulcalMorphometry/
    full_sulci/subjects_id_full_sulci.csv':
    List of subjects who passed the quality control on sulci data
- '/neurospin/brainomics/2013_imagen_bmi/data/fMRI_GCA_motor_left_right/IMAGEN/
    arc001/processed/spmstatsintra/ ...'
    fMRI activation maps for left + right motor tasks

OUTPUT:
- '/neurospin/brainomics/2013_imagen_bmi/data/fMRI_GCA_motor_left_right/
    subjects_id_left_right_motor_fMRI.csv'
    List of subjects ID for whom we have both motor left + right fMRI tasks
    and sulci data
- '/neurospin/brainomics/2013_imagen_bmi/data/fMRI_GCA_motor_left_right/
    GCA_motor_left_right_images.npy'
    unfolded fMRI images for left + right motor tasks saved in an array-like
    format

"""

import os
import numpy as np
import pandas as pd
import nibabel as ni
from glob import glob


# Pathnames
BASE_PATH = '/neurospin/brainomics/2013_imagen_bmi/'
DATA_PATH = os.path.join(BASE_PATH, 'data')
CLINIC_DATA_PATH = os.path.join(DATA_PATH, 'clinic')
SULCI_PATH = os.path.join(DATA_PATH, 'Imagen_mainSulcalMorphometry')
FULL_SULCI_PATH = os.path.join(SULCI_PATH, 'full_sulci')
QC_PATH = os.path.join(FULL_SULCI_PATH, 'Quality_control')
IMAGEN_PATH = '/neurospin/imagen/processed/spmstatsintra/'

OUTPUT_DIR = os.path.join(DATA_PATH, 'GCA_motor_left_right')
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'GCA_motor_left_right_images.npy')
OUTPUT_MASK = os.path.join(OUTPUT_DIR, 'GCA_motor_left_right_mask.nii')

######
# run
######
if __name__ == "__main__":
    # IDs of subjects who passed the quality control on sulci data
    subjects_id = np.genfromtxt(os.path.join(FULL_SULCI_PATH,
                                             'subjects_id_full_sulci.csv'),
                                dtype=np.int64,
                                delimiter=',',
                                skip_header=1)

    subjects_id_list = ['%012d' % (int(subject)) for subject in subjects_id]

    # fMRI images of left + right motor tasks
    # (directly picked up from the IMAGEN database)
    fMRI_left_right_PATH_list = []
    fMRI_subjects_list = []
    for i, subject in enumerate(subjects_id_list):
        for file in glob(os.path.join(IMAGEN_PATH,
                                      subject,
                                      'Session*',
                                      'EPI_global',
                                      'swea_mvtroi',
                                      'con_0005.nii.gz')):
            fMRI_left_right_PATH_list.append(file)
            fMRI_subjects_list.append(
            file[len(IMAGEN_PATH):-len(
            '/SessionA/EPI_global/swea/con_0005.nii.gz')])

    # Save list of subjects ID for whom we have both left + right motor fMRI
    # tasks and sulci data as a .csv file
    subjects_in_study = pd.DataFrame.to_csv(pd.DataFrame
                                                (fMRI_subjects_list,
                                                 columns=['subject_id']),
                                        os.path.join(OUTPUT_DIR,
                                    'subjects_id_left_right_motor_fMRI.csv'),
                                        index=False)

    nb_images = len(fMRI_left_right_PATH_list)
    print "Found", nb_images, "images."

    # Read images in 4D volume
    images = None
    for i, filename in enumerate(fMRI_left_right_PATH_list):
        image = ni.load(filename)
        image_data = image.get_data()
        if images is None:
            images = np.zeros((nb_images,
                               image_data.shape[0],
                               image_data.shape[1],
                               image_data.shape[2]))
        images[i] = image_data

    # Create mask (voxels that are NaN on all subjects are removed)
    bin_mask = np.any(~np.isnan(images), axis=0)
    n_voxels_in_mask = np.count_nonzero(bin_mask)
    mask = ni.Nifti1Image(bin_mask.astype(np.uint8),
                          affine=image.get_affine())
    ni.save(mask, OUTPUT_MASK)

    # Store masked images
    masked_images = images[:, bin_mask]

    # Check extraction
    mask_indexes = np.where(bin_mask)
    vox_to_check = [0, n_voxels_in_mask - 1] + range(2014, 5000)
    for i in range(nb_images):
        for j in vox_to_check:
            a = masked_images[i, j]
            b = images[i,
                       mask_indexes[0][j],
                       mask_indexes[1][j],
                       mask_indexes[2][j]]
            if ~np.isnan(a) and ~np.isnan(b):
                if a != b:
                    print "Kata", i, j
            if np.isnan(a) ^ np.isnan(b):
                print "Kata", i, j

    # Save masked images in an array-like format
    np.save(OUTPUT_FILE, masked_images)
    print "Images saved as numpy-array under ", OUTPUT_FILE