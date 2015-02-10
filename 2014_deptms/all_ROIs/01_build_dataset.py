# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 11:43:03 2014

@author: cp243490

Read the data (clinic data, ROI, MRI AND PET images)
Construct the masks, Build and save the datasets given a modality
(MRI/PET/MRI+PET) for:
    - the whole brain (implicit mask)
    - each ROI (using harvard and oxford atlas to define the regions and
                the implicit mask and finally dilate the obtained mask
                to make sure all the region is contained in the mask )
Construct the matrix X and y for the regression.
X is constructed for each (modality, ROI).
Each row of the X contains and Intercept, the age, the sex and the image of
the patient). X is centered ans scaled.

Masks, X and y for the modality MRI+PET are built by
concatenating MRI and PET masks and matrices X
X associated to modality MRI+PET contain an Intercept, the age, the sex, MRI
image and PET image.

INPUTs
- Clinic data:
    /neurospin/brainomics/2014_deptms/base_data/clinic/deprimPetInfo.csv
- ROIs : /neurospin/brainomics/2014_deptms/ROI_labels/ROI_labels.csv
- resampled cortical and subcortical harvard oxford atlases:
    /neurospin/brainomics/2014_deptms/base_data/images/atalses/
        - HarvardOxford-sub-maxprob-thr0-1mm-nn.nii.gz
        - HarvardOxford-cort-maxprob-thr0-1mm-nn.nii.gz
- MRI images :
/neurospin/brainomics/2014_deptms/base_data/images/MRI_images/smwc1*.img
- PET images :
/neurospin/brainomics/2014_deptms/base_data/images/PET_j0Scaled_images/smw*.img

OUTPUTs
- implicit mask for the whole brain and mask for each ROI :
    /neurospin/brainomics/2014_deptms/results_univariate/*/mask_*.nii
- X for the whole brain and for each ROI associated to a modality :
    /neurospin/brainomics/2014_deptms/results_univariate/*/X*.npy
    (Intercept + Age + Gender + images
    for the whole brain and each ROI)
- y : /neurospin/brainomics/2014_deptms/results_univariate/*/y.npy
    (response to the treatment)

"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import ndimage
import shutil

REP_MAP = {"N": 0, "Y": 1}

MODALITIES = ["MRI", "PET", "MRI+PET"]

BASE_PATH = "/neurospin/brainomics/2014_deptms"

BASE_DATA_PATH = os.path.join(BASE_PATH,    "base_data")

INPUT_CSV = os.path.join(BASE_DATA_PATH,    "clinic", "deprimPetInfo.csv")
INPUT_ROIS_CSV = os.path.join(BASE_DATA_PATH,    "ROI_labels.csv")

OUTPUT_DATASET = os.path.join(BASE_PATH,    "datasets")

if not os.path.exists(OUTPUT_DATASET):
    os.makedirs(OUTPUT_DATASET)

#############################################################################
## Read pop csv
pop = pd.read_csv(INPUT_CSV, sep="\t")
pop['Response.num'] = pop["Response"].map(REP_MAP)

#############################################################################
## Read ROIs csv
atlas = []
dict_rois = {}
df_rois = pd.read_csv(INPUT_ROIS_CSV)
for i, ROI_name_aal in enumerate(df_rois["ROI_name_aal"]):
    cur = df_rois[df_rois.ROI_name_aal == ROI_name_aal]
    label_ho = cur["label_ho"].values[0]
    atlas_ho = cur["atlas_ho"].values[0]
    roi_name = cur["ROI_name_deptms"].values[0]
    if ((not cur.isnull()["atlas_ho"].values[0])
        and (not cur.isnull()["ROI_name_deptms"].values[0])):
        if not roi_name in dict_rois:
            print "ROI: ", roi_name
            labels = np.asarray(label_ho.split(), dtype="int")
            dict_rois[roi_name] = [labels]
            dict_rois[roi_name].append(atlas_ho)
            print dict_rois[roi_name]
            print "\n"

#############################################################################
## Build datasets for all the Modalities
for MODALITY in MODALITIES:
    OUTPUT_MODALITY = os.path.join(OUTPUT_DATASET, MODALITY)
    if not os.path.exists(OUTPUT_MODALITY):
        os.makedirs(OUTPUT_MODALITY)
for MODALITY in MODALITIES:
    print "Modality: ", MODALITY
    OUTPUT_MODALITY = os.path.join(OUTPUT_DATASET, MODALITY)
    if np.logical_or(MODALITY == "MRI", MODALITY == "PET"):
        #####################################################################
        # Read images
        n = len(pop)
        assert n == 34
        Z = np.zeros((n, 3))  # Intercept + Age + Gender
        Z[:, 0] = 1  # Intercept
        y = np.zeros((n, 1))  # Response
        images = list()
        fileName = ""
        image_path = ""
        if MODALITY == "MRI":
            fileName = "MRIm_G_file"
            image_path = "MRI_images"
        elif MODALITY == "PET":
            fileName = "PET_file"
            image_path = "PET_j0Scaled_images"

        for i, MRIm_G_file in enumerate(pop["MRIm_G_file"]):
            cur = pop[pop.MRIm_G_file == MRIm_G_file]
            # print cur
            f = cur[fileName].values[0]
            imagefile_name = os.path.join(BASE_DATA_PATH,
                                           "images",
                                           image_path,
                                           f)
            babel_image = nib.load(imagefile_name)
            images.append(babel_image.get_data().ravel())
            Z[i, 1:] = np.asarray(cur[["Age", "Sex"]]).ravel()
            y[i, 0] = cur["Response.num"]

        shape = babel_image.get_data().shape

        #####################################################################
        # resample one anat
        fsl_cmd = "fsl5.0-applywarp -i %s -r %s -o %s" % \
        ("/usr/share/data/fsl-mni152-templates/MNI152_T1_1mm_brain.nii.gz",
        imagefile_name,
        os.path.join(BASE_DATA_PATH, "images", "atlases",
                     "MNI152_T1_1mm_brain.nii.gz"))

        os.system(fsl_cmd)
        #####################################################################
        # Compute implicit mask and X and y matrix for the whole brain

        # Compute mask
        # Implicit Masking involves assuming that a lower than a given
        # threshold at some voxel, in any of the images, indicates an unknown
        # and is excluded from the analysis.
        # maxk from MRI images
        Xtot = np.vstack(images)
        if MODALITY == "MRI":
            mask = (np.min(Xtot, axis=0) > 0.01)\
                    & (np.std(Xtot, axis=0) > 1e-6)
            mask = mask.reshape(shape)

            out_im = nib.Nifti1Image(mask.astype("int16"),
                                         affine=babel_image.get_affine())
            out_im.to_filename(os.path.join(OUTPUT_DATASET, "MRI",
                                            "mask_MRI_brain.nii"))
            out_im.to_filename(os.path.join(OUTPUT_DATASET, "PET",
                                            "mask_PET_brain.nii"))
            out_im.to_filename(os.path.join(OUTPUT_DATASET, "MRI+PET",
                                            "mask_PET_brain.nii"))
            out_im.to_filename(os.path.join(OUTPUT_DATASET, "MRI+PET",
                                            "mask_MRI_brain.nii"))
            babel_mask = nib.load(os.path.join(OUTPUT_MODALITY,
                                            "mask_" + MODALITY + "_brain.nii"))
            assert np.all(mask == (babel_mask.get_data() != 0))

        mask = nib.load(os.path.join(OUTPUT_DATASET, MODALITY,
                          "mask_" + MODALITY + "_brain.nii")).get_data() != 0

        #####################################################################
        # Xcsi for the whole brain
        X = Xtot[:, mask.ravel()]
        X = np.hstack([Z, X])
        X -= X.mean(axis=0)
        X /= X.std(axis=0)
        X[:, 0] = 1.
        n, p = X.shape
        np.save(os.path.join(OUTPUT_MODALITY, "X_" + MODALITY + "_brain.npy"),
                X)
        fh = open(os.path.join(OUTPUT_MODALITY,
                     "X_" + MODALITY + "_brain.npy").replace("npy", "txt"),
                  "w")
        fh.write('Centered and scaled data. ' \
                 'Shape = (%i, %i): Intercept + Age + Gender + %i voxels' % \
                 (n, p, mask.sum()))
        fh.close()

        #####################################################################
        # Do the same for each ROI
        #####################################################################
        # Compute binarized mask with atlas for the specific ROIs

        # Resample HarvardOxford atlas (cortical and subcortical) into
        # reference space and interpolation

        #fsl_cmd = "fsl5.0-applywarp -i %s -r %s -o %s --interp=nn"
        #ref = imagefile_name[0]
        #input_sub_filename = os.path.join("/usr/share/data",
        #                      "harvard-oxford-atlases",
        #                      "HarvardOxford",
        #                      "HarvardOxford-sub-maxprob-thr0-1mm.nii.gz")
        #input_cort_filename = os.path.join("/usr/share/data",
        #                      "harvard-oxford-atlases",
        #                      "HarvardOxford",
        #                      "HarvardOxford-cort-maxprob-thr0-2mm.nii.gz")
        #output_sub_filename = os.path.join(BASE_PATH, "images", "atlases",
        #                      "HarvardOxford-sub-maxprob-thr0-1m-nn.nii.gz")
        #output_cort_filename = os.path.join(BASE_PATH, "images", "atlases",
        #                      "HarvardOxford-cort-maxprob-thr0-1mm-nn.nii.gz")
        #os.system(fsl_cmd % (input_sub_filename, ref, output_sub_filename))
        #os.system(fsl_cmd % (input_cort_filename, ref, output_cort_filename))

        sub_image = nib.load(os.path.join(BASE_DATA_PATH,
                              "images",
                              "atlases",
                              "HarvardOxford-sub-maxprob-thr0-1mm-nn.nii.gz"))
        sub_arr = sub_image.get_data()
        cort_image = nib.load(os.path.join(BASE_DATA_PATH,
                              "images",
                              "atlases",
                              "HarvardOxford-cort-maxprob-thr0-1mm-nn.nii.gz"))
        cort_arr = cort_image.get_data()

        # Compute the mask associated to each ROI by
        # applying the implicit mask to the atlas
        # and by identifying the ROI in the atlas (with the labels)
        # Then dilate the mask
        mask = nib.load(os.path.join(OUTPUT_DATASET, MODALITY,
                               "mask_" + MODALITY + "_brain.nii")).get_data()
        for ROI, values in dict_rois.iteritems():
            print "ROI: ", ROI
            labels = values[0]
            print "labels", labels
            atlas = values[1]
            print "atlas: ", atlas

            # Compute the mask associated to each ROI by
            # applying the implicit mask to the atlas
            # and by identifying the ROI in the atlas (with the labels)
            # Then dilate the mask
            if (atlas == "sub") or (atlas == "cort"):
                if atlas == "sub":
                    mask_atlas_ROI = np.copy(sub_arr)
                elif atlas == "cort":
                    mask_atlas_ROI = np.copy(cort_arr)
                if len(labels) == 1:
                    mask_atlas_ROI[np.logical_or(np.logical_not(mask),
                                            mask_atlas_ROI != labels[0])] = 0
                mask_atlas_ROI[np.logical_not(mask)] = 0
                mask_ROI = np.zeros(mask_atlas_ROI.shape)
                for lab in labels:
                    mask_ROI[mask_atlas_ROI == lab] = 1
                mask_atlas_ROI[np.logical_not(mask_ROI)] = 0
                np.unique(mask_atlas_ROI)

                # dilate
                # 3x3 structuring element with connectivity 1 and 2 iterations
                mask_bool_ROI = ndimage.morphology.binary_dilation(
                                    mask_atlas_ROI,
                                    iterations=2).astype(mask_atlas_ROI.dtype)
                mask_bool_ROI = mask_bool_ROI.astype("bool")
                out_im = nib.Nifti1Image(mask_bool_ROI.astype("int16"),
                                         affine=babel_image.get_affine())
                out_im.to_filename(os.path.join(OUTPUT_MODALITY,
                                    "mask_" + MODALITY + "_" + ROI + ".nii"))
                im = nib.load(os.path.join(OUTPUT_MODALITY,
                                     "mask_" + MODALITY + "_" + ROI + ".nii"))
                assert np.all(mask_bool_ROI == im.get_data())
                # Xcsi for the specific ROIs
                X_ROI = Xtot[:, mask_bool_ROI.ravel()]
                X_ROI = np.hstack([Z, X_ROI])
                X_ROI -= X_ROI.mean(axis=0)
                X_ROI /= X_ROI.std(axis=0)
                X_ROI[:, 0] = 1.
                n, p = X_ROI.shape
                np.save(os.path.join(OUTPUT_MODALITY,
                                "X_" + MODALITY + "_" + ROI + ".npy"), X_ROI)
                fh = open(os.path.join(OUTPUT_MODALITY,
                                "X_" + MODALITY + "_" + ROI +
                                        ".npy").replace("npy", "txt"), "w")
                fh.write('Centered and scaled data. ' \
                    'Shape = (%i, %i): Intercept + Age + Gender + %i voxels' % \
                    (n, p, mask_bool_ROI.sum()))
                fh.close()
                print '\n'
        # Treat case maskdep separately
        # (need to access MAskdep-sub ans Maskdep-cort data)
        ROI = "Maskdep"
        print "ROI: ", ROI
        labels = values[0]
        print "labels", labels
        atlas = values[1]
        print "atlas: ", atlas
        assert (dict_rois[ROI][1] == "both")
        mask_sub_im = nib.load(os.path.join(OUTPUT_MODALITY,
                                "mask_" + MODALITY + "_Maskdep-sub.nii"))
        mask_sub_arr = mask_sub_im.get_data() != 0
        mask_cort_im = nib.load(os.path.join(OUTPUT_MODALITY,
                               "mask_" + MODALITY + "_Maskdep-cort.nii"))
        mask_cort_arr = mask_cort_im.get_data() != 0
        mask_bool_ROI = (mask_sub_arr | mask_cort_arr)
        out_im = nib.Nifti1Image(mask_bool_ROI.astype("int16"),
                                     affine=babel_image.get_affine())
        out_im.to_filename(os.path.join(OUTPUT_MODALITY,
                            "mask_" + MODALITY + "_" + ROI + ".nii"))
        im = nib.load(os.path.join(OUTPUT_MODALITY,
                             "mask_" + MODALITY + "_" + ROI + ".nii"))
        assert np.all(mask_bool_ROI == im.get_data())

        X_ROI = Xtot[:, mask_bool_ROI.ravel()]
        X_ROI = np.hstack([Z, X_ROI])
        X_ROI -= X_ROI.mean(axis=0)
        X_ROI /= X_ROI.std(axis=0)
        X_ROI[:, 0] = 1.
        n, p = X_ROI.shape
        np.save(os.path.join(OUTPUT_MODALITY,
                            "X_" + MODALITY + "_" + ROI + ".npy"), X_ROI)
        fh = open(os.path.join(OUTPUT_MODALITY,
                            "X_" + MODALITY + "_" + ROI +
                                ".npy").replace("npy", "txt"), "w")
        fh.write('Centered and scaled data. ' \
            'Shape = (%i, %i): Intercept + Age + Gender + %i voxels' % \
            (n, p, mask_bool_ROI.sum()))
        fh.close()
        print '\n'

        np.save(os.path.join(OUTPUT_MODALITY, "y.npy"), y)

    elif (MODALITY == "MRI+PET"):
        #####################################################################
        # Compute MRI+PET implicit mask and X and y matrix for the whole brain
        
        # GET MRI AND PET matrices X, y, mask for the whole brain
        n = len(pop)
        assert n == 34
        Z = np.zeros((n, 3))  # Intercept + Age + Gender
        Z[:, 0] = 1  # Intercept

        X_mri = np.load(os.path.join(OUTPUT_DATASET, "MRI", "X_MRI_brain.npy"))
        X_pet = np.load(os.path.join(OUTPUT_DATASET, "PET", "X_PET_brain.npy"))

        Z[:, 1:3] = X_mri[:, 1:3]  # get Intercerpt + Age + Sex data
        X_mri = X_mri[:, 3:]  # remove Intercerpt + Age + Sex data
        X_pet = X_pet[:, 3:]  # remove Intercerpt + Age + Sex data

        # rep_no_rep
        y = np.load(os.path.join(OUTPUT_DATASET, "MRI", "y.npy"))

        mask_ima_mri = nib.load(os.path.join(OUTPUT_MODALITY,
                                             "mask_MRI_brain.nii"))
        mask_arr_mri = mask_ima_mri.get_data() != 0

        mask_ima_pet = nib.load(os.path.join(OUTPUT_MODALITY,
                                             "mask_PET_brain.nii"))
        mask_arr_pet = mask_ima_pet.get_data() != 0

        #####################################################################
        # X and the masks for MRI + PET modality
        X = np.hstack((X_mri, X_pet))

        mask = np.vstack((mask_arr_mri, mask_arr_pet))
        #####################################################################
        # Xcsi for the whole brain

        X = np.hstack([Z, X])
        X -= X.mean(axis=0)
        X /= X.std(axis=0)
        X[:, 0] = 1.
        n, p = X.shape
        np.save(os.path.join(OUTPUT_MODALITY, "X_" + MODALITY + "_brain.npy"),
                X)
        fh = open(os.path.join(OUTPUT_MODALITY,
                               "X_" + MODALITY +
                               "_brain.npy").replace("npy", "txt"), "w")
        fh.write('Centered and scaled data. '\
                 'Shape = (%i, %i): Intercept + Age + Gender + '\
                 '(%i, %i) voxels' % \
                 (n, p, mask_arr_mri.sum(), mask_arr_pet.sum()))
        fh.close()

        np.save(os.path.join(OUTPUT_MODALITY, "y.npy"), y)

        #####################################################################
        # Do the same for each ROI
        #####################################################################

        for ROI, values in dict_rois.iteritems():
            print "ROI: ", ROI
            print "labels", labels

            # GET MRI AND PET matrices X, y, mask for each ROI
            X_mri_ROI = np.load(os.path.join(OUTPUT_DATASET,
                                             "MRI", "X_MRI_" + ROI + ".npy"))
            X_pet_ROI = np.load(os.path.join(OUTPUT_DATASET,
                                             "PET", "X_PET_" + ROI + ".npy"))
            X_mri_ROI = X_mri_ROI[:, 3:]  # remove Intercerpt + Age + Sex data
            X_pet_ROI = X_pet_ROI[:, 3:]  # remove Intercerpt + Age + Sex data

            mask_ima_mri_roi = nib.load(os.path.join(OUTPUT_DATASET,
                                                   "MRI",
                                                   "mask_MRI_" + ROI + ".nii"))
            mask_arr_mri_roi = mask_ima_mri_roi.get_data() != 0
            shutil.copyfile(os.path.join(OUTPUT_DATASET,
                                                   "MRI",
                                                   "mask_MRI_" + ROI + ".nii"),
                            os.path.join(OUTPUT_MODALITY,
                                            "mask_MRI_" + ROI + ".nii"))

            mask_ima_pet_roi = nib.load(os.path.join(OUTPUT_DATASET,
                                                   "PET",
                                                   "mask_PET_" + ROI + ".nii"))
            mask_arr_pet_roi = mask_ima_pet_roi.get_data() != 0
            shutil.copyfile(os.path.join(OUTPUT_DATASET,
                                                   "PET",
                                                   "mask_PET_" + ROI + ".nii"),
                            os.path.join(OUTPUT_MODALITY,
                                            "mask_PET_" + ROI + ".nii"))

            # Create X_ROI and the masks for MRI + PET modality associated to
            # each ROI
            X_ROI = np.hstack((X_mri_ROI, X_pet_ROI))

            mask_ROI = np.vstack((mask_arr_mri_roi, mask_arr_pet_roi))

            # Xcsi for the specific ROI
            X_ROI = np.hstack([Z, X_ROI])
            X_ROI -= X_ROI.mean(axis=0)
            X_ROI /= X_ROI.std(axis=0)
            X_ROI[:, 0] = 1.
            n, p = X_ROI.shape
            np.save(os.path.join(OUTPUT_MODALITY,
                                "X_" + MODALITY + "_" + ROI + ".npy"), X_ROI)
            fh = open(os.path.join(OUTPUT_MODALITY,
                                "X_" + MODALITY + "_" + ROI +
                                          ".npy").replace("npy", "txt"), "w")
            fh.write('Centered and scaled data. '\
              'Shape = (%i, %i): Intercept + Age + Gender + '\
              '(%i, %i) voxels' % \
              (n, p, mask_arr_mri_roi.sum(), mask_arr_pet_roi.sum()))
            fh.close()