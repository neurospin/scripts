#!/usr/bin/python
import nibabel as ni
import numpy as np
import joblib
import os
import tables
import matplotlib.pyplot as plt
from numpy import array, ones, linalg
import random
import copy
import nibabel
import scipy
import scipy.ndimage
import csv

def check_array_NaN(nparray):
    if np.isnan(nparray).any():
        raise ValueError("np.array contain NaN")


def write_csv(filename, titles, data_table):
    """
    Example
    -------
    import numpy as np
    titles = ["title1", "title2", "title3"]
    data_table = np.random.random((3,3))
    write_csv("/tmp/test.csv", titles, data_table)
    """
    with open(filename, 'wb') as csvfile:
        spamwriter = csv.writer(csvfile,
                                delimiter='\t',
                                quotechar='|',
                                quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(titles)
        for row in data_table:
            spamwriter.writerow(row)
    

def get_voxel_with_peak_score(voxel_image3d, score_image3d, mask):
    """
    voxel_image3d = np.asarray([1,2,3,5,4,6])
    score_image3d = np.asarray([1,1,2,10,3,4])
    mask = np.asarray([False, True, False, False, True, True])
    """
    max_value = np.max(score_image3d[mask])
    max_value_mask = (score_image3d == max_value)
    # It could be several voxels ?
    return np.mean(voxel_image3d[max_value_mask])

BASE_DIR = '/neurospin/brainomics/2013_imagen_anat_vgwas_gpu'
PATH_RES_RESULT = os.path.join(BASE_DIR,
                               'data',
                               'tmp_reduce',
                               'result.joblib')
DATA_DIR = os.path.join(BASE_DIR, '2013_imagen_bmi', 'data')
MASK_FILE = os.path.join(DATA_DIR, 'mask', 'mask.nii')
mask_file = MASK_FILE
OUT_IMAGES = os.path.join(BASE_DIR,
                          'interesting_snp_brain_img')

OUT_DIR = os.path.join(BASE_DIR, 'data')
OUT_HDF5_FILE_FULRES_INTER = os.path.join(OUT_DIR,
                                            'cache_full_res_inter.hdf5')
OUT_SNP_NPZ = os.path.join(OUT_DIR, 'snp')
OUT_SNP_LIST_NPY = os.path.join(OUT_DIR, 'snp_list')

RMASK_FILE = os.path.join(OUT_DIR, 'rmask.nii')

OUT_CLUSTERS_CSV = os.path.join(OUT_DIR, 'export_clusters_csv')

OUT_COV_NPY = os.path.join(OUT_DIR, 'cov.npy')
OUT_SNPS_COV_CSV = os.path.join(OUT_CLUSTERS_CSV, 'snps_cov.csv')


# Load image data
h5file = tables.openFile(OUT_HDF5_FILE_FULRES_INTER, mode="r")
images = h5file.getNode(h5file.root, 'images')

# Load geno data
geno_data = np.load(OUT_SNP_NPZ + ".npz")
geno_data = geno_data[geno_data.files[0]]

# Load snp list
snp_list_name = np.load(OUT_SNP_LIST_NPY + ".npz")
snp_list_name = snp_list_name[snp_list_name.files[0]]

# Load mast
babel_rmask = nibabel.load(RMASK_FILE)
mask = babel_rmask.get_data()
aff = ni.load(RMASK_FILE).get_affine()
mask = (mask != 0)
print "voxel num =", np.sum(mask == 1)

ar = joblib.load(PATH_RES_RESULT)
h1 = ar['h1']
h0 = ar['h0']
snp_of_interest = [122664, 379105]
n_voxels = 336188

threshold = 0.0 

for snp in snp_of_interest:
    sp_h1_snp = h1[h1['x_id'] == snp]
    # compute average grep matter value
    voxels_snp = images[:, sp_h1_snp['y_id']]
    subjects_snp = geno_data[:, snp]
    # create 3d image
    image3d_stat_mask = np.ones(mask.shape) * 0
    image1d_stat_mask = np.zeros((n_voxels))
    image1d_stat_mask[sp_h1_snp['y_id']] = sp_h1_snp['score']
    image1d_stat_mask = (image1d_stat_mask > threshold)
    image3d_stat_mask[mask] = image1d_stat_mask
    # Find clusters
    label_image3d_stat_mask, n_clusts = scipy.ndimage.label(image3d_stat_mask)
    # Save to visualize label image
    label_img = ni.Nifti1Image(label_image3d_stat_mask, aff)
    img_path = os.path.join(OUT_CLUSTERS_CSV, 'snp_%d_labels.nii.gz' % snp)
    ni.save(label_img, img_path)
    # The label 0 means the whole origin mask
    num_labels = len(np.unique(label_image3d_stat_mask))
    num_subjects = len(images)
    csv_arrary_mean = np.zeros((num_subjects, num_labels))
    csv_arrary_peak = np.zeros((num_subjects, num_labels))
    for i in np.unique(label_image3d_stat_mask):
        if i == 0:
            label_mask = mask
        else:
            label_mask = (label_image3d_stat_mask == i)
        score_image3d = np.zeros(mask.shape) * np.nan
        score_image1d = np.zeros((n_voxels))
        score_image1d[sp_h1_snp['y_id']] = sp_h1_snp['score']
        score_image3d[mask] = score_image1d
        score_image3d[(label_mask != True)] = np.nan
        img = ni.Nifti1Image(score_image3d, aff)
        img_path = os.path.join(OUT_CLUSTERS_CSV,
                                'snp_%d_label_%d.nii.gz' % (snp, i))
        ni.save(img, img_path)
        # Try to find average for each subject
        for j in xrange(len(images)):
            #j = 0
            voxel_img1d = images[j, :]
            voxel_image3d = np.zeros(mask.shape) * np.nan
            voxel_image3d[mask] = voxel_img1d
            csv_arrary_mean[j, i] = np.mean(voxel_image3d[label_mask])
            csv_arrary_peak[j, i] = get_voxel_with_peak_score(
                                            voxel_image3d,
                                            score_image3d,
                                            label_mask)
    csv_mean_path = os.path.join(OUT_CLUSTERS_CSV,
                                'snp_%d_labels_mean_voxel.csv' % snp)
    csv_peak_path = os.path.join(OUT_CLUSTERS_CSV,
                                'snp_%d_labels_peak_score_voxel.csv' % snp)
    check_array_NaN(csv_arrary_mean)
    check_array_NaN(csv_arrary_peak)
    write_csv(csv_mean_path,
              np.unique(label_image3d_stat_mask),
              csv_arrary_mean)
    write_csv(csv_peak_path,
              np.unique(label_image3d_stat_mask),
              csv_arrary_peak)

titles = ["Gender",
          "Londres",
          "Nottingham",
          "Dublin",
          "Berlin",
          "Hambourg",
          "Mannheim",
          "Paris",
          "Dresde",
          "mean_pds"]
titles = np.concatenate([snp_list_name[snp_of_interest], titles])
cov_table = np.load(OUT_COV_NPY)
insterest_snps_data = geno_data[:, snp_of_interest]
cov_table = np.column_stack([insterest_snps_data, cov_table])

if np.sum(cov_table[:, 0:2] != geno_data[:, snp_of_interest]) != 0:
    raise ValueError("Wrong stack")
write_csv(OUT_SNPS_COV_CSV, titles, cov_table)
