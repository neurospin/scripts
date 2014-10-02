# -*- coding: utf-8 -*-

"""
Created on Tuesday May 14th 2014

@author: hl237680

Construit la matrice Y à partir des images d'origine, tronquées par une
bounding box, de façon à s'affranchir des limitations mémoire.
"""

import os
import mulm
import numpy as np
import nibabel as ni
from glob import glob


PROJECT_DIR = '/neurospin/brainomics/2013_imagen_anat_vgwas_gpu'
ORIGIN_IMG_DIR = os.path.join(PROJECT_DIR, '2013_imagen_bmi', 'data',
                                      'VBM', 'new_segment_spm8')
SORTED_SUBJECT_LIST = os.path.join(PROJECT_DIR, 'data',
                                   'sorted_subject_list.npy')
INIMG_FILENAME_TEMPLATE = 'smwc1{subject_id:012}*.nii'


# Part 1: Get an ordered list of images / patients

print "PART 1: get an ordered list of images / patients"

subject_all = np.load(SORTED_SUBJECT_LIST)
    
origin_img_list = []
    
for s in subject_all:
    pattern = ORIGIN_IMG_DIR+'/'+INIMG_FILENAME_TEMPLATE
    pattern = pattern.format(subject_id=int(s))
    if len(glob(pattern))==1:
        origin_img_list.append(glob(pattern)[0])
    else:
        print "ERROR cannot find ", s
        
img = ni.load(origin_img_list[0])   #load the first image
img_data = img.get_data()           #load the first image's data (!= header)

#Create mask
img_non_nul_index = np.where(img_data > 1*1e-2)    
zmin, zmax = min(img_non_nul_index[0]), max(img_non_nul_index[0])
ymin, ymax = min(img_non_nul_index[1]), max(img_non_nul_index[1])
xmin, xmax = min(img_non_nul_index[2]), max(img_non_nul_index[2])
    
#Bounding box
mask_data = img_data.copy()     #exact data copy (copie conforme)
mask_data[:] = 0.0
mask_data[(zmin+1):(zmax-1), (ymin+1):(ymax-1), (xmin+1):(xmax-1)] = 1.0
masked_data_index = (mask_data == 1.0)
    
#Load the whole dataset and mask and accumulate in images array
images = np.zeros((subject_all.shape[0], img_data[mask_data == 1].shape[0]))

for i, s in enumerate(subject_all.tolist()):
    print i
    img = ni.load(origin_img_list[i])
    img_data = img.get_data()[masked_data_index]
    images[i,:] = img_data

print images.shape


# Part 2: Construct Y data

print "PART 2: construct Y data"

COV_NPY = os.path.join(PROJECT_DIR, 'data')
SNP_NPZ = os.path.join(PROJECT_DIR, 'data', 'snp.npz')
SNP_LIST_NPY = os.path.join(PROJECT_DIR, 'data', 'snp_list.npy')

#Covariate
cov = np.load(os.path.join(COV_NPY, 'cov.npy'))
cov_util = cov[:, [0, 9]]  #considering gender and PDS
#cov_util = cov[:,:]     #considering gender, PDS and centres

#SNP
snps = np.load(SNP_NPZ)['arr_0']
snpsList = np.load(SNP_LIST_NPY)
m = np.where(snpsList == 'rs7182018')[0][0]
snp = snps[:, (m-1):(m+1)]
X = np.hstack((snp, cov_util))

#STOP    #to interact with the interpreter

#MUOLS
s_map = np.zeros(images.shape[1])
p_map = np.zeros(images.shape[1])

debut = range(0, images.shape[1], 10000)
fin = debut + [images.shape[1]]
fin = fin[1:]

for d, f in zip(debut, fin):
    print d,f
    bigols = mulm.MUOLS()
    bigols.fit(X, images[:, d:f])
    contrast = [0.,1.,0.,0.]
#    contrast = [0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]
    s, p = bigols.stats_t_coefficients(X, images[:,d:f], contrast, pval=True)
    s_map[d:f] = s[:]
    p_map[d:f] = p[:]

template_for_size = os.path.join(PROJECT_DIR, '2013_imagen_bmi', 'data',
                                 'mask', 'mask.nii')
template_for_size_img = ni.load(template_for_size)

image = np.zeros(template_for_size_img.get_data().shape)
image[masked_data_index] = s_map
pn = os.path.join(PROJECT_DIR, 'documents', '2014jan24_Plink',
                  'bbox_stats_7182018_covGenderPDS.nii.gz')
ni.save(ni.Nifti1Image(image, template_for_size_img.get_affine()), pn)
print "The statistic map has been saved."

image = np.zeros(template_for_size_img.get_data().shape)
lp = -np.log10(p_map)
image[masked_data_index] = lp
pn = os.path.join(PROJECT_DIR, 'documents', '2014jan24_Plink',
                  'bbox_pval_7182018_covGenderPDS.nii.gz')
ni.save(ni.Nifti1Image(image, template_for_size_img.get_affine()), pn)
print "The probability map has been saved."

#final path on Linux to get the images: cd /neurospin/brainomics/2013_imagen_anat_vgwas_gpu/documents/2014jan24_Plink/
