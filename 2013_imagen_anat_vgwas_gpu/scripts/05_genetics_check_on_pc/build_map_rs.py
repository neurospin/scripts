# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 15:57:31 2014

@author: vf140245
Construit des cartes stats avec un contraste du type Genetic/dominant en 
lisant les snps.
"""
import os
import csv
import mulm
import tables
import numpy as np
import nibabel as ni


BASE_DIR = '/neurospin/brainomics/2013_imagen_anat_vgwas_gpu'
HDF5_FILE_FULRES_INTER = os.path.join(BASE_DIR,'data','cache_full_res_inter.hdf5')
COV_NPY = os.path.join(BASE_DIR, 'data')
SNP_NPZ = os.path.join(BASE_DIR, 'data', 'snp.npz')
SNP_LIST_NPY = os.path.join(BASE_DIR, 'data', 'snp_list.npy')
SORTED_SUBJECT_LIST = os.path.join(BASE_DIR, 'data', 'sorted_subject_list.npy')
# images files flattened
h5file = tables.openFile(HDF5_FILE_FULRES_INTER, mode="r")
images = h5file.getNode(h5file.root, 'images')

# Covariate : genre only
cov = np.load(os.path.join(COV_NPY,'cov.npy'))
cov_util = cov[:,[0,9]]
#cov_util = cov[:,0].reshape((-1,1))


#SNP
snps = np.load(SNP_NPZ)['arr_0']
snpsList = np.load(SNP_LIST_NPY)
m = np.where(snpsList=='rs13107325')[0][0]
snp = snps[:, (m-1):(m+1)]
X = np.hstack((snp, cov_util))


# Phenotype manuel
csv_mean_path = os.path.join(BASE_DIR, 'documents','2014jan24_Plink','putamen_pheno.phe' )
#y = open(csv_mean_path).read().split('\n')[1:][:-1]
y = open(csv_mean_path).read().split('\n')[1:]
y = np.asarray([float(i.split('\t')[-1]) for i in y ]).reshape((-1,1))

subject_all = np.load(SORTED_SUBJECT_LIST)
mask_subject_path =  os.path.join(BASE_DIR, 'documents','2014jan24_Plink','sorted_subject_listNonNAN.csv' )
mask_subject = open(mask_subject_path).read().split('\n')[:-1]
mask_subject = np.asarray([i.split()[-1] for i in mask_subject])
mask = np.in1d(subject_all, mask_subject, assume_unique=True)

X = X[mask]
y = y[mask]

#filename = '/neurospin/brainomics/2013_imagen_anat_vgwas_gpu/documents/2014jan24_Plink/X.csv'
#with open(filename, 'wb') as csvfile:
#    writer = csv.writer(csvfile,
#                            delimiter='\t',
#                            quotechar='|',
#                            quoting=csv.QUOTE_MINIMAL)
#    titles = ['y','rs12','rs13','gender','pds']
#    writer.writerow(titles)
#    for iy,ix in zip(y,X):
#        writer.writerow(list(iy)+list(ix))


X[:,0] = 1.0 # stocke le reg constant dans la premiere colonne

#olser = mulm.MUOLS()
#olser.fit(X,y)
#contrast = [1,0,0,0]
#olser.stats_t_coefficients(X,y, contrast, pval=True)
#contrast = [0,1,0,0]
#olser.stats_t_coefficients(X,y, contrast, pval=True)
#contrast = [0,0,1,0]
#olser.stats_t_coefficients(X,y, contrast, pval=True)


bigols = mulm.MUOLS()
y = images
ny = np.asarray(y)
y= None
ny = ny[mask, :]
bigols.fit(X, ny)
contrast = [0.,1.,0.,0.]
s, p = bigols.stats_t_coefficients(X,ny, contrast, pval=True)

unflattener_path = os.path.join(BASE_DIR, '2013_imagen_bmi', 'data','mask', 'mask.nii')
unflattener_img = ni.load(unflattener_path)
unflattener_mask = (unflattener_img.get_data() > 0)
image = np.zeros(unflattener_mask.shape)
image[unflattener_mask] = s
pn = os.path.join(BASE_DIR, 'documents','2014jan24_Plink','stats.nii.gz' )
ni.save(ni.Nifti1Image(image,unflattener_img.get_affine()), pn)
image = np.zeros(unflattener_mask.shape)
lp = -np.log10(p)
image[unflattener_mask] = lp
pn = os.path.join(BASE_DIR, 'documents','2014jan24_Plink','pval.nii.gz' )
ni.save(ni.Nifti1Image(image,unflattener_img.get_affine()), pn)

