# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 09:49:05 2014

@author: vf140245
"""
import sys
sys.path.append('/home/vf140245/gits/igutils')
import os
import igutils as ig
import numpy as np
import tables


BASE_DIR = '/neurospin/brainomics/2013_imagen_anat_vgwas_gpu'
OUT_DIR = os.path.join(BASE_DIR, 'data')
OUT_HDF5_FILE = os.path.join(OUT_DIR, 'cache.hdf5')
OUT_HDF5_FILE_FULRES = os.path.join(OUT_DIR, 'cache_full_res.hdf5')
OUT_HDF5_FILE_FULRES_INTER = os.path.join(OUT_DIR, 'cache_full_res_inter.hdf5')
OUT_SNP_NPZ = os.path.join(OUT_DIR, 'snp')
OUT_SNP_LIST_NPY = os.path.join(OUT_DIR, 'snp_list')
OUT_COV_NPY = os.path.join(OUT_DIR, 'cov')
OUT_IMAGE_NPZ = os.path.join(OUT_DIR, 'image')
OUT_IMAGE_NO_CERE_NPZ = os.path.join(OUT_DIR, 'images_without_cerebellum')
OUT_TR_IMAGE_NPZ = os.path.join(OUT_DIR, 'tr_image')
OUT_TR_IMAGE_NO_CERE_NPZ = os.path.join(OUT_DIR, 'tr_images_without_cerebellum')
COVARIATE_CSV = os.path.join(BASE_DIR, 'inputdata', '1534bmi-vincent2.csv')
OUT_SORTED_SUBJECT_LIST = os.path.join(OUT_DIR, 'sorted_subject_list.npy')

#population etudiée par le run GPU
p =open("/home/vf140245/temp/sorted_subject_list2.csv").read().split('\n')[:-1]
p = [p.split()[0] for i in p]
#genotype data
gfn = os.path.join(BASE_DIR,
                   '2012_imagen_shfj',
                   'genetics',
                   'qc_sub_qc_gen_all_snps_common_autosome')
genotype = ig.Geno(gfn)
genotype.setOrderedSubsetIndiv(p)
snp = genotype.snpGenotypeByName('rs13107325')
snp
np.sum(snp==128)
snpI = snp.copy()
np.sum(snpI==128)
#imputation performed in the run
replace_array_value(snpI)
check_array_NaN(snpI)
np.sum(snpI==128)
snp[ind]
snpI[ind]
#individus concernes par les imputations
np.asarray(p)[ind]