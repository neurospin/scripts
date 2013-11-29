# -*- coding: utf-8 -*-
"""
Created on Mon Nov 5th 2013

@author: jl237561

This script perform :
    - combine snp data, voxel data, covariate data

"""
import sys
sys.path.append('/home/vf140245/gits/igutils')
import os
import igutils as ig
import numpy as np
import tables


def replace_array_value(nparray, value2replace=128):
    med = np.median(nparray, axis=0)
    if med.max() == value2replace:
        raise ValueError('too many %d', value2replace)
    nsamples = nparray.shape[0]
    med = np.tile(med, nsamples).reshape((nsamples, -1))
    mask = (nparray == value2replace)
    nparray[mask] = med[mask]


def check_array_NaN(nparray):
    if np.isnan(nparray).any():
        raise ValueError("np.array contain NaN")


BASE_DIR = '/neurospin/brainomics/2013_imagen_anat_vgwas_spu'

# Input
BASE_INPUT_DIR = os.path.join(BASE_DIR, '2013_imagen_bmi')
DATA_DIR = os.path.join(BASE_INPUT_DIR, 'data')


# Output files
BASE_INPUT_DIR = os.path.join(BASE_DIR, 'data')
OUT_DIR = BASE_INPUT_DIR
OUT_HDF5_FILE = os.path.join(OUT_DIR, 'cache.hdf5')
OUT_HDF5_FILE_FULRES=os.path.join(OUT_DIR, 'cache_full_res.hdf5')
OUT_HDF5_FILE_FULRES_INTER=os.path.join(OUT_DIR, 'cache_full_res_inter.hdf5')
OUT_SNP_NPZ = os.path.join(OUT_DIR, 'snp')
OUT_SNP_LIST_NPZ = os.path.join(OUT_DIR, 'snp_list')
OUT_COV_NPY = os.path.join(OUT_DIR, 'cov')
OUT_IMAGE_NPZ = os.path.join(OUT_DIR, 'image')
OUT_IMAGE_NO_CERE_NPZ = os.path.join(OUT_DIR, 'images_without_cerebellum')
OUT_TR_IMAGE_NPZ = os.path.join(OUT_DIR, 'tr_image')
OUT_TR_IMAGE_NO_CERE_NPZ = os.path.join(OUT_DIR, 'tr_images_without_cerebellum')
COVARIATE_CSV = os.path.join(BASE_DIR, 'inputdata', '1534bmi-vincent2.csv')


cfn = COVARIATE_CSV
# load this file to check that there are 1534 common subjects accros
#   - csv file
#   - genotyping file
#   - image file

covdata = open(cfn).read().split('\n')[:-1]
cov_header = covdata[0]
covdata = covdata[1:]
cov_subj = ["%012d" % int(i.split(',')[0]) for i in covdata]

# TODO: verify where is qc_sub_qc_gen_all_snps_common_autosome
gfn = os.path.join(DATA_DIR, 'genetics', 'qc_sub_qc_gen_all_snps_common_autosome')
genotype = ig.Geno(gfn)
geno_subj = genotype.assayIID()
geno_data = genotype.snpGenotypeAll()
replace_array_value(geno_data)
check_array_NaN(geno_data)

nb_samples = len(set(cov_subj).intersection(set(geno_subj)))

indices_cov_subj = np.in1d(np.asarray(cov_subj), np.asarray(geno_subj))
indices_geno_subj = np.in1d(np.asarray(geno_subj), np.asarray(cov_subj))

o1 = np.argsort(np.asarray(cov_subj)[indices_cov_subj])
o2 = np.argsort(np.asarray(geno_subj)[indices_geno_subj])

print "intersetion nb = ", len(set(cov_subj).intersection(set(geno_subj)))
print "nb from indices_cov_subj = ", np.sum(indices_cov_subj)
print "nb from indices_geno_subj = ", np.sum(indices_geno_subj)
print all(o1[i] <= o1[i + 1] for i in xrange(len(o1) - 1))

nb_cols = len(covdata[0].split(',')[1:])
covdata_table = np.asarray([i.split(',')[1:] for i in covdata])
covdata_table = np.asarray(covdata_table)[indices_cov_subj][o1]
covdata_table = covdata_table.astype("float64")

geno_data_table = np.asarray(geno_data)[indices_geno_subj][o2]

np.savez(OUT_SNP_NPZ, geno_data_table)
np.save(OUT_COV_NPY, covdata_table)
np.savez(OUT_SNP_LIST_NPZ, genotype.snpList())

h5file = tables.openFile(OUT_HDF5_FILE_FULRES, mode = "r")
images = h5file.getNode(h5file.root, 'images')
images = np.asarray(images)[indices_cov_subj, :][o1]
images_without_cerebellum = h5file.getNode(h5file.root, "images_without_cerebellum")
images_without_cerebellum = np.asarray(images_without_cerebellum)[indices_cov_subj, :][o1]
h5file.close()

h5file = tables.openFile(OUT_HDF5_FILE_FULRES_INTER, mode = "w", title = 'dataset_pa_prace')
atom = tables.Atom.from_dtype(images.dtype)
ds = h5file.createCArray(h5file.root, 'images', atom, images.shape)
ds[:] = images
ds = h5file.createCArray(h5file.root, 'images_without_cerebellum', atom, images_without_cerebellum.shape)
ds[:] = images_without_cerebellum
h5file.close()

print "Images reduced and dumped"
