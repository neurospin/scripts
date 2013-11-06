# -*- coding: utf-8 -*-
"""
Created on Mon Nov 5th 2013

@author: vf140245

This script perform :
    - the writing in a hdf5 

It uses a recent version of igutils.
The git repository can be cloned at https://github.com:VincentFrouin/igutils.git

Add the following in the script supposing you cloned the repos in ~/gits
import sys
sys.path.append('~/gits/igutils')

"""
import sys
import getpass
sys.path.append('/home/vf140245/gits/igutils')
import os
import igutils as ig
import numpy as np
import tables


def convert_path(path):
    if getpass.getuser() == "jl237561":
        path = '~' + path
        path = os.path.expanduser(path)
    return path

# Input
BASE_DIR='/neurospin/brainomics/2013_imagen_bmi/'
BASE_DIR = convert_path(BASE_DIR)
DATA_DIR=os.path.join(BASE_DIR, 'data')
CLINIC_DIR=os.path.join(DATA_DIR, 'clinic')

# Output files
OUT_DIR=os.path.join(DATA_DIR, 'dataset_pa_prace')
OUT_HDF5_FILE=os.path.join(OUT_DIR, 'cache.hdf5')


cfn = os.path.join(CLINIC_DIR, '1534bmi-vincent2.csv')
# load this file to check that there are 1534 common subjects accros
#   - csv file
#   - genotyping file
#   - image file

covdata = open(cfn).read().split('\n')[:-1]
cov_header = covdata[0]
covdata = covdata[1:]
cov_subj = ["%012d"%int(i.split(',')[0]) for i in covdata]

gfn = os.path.join(DATA_DIR, 'qc_sub_qc_gen_all_snps_common_autosome')
genotype = ig.Geno(gfn)
geno_subj = genotype.assayIID()
geno_data = genotype.snpGenotypeAll()

nb_samples = len(set(cov_subj).intersection(set(geno_subj)))

indices_cov_subj = np.in1d(np.asarray(cov_subj), np.asarray(geno_subj))
indices_geno_subj = np.in1d(np.asarray(geno_subj), np.asarray(cov_subj))

print "intersetion nb = ", len(set(cov_subj).intersection(set(geno_subj)))
print "nb from indices_cov_subj = ", np.sum(indices_cov_subj)
print "nb from indices_geno_subj = ", np.sum(indices_geno_subj)

nb_cols = len(covdata[0].split(',')[1:-1])
covdata_table = np.asarray([i.split(',')[1:-1] for i in covdata])
# covdata_table = np.random.random((4,10))
# indices_cov_subj = np.asarray([False, True, True, True])
# geno_data = np.random.random((4,10))
# indices_geno_subj = np.asarray([False, True, True, True])
covdata_table = covdata_table[indices_cov_subj]
geno_data_table = geno_data[indices_geno_subj]


# ==========================================================================
# Load images
# OUT_HDF5_FILE = "/tmp/data"
# images = np.random.random((4, 50))
# images_without_cerebellum = np.random.random((4, 50))
# h5file = tables.openFile(OUT_HDF5_FILE, mode = "w", title = 'dataset_pa_prace')
# atom = tables.Atom.from_dtype(images.dtype)
# filters = tables.Filters(complib='zlib', complevel=5)
# ds = h5file.createCArray(h5file.root, 'images', atom, images.shape, filters=filters)
# ds[:] = images
# ds = h5file.createCArray(h5file.root, 'images_without_cerebellum', atom, images_without_cerebellum.shape, filters=filters)
# ds[:] = images_without_cerebellum
# h5file.close()
h5file = tables.openFile(OUT_HDF5_FILE, mode = "r+")
images = h5file.getNode(h5file.root, 'images')
images = np.asarray(images)[indices_cov_subj, :]
images_without_cerebellum = h5file.getNode(h5file.root, "images_without_cerebellum")
images_without_cerebellum = np.asarray(images_without_cerebellum)[indices_cov_subj, :]

h5file.removeNode(h5file.root, 'images')
h5file.removeNode(h5file.root, 'images_without_cerebellum')
atom = tables.Atom.from_dtype(images.dtype)
filters = tables.Filters(complib='zlib', complevel=5)
ds = h5file.createCArray(h5file.root, 'images', atom, images.shape, filters=filters)
ds[:] = images
ds = h5file.createCArray(h5file.root, 'images_without_cerebellum', atom, images_without_cerebellum.shape, filters=filters)
ds[:] = images_without_cerebellum
ds = h5file.createCArray(h5file.root, 'covdata', atom, covdata_table.shape, filters=filters)
ds[:] = covdata_table
ds = h5file.createCArray(h5file.root, 'geno_data', atom, geno_data_table.shape, filters=filters)
ds[:] = geno_data_table
h5file.close()

print "Images reduced and dumped"



