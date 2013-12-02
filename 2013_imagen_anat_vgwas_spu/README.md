1. Data preparation:

Three scripts depends on two directories which are linked with symbolic link:

+ /neurospin/brainomics/2012_imagen_shfj
+ /neurospin/brainomics/2013_imagen_bmi

Any file movement in 2012_imagen_shfj and 2013_imagen_bmi could break our scripts in data prepartion.

    + "./scripts/01_subsample_and_dump.py" is used to dump all images into a HDF5 file with mask. (include verifying if images contain zeros or NaN). All the images will be saved in "/neurospin/brainomics/2013_imagen_anat_vgwas_spu/data/cache_full_res.hdf5"

    + "./scripts/02_combine_snp_vox_cov.py" is used to load image data, snp data, covariate data into memory. We find the interset between those data in terms of the same subjects.

Those are the output of files:
'/neurospin/brainomics/2013_imagen_anat_vgwas_spu/data/snp.npz' (snp data, matrix (1292, 466125))
'/neurospin/brainomics/2013_imagen_anat_vgwas_spu/data/snp_list.npy' (snp names, 466125)
'/neurospin/brainomics/2013_imagen_anat_vgwas_spu/data/cov.npy' (covariates, (1292, 10))
'/neurospin/brainomics/2013_imagen_anat_vgwas_spu/data/cache_full_res_inter.hdf5' (images, (1292, 336188))

"./scripts/03_split_into_data_chunks.py" is used to split image data and snp data into data chunks.

We split snp data into 40 chunks. The shape of snp trunk is rougly (1292, 11653).
We split image data into vox trunk with a fix size 384. The shape of vox chunk is (384, 1292) which is saved in Fortran format
We use the whole covariate matrix. 

All the data chunks will be saved in "/neurospin/brainomics/2013_imagen_anat_vgwas_spu/data".






