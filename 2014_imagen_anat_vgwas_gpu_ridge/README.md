Data preparation on PC
----------------------

* [01_dump_images_and_snps.py](https://github.com/neurospin/scripts/blob/master/2014_imagen_anat_vgwas_gpu_ridge/scripts/01_data_preparation_on_pc/01_dump_images_and_snps.py) is used to dump images, snps, and snp list. (include verifying if images contain zeros or NaN, and verifying if snps contain NaN or 128). All the data will be saved in 
    * "/neurospin/brainomics/2014_imagen_anat_vgwas_gpu_ridge/data/images.hdf5"
    * "/neurospin/brainomics/2014_imagen_anat_vgwas_gpu_ridge/data/snps.npz"
    * "/neurospin/brainomics/2014_imagen_anat_vgwas_gpu_ridge/data/snps_list.npy"
    * "/neurospin/brainomics/2014_imagen_anat_vgwas_gpu_ridge/data/cov.npy"