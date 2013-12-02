Data preparation
----------------

Three scripts depends on two directories which are linked with symbolic link:

* /neurospin/brainomics/2012_imagen_shfj
* /neurospin/brainomics/2013_imagen_bmi

Any file movement in 2012_imagen_shfj and 2013_imagen_bmi could break our scripts in data prepartion.

* "./scripts/01_subsample_and_dump.py" is used to dump all images into a HDF5 file with mask. (include verifying if images contain zeros or NaN). All the images will be saved in 
    * "/neurospin/brainomics/2013_imagen_anat_vgwas_spu/data/cache_full_res.hdf5"

* "./scripts/02_combine_snp_vox_cov.py" is used to load image data, snp data, covariate data into memory. We find the interset between those data in terms of the same subjects. Those are the output of files:
    * '/neurospin/brainomics/2013_imagen_anat_vgwas_spu/data/snp.npz' (snp data, matrix (1292, 466125))
    * '/neurospin/brainomics/2013_imagen_anat_vgwas_spu/data/snp_list.npy' (snp names, 466125)
    * '/neurospin/brainomics/2013_imagen_anat_vgwas_spu/data/cov.npy' (covariates, (1292, 10))
    * '/neurospin/brainomics/2013_imagen_anat_vgwas_spu/data/cache_full_res_inter.hdf5' (images, (1292, 336188))

* "./scripts/03_split_into_data_chunks.py" is used to split image data and snp data into data chunks.

    * We split snp data into 40 chunks. The shape of snp trunk is rougly (1292, 11653).
    * We split image data into vox trunk with a fix size 384. The shape of vox chunk is (384, 1292) which is saved in Fortran format.
    * We use the whole covariate matrix.
    * All the data chunks will be saved in "/neurospin/brainomics/2013_imagen_anat_vgwas_spu/data".

On Cluster
----------
* Required libaries:
    * python 2.7.x
    * cuda
    * soma-workflow
    * mpi4py
    * sklearn
    * joblib
    * dill

Copy all data chunks into your cluster. In addition to those libaries, we need to copy the libary below into your cluster for GPU computing:

* /neurospin/brainomics/2013_imagen_anat_vgwas_spu/lib/brainomics

Remember that you need to $ make the libary.

```
$ cd /neurospin/brainomics/2013_imagen_anat_vgwas_spu/lib/brainomics/ml/mulm_gpu/mulm
$ make
```

You need to modify auto_wf.sh and run the script to produce workflows: 
```
./brainomics/ml/mulm_gpu/mulm/extra/auto_wf.sh
```

and then workflows will be produced for map process. For example, here is a submission bash script:

```
#!/bin/bash
#MSUB -r prace_1
#MSUB -T 18200
#MSUB -q hybrid
#MSUB -n 201
#MSUB -c 4
#MSUB -x
#MSUB -A pa1753

module load cuda
module load python/2.7.3

export PYTHONPATH=/ccc/work/cont003/dsv/lijpeng/brainomics/git/brainomics/ml/mulm_gpu/:/ccc/work/cont003/dsv/lijpeng/brainomics/local/lib/python2.7/site-packages:$PYTHONPATH
export OMP_NUM_THREADS=1

ccc_mprun python -m soma_workflow.MPI_workflow_runner Curie_MPI --workflow /ccc/scratch/cont003/dsv/lijpeng/pa_prace_big/wf_1/mu_corr_cuda_0_39_0_49.json
```

```
...
        1993,
        1994,
        1995,
        1996,
        1997,
        1998,
        1999
    ],
    "name": null,
    "serialized_jobs": {
        "344": {
            "name": "mu_corr_mapper_cuda.py x:6 y:44",
            "parallel_job_info": null,
            "native_specification": null,
            "priority": 0,
            "join_stderrout": false,
            "command": [
                "python",
                "/ccc/work/cont003/dsv/lijpeng/brainomics/git/brainomics/ml/mulm_gpu/mulm/mu_corr_mapper_cuda.py",
                "1292",
                "0",
                "10000",
                "/ccc/scratch/cont003/dsv/lijpeng/pa_prace_big/cov.npy",
                "/ccc/scratch/cont003/dsv/lijpeng/pa_prace_big/vox_chunk_44.npz",
                "/ccc/scratch/cont003/dsv/lijpeng/pa_prace_big/snp_chunk_6.npz",
                "/ccc/scratch/cont003/dsv/lijpeng/pa_prace_big/wf_1/result_6_44.joblib",
                "0.0001"
            ],
            "disposal_timeout": 168
        },
...
```

For instance, all the results have been saved on the cluster

```
./brainomics/ml/mulm_gpu/mulm/extra/auto_wf.sh
```




