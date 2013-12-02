Data preparation on PC
----------------------

Three scripts depends on two directories which are linked with symbolic link:

* /neurospin/brainomics/2012_imagen_shfj
* /neurospin/brainomics/2013_imagen_bmi

Any file movement in 2012_imagen_shfj and 2013_imagen_bmi could break our scripts in data prepartion.

* "./scripts/01_subsample_and_dump.py" is used to dump all images into a HDF5 file with mask. (include verifying if images contain zeros or NaN). All the images will be saved in 
    * "/neurospin/brainomics/2013_imagen_anat_vgwas_gpu/data/cache_full_res.hdf5"

* "./scripts/02_combine_snp_vox_cov.py" is used to load image data, snp data, covariate data into memory. We find the interset between those data in terms of the same subjects. Those are the output of files:
    * '/neurospin/brainomics/2013_imagen_anat_vgwas_gpu/data/snp.npz' (snp data, matrix (1292, 466125))
    * '/neurospin/brainomics/2013_imagen_anat_vgwas_gpu/data/snp_list.npy' (snp names, 466125)
    * '/neurospin/brainomics/2013_imagen_anat_vgwas_gpu/data/cov.npy' (covariates, (1292, 10))
    * '/neurospin/brainomics/2013_imagen_anat_vgwas_gpu/data/cache_full_res_inter.hdf5' (images, (1292, 336188))

* "./scripts/03_split_into_data_chunks.py" is used to split image data and snp data into data chunks.

    * We split snp data into 40 chunks. The shape of snp trunk is rougly (1292, 11653).
    * We split image data into vox trunk with a fix size 384. The shape of vox chunk is (384, 1292) which is saved in Fortran format.
    * We use the whole covariate matrix.
    * All the data chunks will be saved in "/neurospin/brainomics/2013_imagen_anat_vgwas_gpu/data".

Map Processing on Cluster
-------------------------
* Required libaries:
    * python 2.7.x
    * cuda
    * soma-workflow
    * mpi4py
    * sklearn
    * joblib
    * dill

Copy all data chunks into your cluster. In addition to those libaries, we need to copy the libary below into your cluster for GPU computing:

* /neurospin/brainomics/2013_imagen_anat_vgwas_gpu/lib/brainomics

Remember that you need to $ make the libary.

```
$ cd /neurospin/brainomics/2013_imagen_anat_vgwas_gpu/lib/brainomics/ml/mulm_gpu/mulm
$ make
```

You need to modify auto_wf.sh and run the script to produce workflows: 
```
./brainomics/ml/mulm_gpu/mulm/extra/auto_wf.sh
```

and then workflows will be produced for map process. All the bash job workflows have been saved in :

* [Map Processing Bash Jobs on github](https://github.com/neurospin/scripts/tree/master/2013_imagen_anat_vgwas_gpu/scripts/02_map_process_on_cluster/bash_jobs) 
* Map Processing soma workflow Jobs on nfs: /neurospin/brainomics/2013_imagen_anat_vgwas_gpu/workflows/swf_jobs.tar.gz


For example, here is a submission bash script example:

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

And a piece of codes for soma-workflow:
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

Once you run through all the bash jobs, all the results have been saved on the cluster for temporary

```
/ccc/scratch/cont003/dsv/lijpeng/pa_prace_big
```
We have made a backup and stored for one year in:
```
/ccc/store/cont003/dsv/lijpeng/brainomics/pa_prace/big/pa_prace_big
```


Reduce Processing on Cluster
----------------------------

After a long-time map process of preparation, you need to first submit two mpi bash jobs red_wf1.sh and red_wf2.sh for reduce processing in:

```
https://github.com/neurospin/scripts/tree/master/2013_imagen_anat_vgwas_gpu/scripts/03_reduce_process_on_cluster
```

The second reduce step can be called by:

```
https://github.com/neurospin/scripts/blob/master/2013_imagen_anat_vgwas_gpu/scripts/03_reduce_process_on_cluster/post_process_2.py
```

This script can obtain h0 and h1. h0 contains all the max scores of each permutation. h1 contains all the scores for all the permutations.

You can read presentation for h0 and h1 on [presentation.pptx](https://github.com/neurospin/scripts/blob/master/2013_imagen_anat_vgwas_gpu/presentation.pptx). 


Post Processing on PC
---------------------
We found two interesting snps in the data sequenses according to presentation.pptx.

```
snp_of_interest = [122664, 379105]
```

[create_brain.py](https://github.com/neurospin/scripts/blob/master/2013_imagen_anat_vgwas_gpu/scripts/04_post_process_on_pc/create_brain.py) performs the creation of brain images according to scores.

The file of /neurospin/brainomics/2013_imagen_anat_vgwas_gpu/inputdata/wmmprage000000001274.nii.gz is used for reference brain image.

