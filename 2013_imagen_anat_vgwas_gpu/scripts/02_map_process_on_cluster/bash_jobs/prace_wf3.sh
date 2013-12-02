#!/bin/bash
#MSUB -r prace_3
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

ccc_mprun python -m soma_workflow.MPI_workflow_runner Curie_MPI --workflow /ccc/scratch/cont003/dsv/lijpeng/pa_prace_big/wf_3/mu_corr_cuda_0_39_100_149.json

