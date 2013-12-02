#!/bin/bash
#MSUB -r red_2
#MSUB -T 1800
#MSUB -q hybrid
#MSUB -Q test
#MSUB -n 64
#MSUB -A pa1753

module load python/2.7.3

export PYTHONPATH=/ccc/work/cont003/dsv/lijpeng/brainomics/git/brainomics/ml/mulm_gpu/:/ccc/work/cont003/dsv/lijpeng/brainomics/local/lib/python2.7/site-packages:$PYTHONPATH
export OMP_NUM_THREADS=1

ccc_mprun -f red2.conf

