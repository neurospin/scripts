python 02_tvenet.py
user_func /home/ed203246/git/scripts/2013_adni/MCIc-CTL/02_tvenet.py
# Make sure parent dir of wd_cluster exists
ssh ed203246@gabriel.intra.cea.fr "mkdir /neurospin/tmp/ed203246"
mkdir: cannot create directory `/neurospin/tmp/ed203246': File exists
Sync data to gabriel.intra.cea.fr: 
sending incremental file list
MCIc-CTL/
MCIc-CTL/X.npy
MCIc-CTL/X.txt
MCIc-CTL/config.json
MCIc-CTL/job_Cati_LowPrio.pbs
MCIc-CTL/job_Cati_long.pbs
MCIc-CTL/job_Global_long.pbs
MCIc-CTL/mask.nii
MCIc-CTL/population.csv
MCIc-CTL/sync_pull.sh
MCIc-CTL/sync_push.sh
MCIc-CTL/y.npy

sent 236094730 bytes  received 225 bytes  5427470.23 bytes/sec
total size is 466744296  speedup is 1.98
# Start by running Locally with 2 cores, to check that everything os OK)
Interrupt after a while CTL-C
mapreduce.py --map /neurospin/brainomics/2013_adni/MCIc-CTL/config.json --ncore 2
# 1) Log on gabriel:
ssh -t gabriel.intra.cea.fr
# 2) Run one Job to test
qsub -I
cd /neurospin/tmp/ed203246/MCIc-CTL
./job_Global_long.pbs
# 3) Run on cluster
qsub job_Global_long.pbs
# 4) Log out and pull Pull
exit
/neurospin/brainomics/2013_adni/MCIc-CTL/sync_pull.sh
# Reduce
mapreduce.py --reduce /neurospin/brainomics/2013_adni/MCIc-CTL/config.json
================================================================================

python 02_tvenet_cs.py
user_func /home/ed203246/git/scripts/2013_adni/MCIc-CTL/02_tvenet_cs.py
# Make sure parent dir of wd_cluster exists
ssh ed203246@gabriel.intra.cea.fr "mkdir /neurospin/tmp/ed203246"
mkdir: cannot create directory `/neurospin/tmp/ed203246': File exists
Sync data to gabriel.intra.cea.fr: 
sending incremental file list
MCIc-CTL_cs/
MCIc-CTL_cs/X.npy
MCIc-CTL_cs/X.txt
MCIc-CTL_cs/config.json
MCIc-CTL_cs/job_Cati_LowPrio.pbs
MCIc-CTL_cs/job_Cati_long.pbs
MCIc-CTL_cs/job_Global_long.pbs
MCIc-CTL_cs/mask.nii
MCIc-CTL_cs/sync_pull.sh
MCIc-CTL_cs/sync_push.sh
MCIc-CTL_cs/y.npy

sent 444229696 bytes  received 206 bytes  22781020.62 bytes/sec
total size is 466736155  speedup is 1.05
# Start by running Locally with 2 cores, to check that everything os OK)
Interrupt after a while CTL-C
mapreduce.py --map /neurospin/brainomics/2013_adni/MCIc-CTL_cs/config.json --ncore 2
# 1) Log on gabriel:
ssh -t gabriel.intra.cea.fr
# 2) Run one Job to test
qsub -I
cd /neurospin/tmp/ed203246/MCIc-CTL_cs
./job_Global_long.pbs
# 3) Run on cluster
qsub job_Global_long.pbs
# 4) Log out and pull Pull
exit
/neurospin/brainomics/2013_adni/MCIc-CTL_cs/sync_pull.sh
# Reduce
mapreduce.py --reduce /neurospin/brainomics/2013_adni/MCIc-CTL_cs/config.json

================================================================================
python 02_gtvenet.py
user_func /home/ed203246/git/scripts/2013_adni/MCIc-CTL/02_gtvenet.py
# Make sure parent dir of wd_cluster exists
ssh ed203246@gabriel.intra.cea.fr "mkdir /neurospin/tmp/ed203246"
mkdir: cannot create directory `/neurospin/tmp/ed203246': File exists
Sync data to gabriel.intra.cea.fr: 
sending incremental file list
MCIc-CTL_gtvenet/
MCIc-CTL_gtvenet/X.npy
MCIc-CTL_gtvenet/X.txt
MCIc-CTL_gtvenet/config.json
MCIc-CTL_gtvenet/job_Cati_LowPrio.pbs
MCIc-CTL_gtvenet/job_Cati_long.pbs
MCIc-CTL_gtvenet/job_Global_long.pbs
MCIc-CTL_gtvenet/mask.nii
MCIc-CTL_gtvenet/sync_pull.sh
MCIc-CTL_gtvenet/sync_push.sh
MCIc-CTL_gtvenet/y.npy

sent 236116370 bytes  received 206 bytes  5427967.26 bytes/sec
total size is 466737795  speedup is 1.98
# Start by running Locally with 2 cores, to check that everything os OK)
Interrupt after a while CTL-C
mapreduce.py --map /neurospin/brainomics/2013_adni/MCIc-CTL_gtvenet/config.json --ncore 2
# 1) Log on gabriel:
ssh -t gabriel.intra.cea.fr
# 2) Run one Job to test
qsub -I
cd /neurospin/tmp/ed203246/MCIc-CTL_gtvenet
./job_Global_long.pbs
# 3) Run on cluster
qsub job_Global_long.pbs
# 4) Log out and pull Pull
exit
/neurospin/brainomics/2013_adni/MCIc-CTL_gtvenet/sync_pull.sh
# Reduce
mapreduce.py --reduce /neurospin/brainomics/2013_adni/MCIc-CTL_gtvenet/config.json


================================================================================

python 02_gtvenet_cs.py
user_func /home/ed203246/git/scripts/2013_adni/MCIc-CTL/02_gtvenet_cs.py
# Make sure parent dir of wd_cluster exists
ssh ed203246@gabriel.intra.cea.fr "mkdir /neurospin/tmp/ed203246"
mkdir: cannot create directory `/neurospin/tmp/ed203246': File exists
Sync data to gabriel.intra.cea.fr: 
sending incremental file list
MCIc-CTL_cs_gtvenet/
MCIc-CTL_cs_gtvenet/X.npy
MCIc-CTL_cs_gtvenet/X.txt
MCIc-CTL_cs_gtvenet/config.json
MCIc-CTL_cs_gtvenet/job_Cati_LowPrio.pbs
MCIc-CTL_cs_gtvenet/job_Cati_long.pbs
MCIc-CTL_cs_gtvenet/job_Global_long.pbs
MCIc-CTL_cs_gtvenet/mask.nii
MCIc-CTL_cs_gtvenet/sync_pull.sh
MCIc-CTL_cs_gtvenet/sync_push.sh
MCIc-CTL_cs_gtvenet/y.npy

sent 444252929 bytes  received 206 bytes  25385893.43 bytes/sec
total size is 466736220  speedup is 1.05
# Start by running Locally with 2 cores, to check that everything os OK)
Interrupt after a while CTL-C
mapreduce.py --map /neurospin/brainomics/2013_adni/MCIc-CTL_cs_gtvenet/config.json --ncore 2
# 1) Log on gabriel:
ssh -t gabriel.intra.cea.fr
# 2) Run one Job to test
qsub -I
cd /neurospin/tmp/ed203246/MCIc-CTL_cs_gtvenet
./job_Global_long.pbs
# 3) Run on cluster
qsub job_Global_long.pbs
# 4) Log out and pull Pull
exit
/neurospin/brainomics/2013_adni/MCIc-CTL_cs_gtvenet/sync_pull.sh
# Reduce
mapreduce.py --reduce /neurospin/brainomics/2013_adni/MCIc-CTL_cs_gtvenet/config.json

