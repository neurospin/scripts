################################################################################
# MCIc-MCInc_cs
#
# Start by running Locally with 2 cores, to check that everything os OK)
Interrupt after a while CTL-C
mapreduce.py --map --config /neurospin/brainomics/2013_adni/MCIc-MCInc_cs/config.json --ncore 2
# 1) Log on gabriel:
ssh -t gabriel.intra.cea.fr
# 2) Run one Job to test
qsub -I
cd /neurospin/tmp/ed203246/MCIc-MCInc_cs
./job_Global_long.pbs
# 3) Run on cluster
qsub job_Global_long.pbs
# 4) Log out and pull Pull
exit
/neurospin/brainomics/2013_adni/MCIc-MCInc_cs/sync_pull.sh
# Reduce
mapreduce.py --mode reduce --config /neurospin/brainomics/2013_adni/MCIc-MCInc_cs/config.json

################################################################################
# MCIc-MCInc

# Start by running Locally with 2 cores, to check that everything os OK)
Interrupt after a while CTL-C
mapreduce.py --map /neurospin/brainomics/2013_adni/MCIc-MCInc/config.json --ncore 2
# 1) Log on gabriel:
ssh -t gabriel.intra.cea.fr
# 2) Run one Job to test
qsub -I
cd /neurospin/tmp/ed203246/MCIc-MCInc
./job_Global_long.pbs
# 3) Run on cluster
qsub job_Global_long.pbs
# 4) Log out and pull Pull
exit
/neurospin/brainomics/2013_adni/MCIc-MCInc/sync_pull.sh
# Reduce
mapreduce.py --reduce /neurospin/brainomics/2013_adni/MCIc-MCInc/config.json

################################################################################
# python 02_gtvenet.py
user_func /home/ed203246/git/scripts/2013_adni/MCIc-MCInc_gtvenet/02_gtvenet.py
# Make sure parent dir of wd_cluster exists
ssh ed203246@gabriel.intra.cea.fr "mkdir /neurospin/tmp/ed203246"
mkdir: cannot create directory `/neurospin/tmp/ed203246': File exists
Sync data to gabriel.intra.cea.fr: 
sending incremental file list
MCIc-MCInc_gtvenet/
MCIc-MCInc_gtvenet/X.npy
MCIc-MCInc_gtvenet/X.txt
MCIc-MCInc_gtvenet/config.json
MCIc-MCInc_gtvenet/job_Cati_LowPrio.pbs
MCIc-MCInc_gtvenet/job_Cati_long.pbs
MCIc-MCInc_gtvenet/job_Global_long.pbs
MCIc-MCInc_gtvenet/mask.nii
MCIc-MCInc_gtvenet/sync_pull.sh
MCIc-MCInc_gtvenet/sync_push.sh
MCIc-MCInc_gtvenet/y.npy

sent 190652449 bytes  received 206 bytes  5370497.32 bytes/sec
total size is 377029410  speedup is 1.98
# Start by running Locally with 2 cores, to check that everything os OK)
Interrupt after a while CTL-C
mapreduce.py --map /neurospin/brainomics/2013_adni/MCIc-MCInc_gtvenet/config.json --ncore 2
# 1) Log on gabriel:
ssh -t gabriel.intra.cea.fr
# 2) Run one Job to test
qsub -I
cd /neurospin/tmp/ed203246/MCIc-MCInc_gtvenet
./job_Global_long.pbs
# 3) Run on cluster
qsub job_Global_long.pbs
# 4) Log out and pull Pull
exit
/neurospin/brainomics/2013_adni/MCIc-MCInc_gtvenet/sync_pull.sh
# Reduce
mapreduce.py --reduce /neurospin/brainomics/2013_adni/MCIc-MCInc_gtvenet/config.json


################################################################################
# python 02_gtvenet_cs.py 
user_func /home/ed203246/git/scripts/2013_adni/MCIc-MCInc/02_gtvenet_cs.py
# Make sure parent dir of wd_cluster exists
ssh ed203246@gabriel.intra.cea.fr "mkdir /neurospin/tmp/ed203246"
mkdir: cannot create directory `/neurospin/tmp/ed203246': File exists
Sync data to gabriel.intra.cea.fr: 
sending incremental file list
MCIc-MCInc_cs_gtvenet/
MCIc-MCInc_cs_gtvenet/X.npy
MCIc-MCInc_cs_gtvenet/X.txt
MCIc-MCInc_cs_gtvenet/config.json
MCIc-MCInc_cs_gtvenet/job_Cati_LowPrio.pbs
MCIc-MCInc_cs_gtvenet/job_Cati_long.pbs
MCIc-MCInc_cs_gtvenet/job_Global_long.pbs
MCIc-MCInc_cs_gtvenet/mask.nii
MCIc-MCInc_cs_gtvenet/sync_pull.sh
MCIc-MCInc_cs_gtvenet/sync_push.sh
MCIc-MCInc_cs_gtvenet/y.npy

sent 358076835 bytes  received 206 bytes  23101744.58 bytes/sec
total size is 377028171  speedup is 1.05
# Start by running Locally with 2 cores, to check that everything os OK)
Interrupt after a while CTL-C
mapreduce.py --map /neurospin/brainomics/2013_adni/MCIc-MCInc_cs_gtvenet/config.json --ncore 2
# 1) Log on gabriel:
ssh -t gabriel.intra.cea.fr
# 2) Run one Job to test
qsub -I
cd /neurospin/tmp/ed203246/MCIc-MCInc_cs_gtvenet
./job_Global_long.pbs
# 3) Run on cluster
qsub job_Global_long.pbs
# 4) Log out and pull Pull
exit
/neurospin/brainomics/2013_adni/MCIc-MCInc_cs_gtvenet/sync_pull.sh
# Reduce
mapreduce.py --reduce /neurospin/brainomics/2013_adni/MCIc-MCInc_cs_gtvenet/config.json

