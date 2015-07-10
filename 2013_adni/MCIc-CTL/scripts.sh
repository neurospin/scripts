python 02_tvenet.py
user_func /home/ed203246/git/scripts/2013_adni/MCIc-CTL/02_tvenet.py

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

================================================================================
user_func /home/ed203246/git/scripts/2013_adni/MCIc-CTL/02_tvenet_csi.py

# Start by running Locally with 2 cores, to check that everything os OK)
Interrupt after a while CTL-C
mapreduce.py --map /neurospin/brainomics/2013_adni/MCIc-CTL_csi/config.json --ncore 2
# 1) Log on gabriel:
ssh -t gabriel.intra.cea.fr
# 2) Run one Job to test
qsub -I
cd /neurospin/tmp/ed203246/MCIc-CTL_csi
./job_Global_long.pbs
# 3) Run on cluster
qsub job_Global_long.pbs
# 4) Log out and pull Pull
exit
/neurospin/brainomics/2013_adni/MCIc-CTL_csi/sync_pull.sh
# Reduce
mapreduce.py --reduce /neurospin/brainomics/2013_adni/MCIc-CTL_csi/config.json

================================================================================
user_func /home/ed203246/git/scripts/2013_adni/MCIc-CTL/03_rndperm_tvenet_csi.py
# Start by running Locally with 2 cores, to check that everything os OK)
Interrupt after a while CTL-C
mapreduce.py --map /neurospin/brainomics/2013_adni/MCIc-CTL_csi/config_rndperm.json --ncore 2
# 1) Log on gabriel:
ssh -t gabriel.intra.cea.fr
# 2) Run one Job to test
qsub -I
cd /neurospin/tmp/ed203246/MCIc-CTL_csi
./job_Global_long.pbs
# 3) Run on cluster
qsub job_Global_long.pbs
# 4) Log out and pull Pull
exit
/neurospin/brainomics/2013_adni/MCIc-CTL_csi/sync_pull.sh
# Reduce
mapreduce.py --reduce /neurospin/brainomics/2013_adni/MCIc-CTL_csi/config_rndperm.json



================================================================================
# Make sure parent dir of wd_cluster exists
ssh ed203246@gabriel.intra.cea.fr "mkdir /neurospin/tmp/ed203246"
mkdir: cannot create directory `/neurospin/tmp/ed203246': File exists
Sync data to gabriel.intra.cea.fr: 
sending incremental file list
MCIc-CTL_csnc/
MCIc-CTL_csnc/X.npy
MCIc-CTL_csnc/X.txt
MCIc-CTL_csnc/config_5cv.json
MCIc-CTL_csnc/job_Cati_LowPrio.pbs
MCIc-CTL_csnc/job_Cati_long.pbs
MCIc-CTL_csnc/job_Global_long.pbs
MCIc-CTL_csnc/mask.nii
MCIc-CTL_csnc/sync_pull.sh
MCIc-CTL_csnc/sync_push.sh
MCIc-CTL_csnc/y.npy

sent 214,194,436 bytes  received 213 bytes  20,399,490.38 bytes/sec
total size is 235,444,949  speedup is 1.10
# Start by running Locally with 2 cores, to check that everything os OK)
Interrupt after a while CTL-C
mapreduce.py --map /neurospin/brainomics/2013_adni/MCIc-CTL_csnc/config_5cv.json --ncore 2
# 1) Log on gabriel:
ssh -t gabriel.intra.cea.fr
# 2) Run one Job to test
qsub -I
cd /neurospin/tmp/ed203246/MCIc-CTL_csnc
./job_Global_long.pbs
# 3) Run on cluster
qsub job_Global_long.pbs
# 4) Log out and pull Pull
exit
/neurospin/brainomics/2013_adni/MCIc-CTL_csnc/sync_pull.sh
# Reduce
mapreduce.py --reduce /neurospin/brainomics/2013_adni/MCIc-CTL_csnc/config.json


================================================================================
user_func /home/ed203246/git/scripts/2013_adni/MCIc-CTL/02_tvenet_csnc_s.py
# Make sure parent dir of wd_cluster exists
ssh ed203246@gabriel.intra.cea.fr "mkdir /neurospin/tmp/ed203246"
mkdir: cannot create directory `/neurospin/tmp/ed203246': File exists
Sync data to gabriel.intra.cea.fr: 
sending incremental file list
MCIc-CTL_csnc_s/
MCIc-CTL_csnc_s/X.npy
MCIc-CTL_csnc_s/X.txt
MCIc-CTL_csnc_s/config_5cv.json
MCIc-CTL_csnc_s/job_Cati_LowPrio.pbs
MCIc-CTL_csnc_s/job_Cati_long.pbs
MCIc-CTL_csnc_s/job_Global_long.pbs
MCIc-CTL_csnc_s/mask.nii
MCIc-CTL_csnc_s/sync_pull.sh
MCIc-CTL_csnc_s/sync_push.sh
MCIc-CTL_csnc_s/y.npy

sent 214,120,551 bytes  received 213 bytes  22,539,027.79 bytes/sec
total size is 235,444,999  speedup is 1.10
# Start by running Locally with 2 cores, to check that everything os OK)
Interrupt after a while CTL-C
mapreduce.py --map /neurospin/brainomics/2013_adni/MCIc-CTL_csnc_s/config_5cv.json --ncore 2
# 1) Log on gabriel:
ssh -t gabriel.intra.cea.fr
# 2) Run one Job to test
qsub -I
cd /neurospin/tmp/ed203246/MCIc-CTL_csnc_s
./job_Global_long.pbs
# 3) Run on cluster
qsub job_Global_long.pbs
# 4) Log out and pull Pull
exit
/neurospin/brainomics/2013_adni/MCIc-CTL_csnc_s/sync_pull.sh
# Reduce
mapreduce.py --reduce /neurospin/brainomics/2013_adni/MCIc-CTL_csnc_s/config.json


################################################################################
user_func /home/ed203246/git/scripts/2013_adni/MCIc-CTL/02_tvenet_modselectcv_csi.py

# Start by running Locally with 2 cores, to check that everything os OK)
Interrupt after a while CTL-C
mapreduce.py --map /neurospin/brainomics/2013_adni/MCIc-CTL_csi_modselectcv/config_modselectcv.json --ncore 2
# 1) Log on gabriel:
ssh -t gabriel.intra.cea.fr
# 2) Run one Job to test
qsub -I
cd /neurospin/tmp/ed203246/MCIc-CTL_csi_modselectcv
./job_Global_long.pbs
# 3) Run on cluster
qsub job_Global_long.pbs
# 4) Log out and pull Pull
exit
/neurospin/brainomics/2013_adni/MCIc-CTL_csi_modselectcv/sync_pull.sh
# Reduce
mapreduce.py --reduce /neurospin/brainomics/2013_adni/MCIc-CTL_csi_modselectcv/config_modselectcv.json


