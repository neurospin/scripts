################################################################################
# MCIc-MCInc
#
# Start by running Locally with 2 cores, to check that everything os OK)
#Interrupt after a while CTL-C
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
# MCIc-MCInc_cs
#
# Start by running Locally with 2 cores, to check that everything os OK)
#Interrupt after a while CTL-C
mapreduce.py --map /neurospin/brainomics/2013_adni/MCIc-MCInc_cs/config.json --ncore 2
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
mapreduce.py --reduce /neurospin/brainomics/2013_adni/MCIc-MCInc_cs/config.json


################################################################################
# MCIc-MCInc_gtvenet
#
#python 02_gtvenet.py
user_func /home/ed203246/git/scripts/2013_adni/MCIc-MCInc_gtvenet/02_gtvenet.py
# Start by running Locally with 2 cores, to check that everything os OK)
# Interrupt after a while CTL-C
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
# MCIc-MCInc_cs_gtvenet
#
# python 02_gtvenet_cs.py 
user_func /home/ed203246/git/scripts/2013_adni/MCIc-MCInc/02_gtvenet_cs.py
# Make sure parent dir of wd_cluster exists
# Start by running Locally with 2 cores, to check that everything os OK)
# Interrupt after a while CTL-C
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

