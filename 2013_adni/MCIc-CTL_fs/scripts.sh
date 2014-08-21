user_func /home/ed203246/git/scripts/2013_adni/MCIc-CTL_fs/03_gtvenet_cs.py

mapreduce.py --map /neurospin/brainomics/2013_adni/MCIc-CTL_fs/config_5cv.json --ncore 2
# 1) Log on gabriel:
ssh -t gabriel.intra.cea.fr
# 2) Run one Job to test
qsub -I
cd /neurospin/tmp/ed203246/MCIc-CTL_fs
./job_Global_long.pbs
# 3) Run on cluster
qsub job_Global_long.pbs
# 4) Log out and pull Pull
exit
/neurospin/brainomics/2013_adni/MCIc-CTL_fs/sync_pull.sh
# Reduce
mapreduce.py --reduce /neurospin/brainomics/2013_adni/MCIc-CTL_fs/config_5cv.json

