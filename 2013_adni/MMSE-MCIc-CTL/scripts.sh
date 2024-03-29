user_func /home/ed203246/git/scripts/2013_adni/MMSE-MCIc-CTL/02_tvenet_cs.py

# Start by running Locally with 2 cores, to check that everything os OK)
#Interrupt after a while CTL-C
mapreduce.py --map /neurospin/brainomics/2013_adni/MMSE-MCIc-CTL/config.json --ncore 2
# 1) Log on gabriel:
ssh -t gabriel.intra.cea.fr
# 2) Run one Job to test
qsub -I
cd /neurospin/tmp/ed203246/MMSE-MCIc-CTL
./job_Global_long.pbs
# 3) Run on cluster
qsub job_Global_long.pbs
# 4) Log out and pull Pull
exit
/neurospin/brainomics/2013_adni/MMSE-MCIc-CTL/sync_pull.sh
# Reduce
mapreduce.py --reduce /neurospin/brainomics/2013_adni/MMSE-MCIc-CTL/config.json

