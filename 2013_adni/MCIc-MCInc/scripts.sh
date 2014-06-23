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

