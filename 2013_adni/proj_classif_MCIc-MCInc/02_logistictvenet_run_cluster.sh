# Build config file
python $HOME/git/scripts/2013_adni/proj_classif_MCIc-MCInc/02_logistictvenet.py

# 1) Log on gabriel:
ssh -t gabriel.intra.cea.fr
# 2) Run one Job to test
qsub -I
cd /neurospin/tmp/brainomics/2013_adni/proj_classif_MCIc-MCInc/logistictvenet_5cv
./job_Global_long.pbs
# 3) Run on cluster
qsub job_Global_long.pbs
find results -name "*_run*"
find results -name "*_lock" | while read f ; do rm -f $f ; done
find results -name beta.npy | while read f ; do gzip $f ; done

# 4) Log out and pull Pull
exit
/neurospin/brainomics/2013_adni/proj_classif_MCIc-MCInc/logistictvenet_5cv/sync_pull.sh
# Reduce
mapreduce.py --mode reduce --config /neurospin/brainomics/2013_adni/proj_classif_MCIc-MCInc/logistictvenet_5cv/config.json

















################################################################################
# Start by running Locally with 2 cores, to check that everything os OK)
Interrupt after a while CTL-C
mapreduce.py --mode map --config /neurospin/brainomics/2013_adni/proj_classif_MCIc-MCInc/logistictvenet_5cv/config.json --ncore 2
# 1) Log on gabriel:
ssh -t gabriel.intra.cea.fr
# 2) Run the jobs
#    Run one Job to test
qsub -I
cd /neurospin/tmp/brainomics/2013_adni/proj_classif_MCIc-MCInc/logistictvenet_5cv
./job_Cati_LowPrio.pbs
# Reduce
mapreduce.py --mode reduce --config /neurospin/brainomics/2013_adni/proj_classif_MCIc-MCInc/logistictvenet_5cv/config.json


# Start by running Locally with 2 cores, to check that everything os OK) Interrupt after a while CTL-C
#mapreduce.py --mode map --config /neurospin/tmp/brainomics/2013_adni/proj_classif_MCIc-MCInc/logistictvenet_5cv/config.json --ncore 2
# Run on the cluster with 30 PBS Jobs
#mapreduce.py --pbs_job --config /neurospin/tmp/brainomics/2013_adni/proj_classif_MCIc-MCInc/logistictvenet_5cv/config.json


# 1) Push your file to gabriel, run:
#/neurospin/tmp/brainomics/2013_adni/proj_classif_MCIc-MCInc/logistictvenet_5cv/sync_push.sh
# 2) Log on gabriel:
ssh -t gabriel.intra.cea.fr "cd /neurospin/tmp/brainomics/2013_adni/proj_classif_MCIc-MCInc/logistictvenet_5cv ; bash"
# 3) Run the jobs
# 4) Run one Job to test
qsub -I
cd /neurospin/tmp/brainomics/2013_adni/proj_classif_MCIc-MCInc/logistictvenet_5cv
./job_Cati_LowPrio.pbs
# Interrupt afetr a while CTL-C
exit
# Execute via qsub, run many time 
qsub job_Cati_LowPrio.pbs
# or
qsub job_Global_long.pbs
exit
find results -name "*_run*" | while read f ; do rm -f $f ; done
find results -name "*_lock" | while read f ; do rm -f $f ; done
find results -name beta.npy | while read f ; do gzip $f ; done

# 5) Pull your file from gabriel, run
/neurospin/tmp/brainomics/2013_adni/proj_classif_MCIc-MCInc/logistictvenet_5cv/sync_pull.sh


# Reduce
mapreduce.py --mode reduce --config /neurospin/tmp/brainomics/2013_adni/proj_classif_MCIc-MCInc/logistictvenet_5cv/config.json


cd /neurospin/tmp/brainomics/2013_adni/proj_classif_MCIc-MCInc/logistictvenet_5cv
# tar gz ones


rsync -avu /neurospin/tmp/brainomics/2013_adni/proj_classif_MCIc-MCInc/logistictvenet_5cv /neurospin/brainomics/2013_adni/proj_classif_MCIc-MCInc/

cd /neurospin/brainomics/2013_adni/proj_classif_MCIc-MCInc/logistictvenet_5cv/
