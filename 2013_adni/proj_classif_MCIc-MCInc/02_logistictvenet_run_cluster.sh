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
# 2 CV 
python $HOME/git/scripts/2013_adni/proj_classif_MCIc-MCInc/02_logistictvenet_2cv.py

mapreduce.py --mode map --config /neurospin/brainomics/2013_adni/proj_classif_MCIc-MCInc/logistictvenet_2cv/config.json --ncore 2
# 1) Log on gabriel:
ssh -t gabriel.intra.cea.fr
# 2) Run one Job to test
qsub -I
cd /neurospin/tmp/brainomics/2013_adni/proj_classif_MCIc-MCInc/logistictvenet_2cv
./job_Global_long.pbs
# 3) Run on cluster
qsub job_Global_long.pbs
# 4) Log out and pull Pull
exit
/neurospin/brainomics/2013_adni/proj_classif_MCIc-MCInc/logistictvenet_2cv/sync_pull.sh
# Reduce
mapreduce.py --mode reduce --config /neurospin/brainomics/2013_adni/proj_classif_MCIc-MCInc/logistictvenet_2cv/config.json

