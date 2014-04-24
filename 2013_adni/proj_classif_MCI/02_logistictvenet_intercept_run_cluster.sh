# Build config file
python $HOME/git/scripts/2013_adni/proj_classif_MCI/02_logistictvenet_intercept.py

# Start by running Locally with 2 cores, to check that everything os OK)
mapreduce.py --mode map --config /neurospin/tmp/brainomics/2013_adni/proj_classif_MCI/logistictvenet_intercept_5cv/config.json --ncore 2
# Run on the cluster with 30 PBS Jobs
mapreduce.py --pbs_njob 30 --config /neurospin/tmp/brainomics/2013_adni/proj_classif_MCI/logistictvenet_intercept_5cv/config.json
## 1) Push your file to gabriel, run:
/neurospin/tmp/brainomics/2013_adni/proj_classif_MCI/logistictvenet_intercept_5cv/sync_push.sh
# 2) Log on gabriel:
ssh -t gabriel.intra.cea.fr "cd /neurospin/tmp/brainomics/2013_adni/proj_classif_MCI/logistictvenet_intercept_5cv ; bash"
# 3) Run the jobs
# 4) Run one Job to test
qsub -I
cd /neurospin/tmp/brainomics/2013_adni/proj_classif_MCI/logistictvenet_intercept_5cv
./job.pbs
# Interrupt afetr a while CTL-C
exit
/neurospin/tmp/brainomics/2013_adni/proj_classif_MCI/logistictvenet_intercept_5cv/jobs_all.sh
exit
# 5) Pull your file from gabriel, run
/neurospin/tmp/brainomics/2013_adni/proj_classif_MCI/logistictvenet_intercept_5cv/sync_pull.sh

# Reduce
mapreduce.py --mode reduce --config /neurospin/tmp/brainomics/2013_adni/proj_classif_MCI/logistictvenet_intercept_5cv/config.json
cd /neurospin/tmp/brainomics/2013_adni/proj_classif_MCI/logistictvenet_intercept_5cv/


find results -name model.pkl | while read f ; do rm -f $f ; done
find results -name beta3d.nii | while read f ; do gzip $f ; done
#rm -rf results
rsync -avu /neurospin/tmp/brainomics/2013_adni/proj_classif_MCI/logistictvenet_intercept_5cv /neurospin/brainomics/2013_adni/proj_classif_MCI/
