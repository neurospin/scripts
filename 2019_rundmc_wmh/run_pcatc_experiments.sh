jobs files in :
/neurospin/brainomics/2019_rundmc_wmh/analyses/201909_rundmc_wmh_pca/jobs

-------------------
job_Global_long.pbs

#!/bin/bash
#PBS -S /bin/bash
#PBS -N enettv_model_selection_5folds_all_subjects
#PBS -l nodes=1:ppn=1
#PBS -l walltime=250:00:00
#PBS -q Global_long

python /home/ed203246/git/scripts/2019_rundmc_wmh/pca_pcatv.py --algo pca --penalties -1 -1 -1
-------------------


# PCA (98.4%, 0.26%)
python /home/ed203246/git/scripts/2019_rundmc_wmh/pca_pcatv.py --algo pca --penalties -1 -1 -1

# =======================================================================
# low l1
python /home/ed203246/git/scripts/2019_rundmc_wmh/pca_pcatv.py --algo enettv --penalties 0.000035 1 0.001

python /home/ed203246/git/scripts/2019_rundmc_wmh/pca_pcatv.py --algo enettv --penalties 0.000035 1 0.005

python /home/ed203246/git/scripts/2019_rundmc_wmh/pca_pcatv.py --algo enettv --penalties 0.000035 1 0.01

python /home/ed203246/git/scripts/2019_rundmc_wmh/pca_pcatv.py --algo enettv --penalties 0.000035 1 0.1

python /home/ed203246/git/scripts/2019_rundmc_wmh/pca_pcatv.py --algo enettv --penalties 0.000035 1 0.5

python /home/ed203246/git/scripts/2019_rundmc_wmh/pca_pcatv.py --algo enettv --penalties 0.000035 1 1



# ======================================================================= Triscotte
# medium l1
python /home/ed203246/git/scripts/2019_rundmc_wmh/pca_pcatv.py --algo enettv --penalties 0.00035 1 0.001

python /home/ed203246/git/scripts/2019_rundmc_wmh/pca_pcatv.py --algo enettv --penalties 0.00035 1 0.01

python /home/ed203246/git/scripts/2019_rundmc_wmh/pca_pcatv.py --algo enettv --penalties 0.00035 1 0.1

python /home/ed203246/git/scripts/2019_rundmc_wmh/pca_pcatv.py --algo enettv --penalties 0.00035 1 0.5

python /home/ed203246/git/scripts/2019_rundmc_wmh/pca_pcatv.py --algo enettv --penalties 0.00035 1 1

# ======================================================================= is152871
# high l1
python /home/ed203246/git/scripts/2019_rundmc_wmh/pca_pcatv.py --algo enettv --penalties 0.0035 1 0.001

python /home/ed203246/git/scripts/2019_rundmc_wmh/pca_pcatv.py --algo enettv --penalties 0.0035 1 0.01

# ======================================================================= local

python /home/ed203246/git/scripts/2019_rundmc_wmh/pca_pcatv.py --algo enettv --penalties 0.0035 1 0.1

python /home/ed203246/git/scripts/2019_rundmc_wmh/pca_pcatv.py --algo enettv --penalties 0.0035 1 0.5

python /home/ed203246/git/scripts/2019_rundmc_wmh/pca_pcatv.py --algo enettv --penalties 0.0035 1 1


# ======================================================================= ssh $NS_AMICIE
# max l1
python /home/ed203246/git/scripts/2019_rundmc_wmh/pca_pcatv.py --algo enettv --penalties 0.035 1 0.001

python /home/ed203246/git/scripts/2019_rundmc_wmh/pca_pcatv.py --algo enettv --penalties 0.035 1 0.01

python /home/ed203246/git/scripts/2019_rundmc_wmh/pca_pcatv.py --algo enettv --penalties 0.035 1 0.1

python /home/ed203246/git/scripts/2019_rundmc_wmh/pca_pcatv.py --algo enettv --penalties 0.035 1 0.5

python /home/ed203246/git/scripts/2019_rundmc_wmh/pca_pcatv.py --algo enettv --penalties 0.035 1 1

# =======================================================================
# like PCA
python /home/ed203246/git/scripts/2019_rundmc_wmh/pca_pcatv.py --algo enettv --penalties 0.000001 1 0.000001 # Amicie

