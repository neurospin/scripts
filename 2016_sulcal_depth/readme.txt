# NOTE:
In all of these file directory refering to "/volatile/yann/" should be replaced by "/neurospin/brainomics/"


This directory should contains the following scripts:

create_grm.py 
creates the GRM matrix from SNPs of subjects who passed QC genetics ~ 1830 subjects
need to input the following parameters used as thresholds:
maf (minor allele frequency), hwe (Hardy-Weinberg equilibrium), genorate (genotyping rate), vif (variance inflation factor) with its window size w_size.

qc_subjects_sulci.py 
performs the quality control on the subjects and sulci
It excludes sulci not recognize in at least 25% as well as subjects with less than 25% sulci recognized.
In addition, it excludes subject who have less than % tolerance_threshold of their sulci features within mu+/-sigma.
There are 6 sulci features per sulci on each side.

create_covar_MEGHA.py 
creates two covar files in the MEGHA format
first one with the covariates: Gender, Cities, 5 first PCA of the Identity by state matrix and the ICV (from the eTIV of Freesurfer)
second one same covariate and adding Handedness (loosing 198 subjects in IMAGEN because of missing data).
Note it takes a long time ~2-5min

create_sulcus_files.py 
Collect the information output of Morphologist for each subject and assemble it in one file per sulci for each side.
Note it takes a long time ~10min

create_pheno.py 
performs the QC using the function available in qc_subjects_sulci.py
Then, it extracts the feature of interest (ex: the depthMax) from the individual sulcus.
Producing phenotype in the output format for PLINK

concatenate_pheno.py 
concatenates all the phenotypes of interest in one file to run in one time using MEGHA for a fix GRM and covar.


megha/
this directory contains matlab files to run megha see readme inside to know how to use it

analyse_megha_ouput.py
output in the shell the results of a *MEGHAstat.txt (for a MEGHA.m run, not for MEGHASurf.m)
It is uses to analyse the heritability result when working on a feature for each sulci
outputs by hemisphere LEFT, RIGHT and ASYMMETRY

map_maxdepth_sulci.py
maps the heritability or pvalue given by MEGHA.m on each sulci using PyAnatomist

map_sulcaldepth_clusters.py
maps the significant cluster given by MEGHASurf.m using PyAnatomist
#NOTE:
to map the heritability or pvalue use the example freeview command available inside at the end
such as:
freeview -f /neurospin/imagen/BL/processed/freesurfer/fsaverage/surf/lh.inflated:overlay=/volatile/yann/megha/cluster_1000perm_covar_done/covar_GenCit5PCA_ICV_MEGHALogPvalLh.mgh

draw_histogram.py
plots the histograms of the depthMax for a particular sulci
possibility to distinguish gender or handedness
