Project to study patterns in the white matter hyper-intensities (WMH).

# History

The first batch of results have been presented to MESCOG meeting in January 2014.
They are stored in `/neurospin/mescog/proj_wmh_patterns.2014_01`.
In these experiments we always used the French subpopulation as training and the German one as test.

Another batch of experiments was done with alternatively using the French and Germans as training (and the other one as testing).

The current experiments are done by stratifying on the site (therefore mixing both countries) and with cleaned clinical data and a smoother mask.
We now have only 301 subjects.

# Data

Data are stored in `/neurospin/mescog/proj_wmh_patterns`.
We use the images processed by Marco as input (they are in `/neurospin/mescog/neuroimaging/original/munich/CAD_norm_M0/WMH_norm/`).

We have found that some subjects are very far from the average.
We have tried to detect outliers with the scripts `00_outliers_detection.py` and `00_inspect_outliers.py` but this is no longer used since Marco cleaned them.

# Scripts

* `00_build_dataset.py`: concatenates images and create data set and masks.
  * inputs:
    * images: `/neurospin/mescog/neuroimaging/original/munich/CAD_norm_M0/WMH_norm/*M0-WMH_norm.nii.gz`
    * clinic: `neurospin/mescog/proj_predict_cog_decline/data/dataset_clinic_niglob_20140205_nomissing_BPF-LLV_imputed.csv`
  * outputs:
    * output dir: `/neurospin/mescog/proj_wmh_patterns`
    * population.csv: clinical status of subjects
    * X.npy: concatenation of masked images
    * mask_atlas.nii: mask of WMH (with regions)
    * mask_bin.nii: binary mask of WMH
    * X_center.npy: centerd data
    * means.npy: means used to center

* `01_all_models.py`: compute PCA and PCA struct for many models

* `03_pcs_vs_clinic_mulm_plot.py`: calculate association with clinic

# Results
Findings Fronto-temporo-Inusla patern (protector) vs Deep white matter (worsen)

# TODO

Coronal slices
47 82 113 137

Sagital slices
98 108

PC1 weights map
Test

score ~ PCi + Age + Sex + Education
except for MRS
=> table with 12 lines: 3 PCs x 4 scores

Imaging ~ PCi
=> table with 12 lines: 3 PCs x 4 Imagin (WMH, LLV, MB, BPF)

# Story

Many population of WMH linked with WMH patterns
Identify pattern using spatially regularized PCA

PC1
- association with WMHVolume

PC2
- association with WMHVolume
- association with clinical scores

PC3
- no association with WMHVolume
- Strong associations with clinical scores

 


