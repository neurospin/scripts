Population file
===============

/neurospin/brainomics/2014_deptms/clinic/deprimPetInfo.csv

Target:
rep_norep

Images:
- PET_file
- MRIm_G_file

Covariables:
- Sex
- Age

Images
======
/neurospin/brainomics/2014_deptms/images

MRI_images *
MRI_rois
PET_j0_images
PET_j0Scaled_images *
PET_rois

Working plan
============
Use "git/scripts/2013_adni/MCIc-CTL/" ad model
- 00_create_population.py => deprimPetInfo.csv already done
- 01_build_dataset.py


Do like "MCIc-CTL_csi": center/scale X + add intercept
Do not suffix with _csi
y ~ mri_wb
y ~ pet_wb
y ~ both_wb

y ~ mri_roiaal-hippo
y ~ mri_roiaal-amyg
y ~ both_roiaal-amyg
...
y ~ pet_roiaal-hippo
y ~ pet_roiaal-amyg
y ~ both_roiaal-amyg





