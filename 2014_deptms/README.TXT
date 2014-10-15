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


Resample harvard-oxford into study referential
==============================================
AimsFileInfo mask.nii
cd /neurospin/brainomics/2014_deptms/mri_wb

ls ../images/MRI_images/smwc1*.img|while read f ; do AimsFileInfo $f | grep volume_dimension; done
ls ../images/MRI_images/smwc1*.img|while read f ; do AimsFileInfo $f | grep transformations; done
ls ../images/MRI_images/smwc1*.img|while read f ; do AimsFileInfo $f | grep voxel_size; done

ls ../images/PET_j0Scaled_images/smw*.img |while read f ; do AimsFileInfo $f|grep volume_dimension ; done
ls ../images/PET_j0Scaled_images/smw*.img |while read f ; do AimsFileInfo $f|grep transformations ; done
ls ../images/PET_j0Scaled_images/smw*.img |while read f ; do AimsFileInfo $f|grep voxel_size ; done


All are:
    'volume_dimension' : [ 91, 109, 91 ],
    'transformations' : [ [ -1, 0, 0, 90, 0, -1, 0, 90, 0, 0, -1, 108, 0, 0, 0, 1 ], [ -1, 0, 0, 90, 0, -1, 0, 90, 0, 0, -1, 108, 0, 0, 0, 1 ] ],
    'voxel_size' : [ 2, 2, 2 ],



fsl5.0-applywarp -i /usr/share/data/harvard-oxford-atlases/HarvardOxford/HarvardOxford-sub-maxprob-thr0-1mm.nii.gz -r /neurospin/brainomics/2014_deptms/mri_wb/mask.nii -o /neurospin/brainomics/2014_deptms/images/atlases/HarvardOxford-sub-maxprob-thr0-1mm.nii.gz

AimsFileInfo mask.nii 

AimsFileInfo /neurospin/brainomics/2014_deptms/images/atlases/HarvardOxford-sub-maxprob-thr0-1mm.nii.gz

fslview /neurospin/brainomics/2014_deptms/images/atlases/HarvardOxford-sub-maxprob-thr0-1mm.nii.gz t_stat_rep_min_norep_mri_wb.nii.gz pval_rep_min_norep_mri_wb.nii.gz
=> Same referential


