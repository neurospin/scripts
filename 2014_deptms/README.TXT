Population file
===============

/neurospin/brainomics/2014_deptms/clinic/deprimPetInfo.csv

Target:
Response

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

y ~ mri_roiho-hippo
y ~ mri_roiho-amyg
y ~ both_roiho-amyg
...
y ~ pet_roiho-hippo
y ~ pet_roiho-amyg
y ~ both_roiho-amyg


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


ROIs
================================================
/neurospin/brainomics/2014_deptms/ROI_labels.csv

ROIs				AAL label		Harvard Oxford label
						
Thalamus Right	???		78			15	(subcortical atlas)
Thalamus Left	???		77			04	(subcortical atlas)
Frontal_Mid_Orb_L		9			1	(cortical atlas)
Frontal_Mid_Orb_R		10			1	(cortical atlas)
Frontal_Inf_Orb_L		15			33	(cortical atlas)
Frontal_Inf_Orb_R		16			33	(cortical atlas)
Rectus_L			27
Rectus_R			28
Frontal_Mid_R			8			4	(cortical atlas)
Frontal_Mid_L			7			4	(cortical atlas)
Cingulum_Ant_L			31			29	(cortical atlas)
Cingulum_Ant_R			32			29	(cortical atals)
Hippocampus_L			37			09	(subcortical atlas)
Hippocampus_R			38			19	(subcortical atlas)
Amygdala_L			41			10	(subcortical atlas)
Amygdala_R			42			20	(subcortical atlas)
Caudate_L			71			5	(subcortical atlas)
Caudate_R			72			16	(subcortical atlas)
Putamen_L			73			6	(subcortical atlas)
Putamen_R			74			17	(subcortical atlas)
Frontal_Sup_Medial_L		23			3	(cortical atlas)
Frontal_Sup_Medial_R		24			3	(cortical atlas)
Frontal_Med_Orb_L		25			25	(cortical atlas)
Frontal_Med_Orb_R		26			25	(cortical atlas)
Frontal_Sup_L			3			3	(cortical atlas)
Frontal_Sup_R			4			3	(cortical atlas)
Insula_L			29			2	(cortical atlas)	
Insula_R			30			2	(cortical atlas)
Olfactory_L			21
Olfactory_R		 	22


AAL in
/neurospin/brainomics/neuroimaging_ressources/atlases/WFU_PickAtlas_3.0.3/wfu_pickatlas/MNI_atlas_templates

fslview WFU_PickAtlas_3.0.3/wfu_pickatlas/MNI_atlas_templates/
gedit WFU_PickAtlas_3.0.3/wfu_pickatlas/MNI_atlas_templates/aal_MNI_V4.txt
