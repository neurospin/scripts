=============================================================================================================================
Brain Imaging Predictors of Symptom Improvment  of Unipolar Depression atfer Transcranial Magnetic Simulation (TMS) Treatment
=============================================================================================================================

:Author: Clemence.Pinaud@cea.fr
:Date: 2014/10/24
:Build: ``rst2pdf deptms.rst``

Prediction of Response to TMS Treatment for pharmacoresistant patients with unipolar depression

.. contents::

Data description
=================

Longitudinal study on thirty-four patients with unipolar depression (shfj cohort):
- MRI and PET before TMS Treatment
- After treatment: Response to Treatment (rep/no rep)
Around 40% of the patients that receive the TMS respond to the treatement.

Based on MRI and PET images, we want to find brain markers to predict the response to TMS treatment.

BASE_PATH = /neurospin/brainomics/2014_deptms
SCRIPT_PATH = ~/gits/scripts/2014_deptms

1. **Clinic Data**
::

	/neurospin/brainomics/2014_deptms/base_data
		clinic/deprimPetInfo.csv

- rep_norep: response to the treatment
- Images:
	- PET_file (modulated and scaled PET) 
	- MRIm_G_file	(modulated MRI T1) 

- Covariables: 
	- Sex 
	- Age 

2. **Images**
::

	/neurospin/brainomics/2014_deptms/base_data
		# MRI images (modulated MRI T1 images):
		MRI_images/smwc1*.img
		# PET images (modulated and scaled PET images):
		PET_j0Scaled_images/smw*.img  


ROIs definition and mapping
===========================

Study for 3 modalities:
	* MRI
	* PET
	* MRI+PET

Study for the whole brain and for a set of ROIs

UNI lab defined a list of Regions Of Interest (ROI) to constitue the depmask for the response to TMS treatment.

ROIs have been defined from the Automated Anatomical Labeling atlas (AAL).

These ROIs then need to be mapped to the Harvard Oxford atlas that will be used in the study.

**ROIs**

===================================	     ===================================
AAL Names                                    Harvard Oxford Names
===================================	     ===================================
Frontal_Mid_Orb_L			     Frontal Pole
Frontal_Mid_Orb_R			     Frontal Pole
Frontal_Inf_Orb_L			     Frontal Orbital Cortex
Frontal_Inf_Orb_R			     Frontal Orbital Cortex
Rectus_L				     -
Rectus_R				     -
Frontal_Mid_R				     Middle Frontal Gyrus
Frontal_Mid_L				     Middle Frontal Gyrus
Cingulum_Ant_L				     Cingulate Gyrus anterior division
Cingulum_Ant_R				     Cingulate Gyrus anterior division
Hippocampus_L				     Left Hippocampus
Hippocampus_R				     Right Hippocampus
Amygdala_L				     Left Amygdala
Amygdala_R				     Right Amygdala
Caudate_L				     Left Caudate
Caudate_R				     Right Caudate
Putamen_L				     Left Putamen
Putamen_R				     Right Putamen
Frontal_Sup_Medial_L			     Superior Frontal Gyrus
Frontal_Sup_Medial_R			     Superior Frontal Gyrus
Frontal_Med_Orb_L			     Frontal Medial Cortex
Frontal_Med_Orb_R			     Frontal Medial Cortex
Frontal_Sup_L				     Superior Frontal Gyrus
Frontal_Sup_R				     Superior Frontal Gyrus
Insula_L				     Insular Cortex
Insula_R				     Insular Cortex
Olfactory_L				     -
Olfactory_R				     -
===================================	     ===================================

**OUTPUT**: ROIs mapping between AAl atlas and Harvard Oxford atlas
::

	/neurospin/brainomics/2014_deptms/base_data	
		ROI_labels.csv

Build Dataset
=============

1) Datasets associated to MRI images and to PET images for the whole brain and for each ROI

**Script**
::

	01_build_dataset.py

Read the data (clinic data, ROI, MRI AND PET images).

Construct an implicit mask associated to the whole brain. Since PET ans MRI images have exactly the same caracteristics (e.g: same size), the implicit mask is the same for both MRI and PET images so we constructed it from MRI images.

For each ROI construct a specific mask defining the region using harvard oxford atlases and dilate the obtained mask to make sure that the entire region is contained in the mask ).

Construct the matrix X and y for the regression. X is constructed for each pair (modality, ROI). Each row of the X matrix contains and Intercept, the age, the sex and the image of the patient. The matrix X is then centered ans scaled.

For the modality MRI+PET, implicit masks, matrices X and Y are obtained by concatenating MRI and PET masks and matrices.

**INPUTS**: clinic data, ROIs labels, atlases
::

	/neurospin/brainomics/2014_deptms/base_data
		# Clinic data
		clinic/deprimPetInfo.csv
		# ROIs
		ROI_labels.csv
		# Resampled cortical and subcortical harvard oxford atlases:
    		images/atlases/
			HarvardOxford-sub-maxprob-thr0-1mm-nn.nii.gz
        		HarvardOxford-cort-maxprob-thr0-1mm-nn.nii.gz
		# MRI images :
    		images/MRI_images/smwc1*.img
		# PET images :
    		images/PET_j0Scaled_images/smw*.img

**OUTPUTS**: masks, X, y associated to MRI and PET images
MODALITY: {MRI, PET} 
::

	/neurospin/brainomics/2014_deptms/datasets
		# outputs for each modality
		*{MRI, PET, MRI+PET}/
			# implicit mask for the whole brain and mask for each ROI
			mask_*_wb.nii
	    		mask_*_Roiho-amyg.nii
			mask_*_Roiho-caudate.nii
			mask_*_Roiho-cingulumAnt.nii
			mask_*_Roiho-frontalOrb.nii
			mask_*_Roiho-frontalPole.nii
			mask_*_Roiho-hippo.nii
			mask_*_Roiho-insula.nii
			mask_*_Roiho-medFrontal.nii
			mask_*_Roiho-midFrontal.nii
			mask_*_Roiho-putamen.nii
			mask_*_Roiho-supFrontal.nii
			# X for the whole brain and for each ROI
			  (Intercept + Age + Sex + images)
	    		X_*_wb.npy
			X_*_Roiho-amyg.npy
			X_*_Roiho-caudate.npy
			X_*_Roiho-cingulumAnt.npy
			X_*_Roiho-frontalOrb.npy
			X_*_Roiho-frontalPole.npy
			X_*_Roiho-hippo.npy
			X_*_Roiho-insula.npy
			X_*_Roiho-medFrontal.npy
			X_*_Roiho-midFrontal.npy
			X_*_Roiho-putamen.npy
			X_*_Roiho-supFrontal.npy
	 		# Response to the treatment y
			y.npy

Univariate Analysis
====================

Methods
-------

Univariate analysis between brain images and the response to the treatment for the whole brain and for each modality.

First, X and y are fitted and then statistic coefficients are evaluated at each voxel for the constrast [1 0 0 0] (rep min no rep). It is a two-tailed analysis:
	
	- t-stat
	- quantile p-value
	- p-value
	- -log10(pvalue)


**Script**
::

	02_univariate_analysis.py

**INPUTs**
::

	/neurospin/brainomics/2014_deptms/datasets
		# MRI implicit mask, X and response y for the whole brain		
		MRI/
			mask_MRI_wb.nii
			X_MRI_wb.npy
			y.npy
		# PET implicit mask, X and response y for the whole brain		
		PET/
			mask_PET_wb.nii
			X_PET_wb.npy
			y.npy
		# MRI+PET implicit mask, X and response y for the whole brain		
		MRI+PET/
			mask_MRI+PET_wb.nii
			X_MRI+PET_wb.npy
			y.npy

**OUTPUTs**

MODALITY: {MRI,PET, MRI+PET} 
::
	
	/neurospin/brainomics/2014_deptms/results_univariate
		MRI/
			t_stat_rep_min_norep_MRI_wb.nii.gz
			pval-quantile_rep_min_norep_MRI_wb.nii.gz
			pval_rep_min_norep_MRI_wb.nii.gz
			pval-log10_rep_min_norep_MRI_wb.nii.gz
		PET/
			t_stat_rep_min_norep_PET_wb.nii.gz
			pval-quantile_rep_min_norep_PET_wb.nii.gz
			pval_rep_min_norep_PET_wb.nii.gz
			pval-log10_rep_min_norep_PET_wb.nii.gz
		MRI+PET/
			t_stat_rep_min_norep_MRI+PET_wb.nii.gz
			pval-quantile_rep_min_norep_MRI+PET_wb.nii.gz
			pval_rep_min_norep_MRI+PET_wb.nii.gz
			pval-log10_rep_min_norep_MRI+PET_wb.nii.gz

Results
-------
1. We observe high statistic coefficients for MRI images especially in hippocampus and frontal pole areas. The resistence to the treatment could be explained by a grey matter atrophy in those regions.

.. figure:: /neurospin/brainomics/2014_deptms/results_univariate/Result_images/t_stat_rep_min_norep_MRI_wb
	:scale: 50 %

	t statistic coefficient of MRI images in the whole brain. 

.. figure:: /neurospin/brainomics/2014_deptms/results_univariate/Result_images/pval-log10_rep_min_norep_MRI_wb
	:scale: 50 %

	p-values (-log10) associated to t statistic coefficients of MRI images in the whole brain.

2. Statistic coefficient for PET images are less high so the results for PET images seem to be less relevent. But still t statistic coefficient greater than 3 are observed in some of the ROI regions such that hippocampus.

.. figure:: /neurospin/brainomics/2014_deptms/results_univariate/Result_images/t_stat_rep_min_norep_PET_wb
	:scale: 50 %

	t statistic coefficient of PET images in the whole brain.

.. figure:: /neurospin/brainomics/2014_deptms/results_univariate/Result_images/pval-log10_rep_min_norep_PET_wb
	:scale: 50 %

	p-values (-log10) associated to t statistic coefficients of PET images in the whole brain.


Univariate Analysis with permutations
=====================================

Method
-------

The familywise error rate is now controlled using a permutation procedure, max T.
Permutation procedures provide a computationally intensive approach to generating significance levels empirically.

N = 1000 permutations are performed and for each permutation the maximal statistic t is retained.

To estimate each empirical p-value, the observed statistic is compared to the maximal statistic of every permuted test.

Hence the empirical p-value of a test is obtained as follows:

p=R/N; where R is the number of times the permuted test is greater than the observed test; N is the number of permutations.

The contrast is the same as previously [1 0 0 0] and it is a two-tailed analysis.

1. Permutation procedure performing for the whole brain 

**Script**
::

	02_univariate_analysis.py

**INPUTs**
::

	/neurospin/brainomics/2014_deptms/datasets
		# MRI implicit mask, X and response y for the whole brain		
		MRI/
			mask_MRI_wb.nii
			X_MRI_wb.npy
			y.npy
		# PET implicit mask, X and response y for the whole brain		
		PET/
			mask_PET_wb.nii
			X_PET_wb.npy
			y.npy
		# MRI+PET implicit mask, X and response y for the whole brain		
		MRI+PET/
			mask_MRI+PET_wb.nii
			X_MRI+PET_wb.npy
			y.npy

**OUTPUTs**: Empirical pvalues for each modality
::

	/neurospin/brainomics/2014_deptms/results_univariate
		MRI/
			pval-perm-log10_rep_min_norep_MRI_wb.nii.gz
			pval-perm_rep_min_norep_MRI_wb.nii.gz
			
		PET/
			pval-perm-log10_rep_min_norep_PET_wb.nii.gz
			pval-perm_rep_min_norep_PET_wb.nii.gz

		MRI+PET/
			pval-perm-log10_rep_min_norep_MRI+PET_wb.nii.gz
			pval-perm_rep_min_norep_MRI+PET_wb.nii.gz


Results 1
---------
There is no significant empirical p-value for the PET images after having done the permutation procedure.

For MRI images, only the empirical p-values in one small region light up. This region corresponds to the Middle Frontal Gyrus (mapping with the harvard oxford cortical atlas, label 4). In this region the empirical p-value is around 0.033 (pvalue resulting from a one-tailed analysis) and the statistic t is 5.4.

.. figure:: /neurospin/brainomics/2014_deptms/results_univariate/Result_images/t_stat-perm_rep_min_norep_MRI_wb
	:scale: 50 %
	
	t statistic coefficient of MRI images in the whole brain. 

.. figure:: /neurospin/brainomics/2014_deptms/results_univariate/Result_images/pval-perm_rep_min_norep_MRI_wb
	:scale: 50 %

	p-values associated to t statistic coefficients of MRI images in the whole brain

Conclusion:

Hence there is no significant result after having corrected p-values with permutation procedure max T. Nothing can be concluded.

2. p-value permutation for each ROI

Since there is no significant result considering the whole brain, we will now perform the permutation procedure max T for each ROI. 

**Script**
::

	02_univariate_analysis.py

**INPUTs**
::

	/neurospin/brainomics/2014_deptms/datasets
		# MRI mask and X associated to each ROI, and response y		
		MRI/
			mask_MRI_*.nii
			X_MRI_*.npy
			y.npy
		# PET mask and X associated to each ROI, and response y		
		PET/
			mask_PET_*.nii
			X_PET_*.npy
			y.npy
		# MRI+PET mask and X associated to each ROI, and response y		
		MRI+PET/
			mask_MRI+PET_*.nii
			X_MRI+PET_*.npy
			y.npy


**OUTPUTs**: Empirical pvalues for each modality and each ROI
::

	/neurospin/brainomics/2014_deptms/results_univariate/*{MRI, PET, MRI+PET}
		pval-perm-log10_rep_min_norep_*.nii.gz
		pval-perm_rep_min_norep_*.nii.gz

Results 2
---------
For every ROI, there is no significant result when the permutation procedure is performed on PET images.

For three ROIs, some interesting results are though obtained when the permutation procedure is performed on MRI images.

Thoses ROIs are:
	* Hippocampus
	* Frontal Pole
	* Middle Frontal Gyrus

* Hyppocampus:

significant empirical pvalue = 0.032;  -log10(pvalue) = 1.49. (pvalue resulting from a one-tailed analysis)

associated t statistique = 3.82

location : Hyppocampus Right (64.1236; 130.863; 108)

.. figure:: /neurospin/brainomics/2014_deptms/results_univariate/Result_images/t_stat-perm_rep_min_norep_MRI_hippo
	:scale: 50 %
	
	t statistic coefficient of MRI images in the whole brain. 

.. figure:: /neurospin/brainomics/2014_deptms/results_univariate/Result_images/pval-perm_rep_min_norep_MRI_hippo
	:scale: 50 %

	p-values associated to t statistic coefficients of MRI images in the Hippocampus

* Frontal Pole:

significant empirical pvalue = 0.034;  -log10(pvalue) = 1.47. (pvalue resulting from a one-tailed analysis)

associated t statistique = 4.59

location : (68.33; 40.33; 100)

.. figure:: /neurospin/brainomics/2014_deptms/results_univariate/Result_images/t_stat-perm_rep_min_norep_MRI_frontalPole
	:scale: 50 %
	
	t statistic coefficient of MRI images in the whole brain. 

.. figure:: /neurospin/brainomics/2014_deptms/results_univariate/Result_images/pval-perm_rep_min_norep_MRI_frontalPole
	:scale: 50 %

	p-values associated to t statistic coefficients of MRI images in the Frontal Pole

* Middle Frontal Gyrus:

significant empirical pvalue = 0.005;  -log10(pvalue) = 2.3. (pvalue resulting from a one-tailed analysis).

associated t statistique = 5.32

location : (118.136; 86.2886; 68)

.. figure:: /neurospin/brainomics/2014_deptms/results_univariate/Result_images/t_stat-perm_rep_min_norep_MRI_midFrontal
	:scale: 50 %
	
	t statistic coefficient of MRI images in the whole brain. 

.. figure:: /neurospin/brainomics/2014_deptms/results_univariate/Result_images/pval-perm_rep_min_norep_MRI_midFrontal
	:scale: 50 %

	p-values associated to t statistic coefficients of MRI images in the Middle Frontal Gyrus

* Cingulate Gyrus, anterior division:

significant empirical pvalue = 0.033; -log10(pvalue) = 1.48. (pvalue resulting from a one-tailed analysis).

associated t statistique = 3.93

location : (106.483; 44.7268; 92)

.. figure:: /neurospin/brainomics/2014_deptms/results_univariate/Result_images/t_stat-perm_rep_min_norep_MRI_cingulumAnt
	:scale: 50 %
	
	t statistic coefficient of MRI images in the whole brain. 

.. figure:: /neurospin/brainomics/2014_deptms/results_univariate/Result_images/pval-perm_rep_min_norep_MRI_cingulumAnt
	:scale: 50 %

	p-values associated to t statistic coefficients of MRI images in the Cingulate Gyrus, anterior division

The next step will be to perform multivariate analysis on these three ROIs.


Multivariate Analysis: ElasticNet + TV
=======================================


Conclusion
==========
