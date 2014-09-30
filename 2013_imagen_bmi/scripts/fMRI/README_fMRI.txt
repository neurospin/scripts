#############################################################################

01-a_unfold_left_motor_fMRI.py:

Read activation maps from right motor tasks fMRI acquisitions, unfold them
for each subject in order to get an array where each row corresponds to a
subject and each column to the different voxels within the subject's image.

The order is read from the file giving the list of subjects who passed the
quality control on SNPs data.

INPUT:
- '/neurospin/brainomics/2013_imagen_bmi/data/Imagen_mainSulcalMorphometry/
    full_sulci/subjects_id_full_sulci.csv':
    List of subjects who passed the quality control on sulci data
- '/neurospin/brainomics/2013_imagen_bmi/data/fMRI_GCA_motor_left/IMAGEN/
    arc001/processed/spmstatsintra/ ...'
    fMRI activation maps for left motor tasks

OUTPUT:
- '/neurospin/brainomics/2013_imagen_bmi/data/fMRI_GCA_motor_left/
    subjects_id_left_motor_fMRI.csv'
    List of subjects ID for whom we have both left motor fMRI tasks and
    sulci data
- '/neurospin/brainomics/2013_imagen_bmi/data/fMRI_GCA_motor_left/
    GCA_motor_left_images.npy'
    Unfolded fMRI images for left motor tasks saved in an array-like format

#############################################################################

01-b_unfold_right_motor_fMRI.py:

Read activation maps from right motor tasks fMRI acquisitions, unfold them
for each subject in order to get an array where each row corresponds to a
subject and each column to the different voxels within the subject's image.

The order is read from the file giving the list of subjects who passed the
quality control on SNPs data.

INPUT:
- '/neurospin/brainomics/2013_imagen_bmi/data/Imagen_mainSulcalMorphometry/
    full_sulci/subjects_id_full_sulci.csv':
    List of subjects who passed the quality control on sulci data
- '/neurospin/brainomics/2013_imagen_bmi/data/fMRI_GCA_motor_right/IMAGEN/
    arc001/processed/spmstatsintra/ ...'
    fMRI activation maps for right motor tasks

OUTPUT:
- '/neurospin/brainomics/2013_imagen_bmi/data/fMRI_GCA_motor_right/
    subjects_id_right_motor_fMRI.csv'
    List of subjects ID for whom we have both motor right fMRI tasks and
    sulci data
- '/neurospin/brainomics/2013_imagen_bmi/data/fMRI_GCA_motor_right/
    GCA_motor_right_images.npy'
    Unfolded fMRI images for right motor tasks saved in an array-like format

#############################################################################

01-c_unfold_left+right_motor_fMRI.py:

Read activation maps from left + right motor tasks fMRI acquisitions, unfold
them for each subject in order to get an array where each row corresponds to
a subject and each column to the different voxels within the subject's image.

The order is read from the file giving the list of subjects who passed the
quality control on SNPs data.

INPUT:
- '/neurospin/brainomics/2013_imagen_bmi/data/Imagen_mainSulcalMorphometry/
    full_sulci/subjects_id_full_sulci.csv':
    List of subjects who passed the quality control on sulci data
- '/neurospin/brainomics/2013_imagen_bmi/data/fMRI_GCA_motor_left_right/IMAGEN/
    arc001/processed/spmstatsintra/ ...'
    fMRI activation maps for left + right motor tasks

OUTPUT:
- '/neurospin/brainomics/2013_imagen_bmi/data/fMRI_GCA_motor_left_right/
    subjects_id_left_right_motor_fMRI.csv'
    List of subjects ID for whom we have both motor left + right fMRI tasks
    and sulci data
- '/neurospin/brainomics/2013_imagen_bmi/data/fMRI_GCA_motor_left_right/
    GCA_motor_left_right_images.npy'
    unfolded fMRI images for left + right motor tasks saved in an array-like
    format

#############################################################################

02-a_univariate_left_motor_fMRI_SNPs.py:

Univariate association study between left motor tasks in fMRI and SNPs at
the intersection between BMI-associated SNPs referenced in the literature
and SNPs read by the Illumina platform.

INPUT:
- '/neurospin/brainomics/2013_imagen_bmi/data/clinic/population.csv'
    useful clinical data

- '/neurospin/brainomics/2013_imagen_bmi/data/fMRI_GCA_motor_left/
    subjects_id_left_motor_fMRI.csv'
    List of subjects ID for whom we have both motor left fMRI tasks and
    sulci data

- '/neurospin/brainomics/2013_imagen_bmi/data/BMI_associated_SNPs_measures.csv'
    Genetic measures on SNPs of interest, that is SNPs at the intersection
    between BMI-associated SNPs referenced in the literature and SNPs read
    by the Illumina platform

- '/neurospin/brainomics/2013_imagen_bmi/data/fMRI_GCA_motor_left/
    GCA_motor_left_images.npy'
    unfolded fMRI images for left motor tasks saved in an array-like format

METHOD: MUOLS

NB: Covariates are centered-scaled.

OUTPUT:
- '/neurospin/brainomics/2013_imagen_bmi/data/fMRI_GCA_motor_left/Results/
    MULM_left_motor_fMRI_SNPs_after_Bonferroni_correction.txt'
    Since we focus here on 22 SNPs, we keep the probability-values
    p < (0.05 / 22) that meet a significance threshold of 0.05 after
    Bonferroni correction.

- '/neurospin/brainomics/2013_imagen_bmi/data/fMRI_GCA_motor_left/Results/
    MUOLS_left_motor_fMRI_SNPs_beta_values_df.csv'
    Beta values from the General Linear Model run on SNPs for fMRI left motor
    tasks.

#############################################################################

03-a_multivariate_left_motor_fMRI_SNPs.py:

Multivariate association analysis between left motor tasks fMRI and SNPs at
the intersection between BMI-associated SNPs referenced in the literature
and SNPs read by the Illumina platform.

INPUT:
- '/neurospin/brainomics/2013_imagen_bmi/data/clinic/population.csv'
    useful clinical data

- '/neurospin/brainomics/2013_imagen_bmi/data/fMRI_GCA_motor_left/
    subjects_id_left_motor_fMRI.csv'
    List of subjects ID for whom we have both motor left fMRI tasks and
    sulci data

- '/neurospin/brainomics/2013_imagen_bmi/data/BMI_associated_SNPs_measures.csv'
    Genetic measures on SNPs of interest, that is SNPs at the intersection
    between BMI-associated SNPs referenced in the literature and SNPs read
    by the Illumina platform

- '/neurospin/brainomics/2013_imagen_bmi/data/fMRI_GCA_motor_left/
    GCA_motor_left_images.npy'
    unfolded fMRI images for left motor tasks saved in an array-like format

METHOD: Search for the optimal set of hyperparameters (alpha, l1_ratio) that
        maximizes the correlation between predicted BMI values via the Linear
        Model  and true ones using Elastic Net algorithm, mapreduce and
        cross validation.
--NB: Computation involves to send jobs to Gabriel.--

NB: Covariates are centered-scaled.

OUTPUT:
- the Mapper returns predicted and true values of BMI, model estimators.
- the Reducer returns R2 scores between prediction and true values.

#############################################################################