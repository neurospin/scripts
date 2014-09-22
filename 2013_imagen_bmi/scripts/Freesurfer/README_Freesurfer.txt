#############################################################################

NB: Quality control not necessary here since NaN rows are dropped when reading
    the csv file.

#############################################################################

01_univariate_bmi_freesurfer:

Univariate correlation between BMI and volume of subcortical structures
(Freesurfer) on IMAGEN subjects.

The resort to Freesurfer should prevent us from the artifacts that may be
induced by the normalization step of the SPM segmentation process.

INPUT:
- /neurospin/brainomics/2013_imagen_bmi/data/Freesurfer:
   .csv file containing volume of subcortical structures obtained by Freesurfer

- /neurospin/brainomics/2013_imagen_bmi/data/BMI.csv:
   BMI of the 1265 subjects for whom we also have neuroimaging and genetic
   data

METHOD: MUOLS

NB: Features extracted by Freesurfer, BMI and covariates are centered-scaled.

OUTPUT:
- /neurospin/brainomics/2013_imagen_bmi/data/Freesurfer/Results/
  MULM_bmi_freesurfer.txt:
    Results of MULM computation, i.e. p-value for each feature of interest,
    that this feature extracted by Freesurfer algorithm is significantly
    associated to BMI

- /neurospin/brainomics/2013_imagen_bmi/data/Freesurfer/Results/
  MULM_after_Bonferroni_correction.txt:
    Since we focus here on 9 features extracted by Freesurfer, we only keep
    the probability-values p < (0.05 / 9) that meet a significance threshold
    of 0.05 after Bonferroni correction.

- /neurospin/brainomics/2013_imagen_bmi/data/Imagen_mainSulcalMorphometry/
  full_sulci/Results/MUOLS_beta_values_df.csv:
    Beta values from the General Linear Model run on subcortical structures
    extracted by Freesurfer.

#############################################################################

02_multivariate_bmi_freesurfer:

Multivariate correlation between BMI and one subcortical feature of interest:
   CV using mapreduce and ElasticNet between the feature of interest and BMI.

The resort to Freesurfer should prevent us from the artifacts that may be
induced by the normalization step of the SPM segmentation process.

INPUT:
- /neurospin/brainomics/2013_imagen_bmi/data/Freesurfer:
   .csv file containing volume of subcortical structures obtained by Freesurfer
- /neurospin/brainomics/2013_imagen_bmi/data/BMI.csv:
   BMI of the 1265 subjects for which we also have neuroimaging data

METHOD: Search for the optimal set of hyperparameters (alpha, l1_ratio) that
        maximizes the correlation between predicted BMI values via the Linear
        Model  and true ones using Elastic Net algorithm, mapreduce and
        cross validation.
--NB: Computation involves to send jobs to Gabriel.--

NB: Features extracted by Freesurfer, BMI and covariates are centered-scaled.

OUTPUT:
- the Mapper returns predicted and true values of BMI, model estimators.
- the Reducer returns R2 scores between prediction and true values.

#############################################################################