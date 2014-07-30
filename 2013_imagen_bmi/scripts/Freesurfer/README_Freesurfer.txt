#############################################################################

00_quality_control?

#############################################################################

01_univariate_bmi_sulci:
Univariate correlation between residualized BMI and volume of subcortical
structures (Freesurfer) on IMAGEN subjects.

The resort to Freesurfer should prevent us from the artifacts that may be
induced by the normalization step of the SPM segmentation process.

INPUT:
- /neurospin/brainomics/2013_imagen_bmi/data/Freesurfer:
   .csv file containing volume of subcortical structures obtained by Freesurfer
- /neurospin/brainomics/2013_imagen_bmi/data/BMI.csv:
   BMI of the 1265 subjects for which we also have neuroimaging data

METHOD: MUOLS

OUTPUT: returns a probability for each subcortical structure to be
        significantly associated to BMI.

#############################################################################

02_multivariate_bmi_sulci:
Multivariate correlation between residualized BMI and one subcortical feature
of interest:
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

OUTPUT:
- the Mapper returns predicted and true values of BMI, model estimators.
- the Reducer returns R2 scores between prediction and true values.

#############################################################################