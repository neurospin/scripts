#############################################################################

00_clinical_data.py:

This script aims at generating a csv file from various xls files in
order to gather all useful clinical data.

INPUT: xls files
CLINIC_DATA_PATH: /neurospin/brainomics/2013_imagen_bmi/data/clinic/
SHFJ_DATA_PATH: /neurospin/brainomics/2013_imagen_bmi/data/clinic/source_SHFJ/
    - SHFJ_DATA_PATH: 1534bmi-vincent2.xls: initial data file.
    - CLINIC_DATA_PATH: BMI_status.xls: file containing reference values to
    attribute a weight status (i.e. Insuff, Normal, Overweight, Obese) to
    subjects according to gender, age and their BMI.

OUTPUT: csv file
    "/neurospin/brainomics/2013_imagen_bmi/data/clinic/population.csv"

Only the 1265 subjects for which we have neuroimaging data (masked_images)
are selected among the total number of subjects.

(Possibility to select subjects according to their status.)

#############################################################################

00-b_additional_clinical_data.py:

Merging of two dataframes in order to get a new .csv file containing clinical
data (initial .xls file from the SHFJ) but also additional parameters such as
gestational duration and socio-economic factor for the IMAGEN cohort.

INPUT:
- "/neurospin/brainomics/2013_imagen_bmi/data/clinic/source_SHFJ/
    1534bmi-vincent2.xls"
    clinical data on IMAGEN population (.xls initial data file)
- "/neurospin/brainomics/2013_imagen_bmi/data/clinic/
    socio_eco_factor_and_gestational_time.xls":
    additional factors for the IMAGEN cohort

OUTPUT:
- "/neurospin/brainomics/2013_imagen_bmi/data/clinic/more_clinics.csv:"
    .csv file containing both traditional clinical data but also additional
    parameters such as gestational duration and socio-economic factor

#############################################################################

12_cv_multivariate_BMI_image.py:

Multivariate study between smoothed gaser images and BMI using ElasticNet
and 10-fold cross-validation "by hand".
Computation of R2 value for various sets (alpha, l1_ratio), where alpha and
l1_ratio are defined by two lists respectively, using ElasticNet or Parsimony.

INPUT:
    - images:
IMAGES_FILE: /neurospin/brainomics/2013_imagen_bmi/data/smoothed_images.hdf5

    - BMI:
BMI_FILE: /neurospin/brainomics/2013_imagen_bmi/data/bmi.csv

OUTPUT: probability map
OUTPUT_FILE: /neurospin/brainomics/2013_imagen_bmi/data/results/
                                                    BMI_beta_map_opt.nii.gz

#############################################################################

12-1_cv_multivariate_BMI_image_fast.py:

Multivariate study between smoothed gaser images and BMI using ElasticNet
and 10-fold cross-validation "by hand".
Computation of R2 value for various sets (alpha, l1_ratio), where alpha
is fixed and l1_ratio is defined in a list, using ElasticNet or Parsimony.

INPUT:
    - images:
IMAGES_FILE: /neurospin/brainomics/2013_imagen_bmi/data/smoothed_images.hdf5

    - BMI:
BMI_FILE: /neurospin/brainomics/2013_imagen_bmi/data/bmi.csv

OUTPUT: list of R2-values

#############################################################################

13_cv_mapreduce_multivariate_BMI.py:

Cross-validation through the algorithm mapreduce, using ElasticNet or Parsimony.
UPD: BMI residualized

#############################################################################

14_cv_mltva_BMI.py: cross validation using mapreduce to be sent to Gabriel

#############################################################################

15_cv_mltva_BMI.py:

CV on BMI and images using mapreduce and ElasticNet between images and BMI.

INPUT:
- /neurospin/brainomics/2013_imagen_bmi/data/standard_mask/
  residualized_images_gender_center_TIV_pds/smoothed_images.hdf5:
    masked images (hdf5 format)
- /neurospin/brainomics/2013_imagen_bmi/data/BMI.csv:
    BMI of the 1265 subjects for which we also have neuroimaging data

METHOD: Search for the optimal set of hyperparameters (alpha, l1_ratio) that
        maximizes the correlation between the Linear Model predicted BMI
        values and true ones using Elastic Net algorithm, mapreduce and
        cross validation.
        Computation involves to send jobs to Gabriel (here, connexion
        directly specified with hl237680's account).

OUTPUT:
- the Mapper returns predicted and true values of BMI, model estimators.
- the Reducer returns R2 scores between prediction and truth.

#############################################################################

16_hot_spot_mapping.py:

Mapping of hot spots which have been revealed by high correlation ratio
after optimization of (alpha, l1) hyperparameters using Enet algorithm.
This time, we have determined the optimum hyperparameters, so that we can
run the model on the whole data.

#############################################################################

16-1_hot_spot_mapping.py: Work on normal group.

Mapping of hot spots which have been revealed by high correlation ratio
after optimization of (alpha, l1) hyperparameters using Enet algorithm.
This time, we have determined the optimum hyperparameters, so that we can
run the model on the whole dataset.

#############################################################################

17_cv_adaptive_enet_multivariate_residual_bmi_images.py:
Implementation of adaptive Enet algorithm based on Zou's article in order
to maximize the prediction score between images' voxels and BMI.

Pb: relevance of using goup lasso in images???

#############################################################################

19_connected_component_analysis.py:

Implementation of connected component analysis.
1) Label connected components;
2) Compute size and mean value of eacg region;
3) Clean up small connect components;
4) Reassign labels.
cf http://scipy-lectures.github.io/advanced/image_processing/

The function connected component analysis extracts one region of interest,
which is the intersection of non-zero beta values obtained from the beta map
and a ROI of the Harvard Oxford atlas of the brain we are focused on.
From this intersection, we get an area. We calculate the mean intensity
within this ROI for all individuals. We can thus draw the mean value profile
of GM density among all individuals. We can also draw a graph which shows
which SNPs among the selected genes are significantly correlated with GM
density, thus with BMI.
CV using mapreduce and ElasticNet between SNPs and grey matter density.

#############################################################################

20_CCA_ElasticNet_single_area.py:

The function connected component analysis extracts one region of interest,
which is the intersection of non-zero beta values obtained from the beta map
and a ROI of the Harvard Oxford atlas of the brain we are focused on.
From this intersection, we get an area. We calculate the mean intensity
within this ROI for all individuals. We can thus draw the mean value profile
of GM density among all individuals. We can also draw a graph which shows
which SNPs among the selected genes are significantly correlated with GM
density, thus with BMI.
CV using mapreduce and ElasticNet between SNPs and grey matter density.

Extracts one region of interest, which is the intersection of non-zero beta
values obtained from the beta map (output of 16-) and a ROI of the Harvard
Oxford atlas of the brain we are focused on. From this intersection, we get
an area. We calculate the mean intensity within this ROI for all individuals.
We can thus draw the mean value profile of GM density among all individuals.
We can also draw a graph which shows which SNPs among the selected genes are
significantly correlated with GM density, thus with BMI.
CV using mapreduce and ElasticNet between SNPs and grey matter density.
Here: example on the label gathering positive beta values which corresponds
to the putamen.

#############################################################################

21_cv_multivariate_bmi_SNPs_opt_hyperparameter.py:

CV using mapreduce and ElasticNet between SNPs and BMI to check out whether
the selected SNPs (from Graff - 2012 - Nature, Speliotes - 2011 - Nature,
Guo - 2013 - Human Molecular Genetics) are relevant and strongly associated
to BMI - as it should be ...

INPUT:
- /neurospin/brainomics/2013_imagen_bmi/data/clinic/source_SHFJ/1534bmi-vincent2.xls:
 Initial data file.
- /neurospin/brainomics/2013_imagen_bmi/data/BMI.csv:
 BMI of the 1.265 subjects for which we also have neuroimaging data.

METHOD: Search for the optimal set of hyperparameters (alpha, l1_ratio) that
        maximizes the correlation between the Linear Model predicted BMI
        values and true ones using Elastic Net algorithm, mapreduce and
        cross validation.

OUTPUT:
- The Mapper returns predicted and true values of BMI, model estimators,
- The Reducer returns R2 scores between prediction and truth.

#############################################################################

22_cv_multivariate_bmi_SNPs_image_opt_hyperparameter.py:

CV between images masked by beta map concatenated to SNPs and the BMI.
This script aims at checking out whether we improve the prediction model.

#############################################################################

23_univariate_residual_bmi_images.py:

Univariate correlation between BMI and images on IMAGEN subjects.
This script aims at checking out whether the crowns that appear on beta maps
obtained by multivariate analysis should be considered as an artifact due to
the segmentation process or are to be taken into account.

23-1_univariate_residual_bmi_images_normal_group.py:

Univariate correlation between BMI and images on normal subjects.
This script aims at checking out whether the crowns that appear on beta maps
obtained by multivariate analysis should be considered as an artifact due to
the segmentation process or are to be taken into account.

#############################################################################

26_univariate_residual_bmi_SNPs.py:

Univariate correlation between residualized BMI and SNPs on IMAGEN subjects.

#############################################################################