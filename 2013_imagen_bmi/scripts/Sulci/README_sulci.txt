#############################################################################

00_quality_control:
Quality control on sulci data from the IMAGEN study.

The resort to sulci -instead of considering images of anatomical structures-
should prevent us from the artifacts that may be induced by the normalization
step of the segmentation process.

INPUT:
- /neurospin/brainomics/2013_imagen_bmi/data/Imagen_mainSulcalMorphometry:
    one .csv file for each sulcus containing relevant features

OUTPUT:
- ~/gits/scripts/2013_imagen_bmi/scripts/Sulci/quality_control:
    sulci_df.csv: dataframe containing data for all sulci according to the
                  selected feature of interest.
                  We select subjects for who we have all genetic and
                  neuroimaging data.
                  We removed NaN rows.
    GM_thickness_qc.csv: statistics for each sulcus
    sulci_df_qc.csv: sulci_df.csv after quality control

The quality control first consists in removing sulci that are not recognized
in more than 25% of subjects.
Then, we get rid of outliers, that is we drop subjects for which more than
25% of the remaining robust sulci have not been detected.

#############################################################################

01_univariate_bmi_sulci:
Univariate correlation between residualized BMI and sulci features (mean
geodesic depth, gray matter thickness, fold opening) on IMAGEN subjects.

The resort to sulci -instead of considering images of anatomical structures-
should prevent us from the artifacts that may be induced by the normalization
step of the segmentation process.

INPUT:
- /neurospin/brainomics/2013_imagen_bmi/data/Imagen_mainSulcalMorphometry:
    one .csv file for each sulcus containing relevant features
- /neurospin/brainomics/2013_imagen_bmi/data/BMI.csv:
    BMI of the 1265 subjects for which we also have neuroimaging data

METHOD: MUOLS

OUTPUT: returns a probability for each sulcus file that the feature of
        interest for this sulcus is significantly correlated to BMI.

NB: Since some sulci are split into various parts during the segmentation
    process, Clara Fischer helped us to re-assemble the main sulci before
    processing our data.
    This leads to scripts 04_univariate_bmi_central_sulcus.py and
    05_univariate_bmi_full_sulci.py.

#############################################################################

02_multivariate_bmi_sulci:
Multivariate correlation between residualized BMI and one feature of interest
(mean geodesic depth, gray matter thickness, fold opening) along all sulci on
IMAGEN subjects:
   CV using mapreduce and ElasticNet between the feature of interest and BMI.

The resort to sulci -instead of considering images of anatomical structures-
should prevent us from the artifacts that may be induced by the normalization
step of the segmentation process.


INPUT:
- /neurospin/brainomics/2013_imagen_bmi/data/Imagen_mainSulcalMorphometry:
    .csv files containing relevant features for each sulcus of interest
    that are merged into one single dataframe (subject_id vs. same feature
    along all sulci)
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

04_univariate_bmi_central_sulcus:
Univariate correlation between residualized BMI and one sulcus of interest
on IMAGEN subjects.
Here, we select the central sulcus.

The resort to sulci -instead of considering images of anatomical structures-
should prevent us from the artifacts that may be induced by the normalization
step of the segmentation process.

INPUT:
- /neurospin/brainomics/2013_imagen_bmi/data/Imagen_mainSulcalMorphometry/
   mainmorpho_S.C._left.csv
- /neurospin/brainomics/2013_imagen_bmi/data/BMI.csv:
    BMI of the 1265 subjects for which we also have neuroimaging data

METHOD: MUOLS

OUTPUT: returns a probability that the feature of interest of the central
        sulcus is significantly associated to BMI.

#############################################################################

05_univariate_bmi_full_sulci.py:
Univariate correlation between residualized BMI and some sulci of interest
on IMAGEN subjects.
The selected sulci are particularly studied because of their robustness to
the segmentation process. These sulci are respectively split into various
subsamples by the segmentation process. As a results, they have previously
been gathered again.
Here, we select the central, precentral, collateral sulci and the calloso-
marginal fissure.

The resort to sulci -instead of considering images of anatomical structures-
should prevent us from the artifacts that may be induced by the normalization
step of the segmentation process.

INPUT:
- /neurospin/brainomics/2013_imagen_bmi/data/Imagen_mainSulcalMorphometry/full_sulci/
    mainmorpho_S.C._left.csv
    mainmorpho_S.C._right.csv
    mainmorpho_S.Pe.C._left.csv
    mainmorpho_S.Pe.C._right.csv
    mainmorpho_F.Coll._left.csv
    mainmorpho_F.Coll._right.csv
    mainmorpho_F.C.M._left.csv
    mainmorpho_F.C.M._right.csv

- /neurospin/brainomics/2013_imagen_bmi/data/BMI.csv:
    BMI of the 1265 subjects for which we also have neuroimaging data

METHOD: MUOLS

OUTPUT: returns a probability that the feature of interest of the selected
        sulci is significantly associated to BMI.

#############################################################################