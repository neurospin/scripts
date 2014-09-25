#############################################################################

00-a_quality_control.py:

Quality control on sulci data from the IMAGEN study.

The resort to sulci -instead of considering images of anatomical structures-
should prevent us from the artifacts that may be induced by the normalization
step of the segmentation process.

INPUT:
- /neurospin/brainomics/2013_imagen_bmi/data/Imagen_mainSulcalMorphometry/full_sulci/
    one .csv file containing relevant features for each reconstructed sulcus:
 'mainmorpho_S.T.s._right.csv',
 'mainmorpho_S.T.s._left.csv',
 'mainmorpho_S.T.pol._right.csv',
 'mainmorpho_S.T.pol._left.csv',
 'mainmorpho_S.T.i._right.csv',
 'mainmorpho_S.T.i._left.csv',
 'mainmorpho_S.s.P._right.csv',
 'mainmorpho_S.s.P._left.csv',
 'mainmorpho_S.Rh._right.csv',
 'mainmorpho_S.Rh._left.csv',
 'mainmorpho_S.R.inf._right.csv',
 'mainmorpho_S.T.s.ter.asc._right.csv',
 'mainmorpho_S.T.s.ter.asc._left.csv',
 'mainmorpho_S.R.inf._left.csv',
 'mainmorpho_S.Po.C.sup._right.csv',
 'mainmorpho_S.Po.C.sup._left.csv',
 'mainmorpho_S.Pe.C._right.csv',
 'mainmorpho_S.Pe.C._left.csv',
 'mainmorpho_S.Pa.t._right.csv',
 'mainmorpho_S.Pa.t._left.csv',
 'mainmorpho_S.Pa.sup._right.csv',
 'mainmorpho_S.Pa.sup._left.csv',
 'mainmorpho_S.Pa.int._right.csv',
 'mainmorpho_S.Pa.int._left.csv',
 'mainmorpho_S.p.C._right.csv',
 'mainmorpho_S.p.C._left.csv',
 'mainmorpho_S.Or._right.csv',
 'mainmorpho_S.Or._left.csv',
 'mainmorpho_S.Or.l._right.csv',
 'mainmorpho_S.Or.l._left.csv',
 'mainmorpho_S.Olf._right.csv',
 'mainmorpho_S.Olf._left.csv',
 'mainmorpho_S.O.T.lat._right.csv',
 'mainmorpho_S.O.T.lat._left.csv',
 'mainmorpho_S.O.p._right.csv',
 'mainmorpho_S.O.p._left.csv',
 'mainmorpho_S.Li._right.csv',
 'mainmorpho_S.Li._left.csv',
 'mainmorpho_S.F.sup._right.csv',
 'mainmorpho_S.F.sup._left.csv',
 'mainmorpho_S.F.polaire.tr._right.csv',
 'mainmorpho_S.F.polaire.tr._left.csv',
 'mainmorpho_S.F.orbitaire._right.csv',
 'mainmorpho_S.F.orbitaire._left.csv',
 'mainmorpho_S.F.median._right.csv',
 'mainmorpho_S.F.median._left.csv',
 'mainmorpho_S.F.marginal._right.csv',
 'mainmorpho_S.F.marginal._left.csv',
 'mainmorpho_S.F.inter._right.csv',
 'mainmorpho_S.F.inter._left.csv',
 'mainmorpho_S.F.int._right.csv',
 'mainmorpho_S.F.int._left.csv',
 'mainmorpho_S.F.inf._right.csv',
 'mainmorpho_S.F.inf._left.csv',
 'mainmorpho_S.Cu._right.csv',
 'mainmorpho_S.Cu._left.csv',
 'mainmorpho_S.Call._right.csv',
 'mainmorpho_S.Call._left.csv',
 'mainmorpho_S.C._right.csv',
 'mainmorpho_S.C.LPC._right.csv',
 'mainmorpho_S.C.LPC._left.csv',
 'mainmorpho_S.C._left.csv',
 'mainmorpho_OCCIPITAL_right.csv',
 'mainmorpho_OCCIPITAL_left.csv',
 'mainmorpho_INSULA_right.csv',
 'mainmorpho_INSULA_left.csv',
 'mainmorpho_F.P.O._right.csv',
 'mainmorpho_F.P.O._left.csv',
 'mainmorpho_F.I.P._right.csv',
 'mainmorpho_F.I.P.r.int.2_right.csv',
 'mainmorpho_F.I.P.r.int.2_left.csv',
 'mainmorpho_F.I.P.r.int.1_right.csv',
 'mainmorpho_F.I.P.r.int.1_left.csv',
 'mainmorpho_F.I.P.Po.C.inf._right.csv',
 'mainmorpho_F.I.P.Po.C.inf._left.csv',
 'mainmorpho_F.I.P._left.csv',
 'mainmorpho_F.Coll._right.csv',
 'mainmorpho_F.Coll._left.csv',
 'mainmorpho_F.Cal.ant.-Sc.Cal._right.csv',
 'mainmorpho_F.Cal.ant.-Sc.Cal._left.csv',
 'mainmorpho_F.C.M._right.csv',
 'mainmorpho_F.C.M._left.csv',
 'mainmorpho_F.C.L._right.csv',
 'mainmorpho_F.C.L.r.sc._right.csv',
 'mainmorpho_F.C.L.r.sc._left.csv',
 'mainmorpho_F.C.L.r._right.csv',
 'mainmorpho_F.C.L.r.retroC.tr._right.csv',
 'mainmorpho_F.C.L.r.retroC.tr._left.csv',
 'mainmorpho_F.C.L.r._left.csv',
 'mainmorpho_F.C.L._left.csv'

OUTPUT:
- ~/gits/scripts/2013_imagen_bmi/scripts/Sulci/quality_control:
    sulci_df.csv: dataframe containing data for all sulci according to the
                  selected feature of interest.
                  We select subjects for who we have all genetic and
                  neuroimaging data.
                  We removed NaN rows.
    sulci_df_qc.csv: sulci_df.csv after quality control

The quality control first consists in removing sulci that are not recognized
in more than 25% of subjects.
Then, we get rid of outliers, that is we drop subjects for whom more than
25% of the remaining robust sulci have not been detected.
Finally, we eliminate subjects for whom at least one measure is aberrant,
that is we filter subjects whose features lie outside the interval
' mean +/- 3 * sigma '.

#############################################################################

00-b_demographics.py:

This script gives an insight of the distribution of ponderal status among
subjects who passed the quality control for the study on sulci.

INPUT:
- Clinical data of IMAGEN subjects:
    "/neurospin/brainomics/2013_imagen_bmi/data/clinic/population.csv"
- Sulci features after quality control:
    "/neurospin/brainomics/2013_imagen_bmi/data/Imagen_mainSulcalMorphometry/
    full_sulci/Quality_control/sulci_df_qc.csv"

OUTPUT:
    returns the number of subjects, among those who passed the quality control
    for the study on sulci, who are insuff, normal, overweight or obese.

#############################################################################

00-c_extract_clinical_data.py:

This script generates the list of IDs of subjects who passed the quality
control on sulci data.
Then, it creates a .csv file giving gender, age in years, BMI and weight
status for the 745 subjects who passed the quality control on sulci data.

INPUT:
    "/neurospin/brainomics/2013_imagen_bmi/data/Imagen_mainSulcalMorphometry/
    full_sulci/Quality_control/sulci_df_qc.csv"

OUTPUT: .csv file
    "/neurospin/brainomics/2013_imagen_bmi/data/Imagen_mainSulcalMorphometry/
    full_sulci/subjects_id_full_sulci.csv"

    "/neurospin/brainomics/2013_imagen_bmi/data/Imagen_mainSulcalMorphometry/
    full_sulci/full_sulci_clinics.csv"

Only 745 subjects among the 1265 for who we have both genetic and neuroimaging
data (masked_images) have passed the quality control step.

#############################################################################

00-d_additional_clinical_data.py:

Generation of a .csv file containing clinical data with additional parameters
such as gestational duration and socio-economic factor for IMAGEN subjects
who passed the QC on sulci.

INPUT:
- "/neurospin/brainomics/2013_imagen_bmi/data/clinic/more_clinics.csv"
    .csv file containing both traditional clinical data but also additional
    parameters such as gestational duration and socio-economic factor

- "/neurospin/brainomics/2013_imagen_bmi/data/Imagen_mainSulcalMorphometry/
    full_sulci/Quality_control/sulci_depthMax_df.csv"
    sulci maximal depth after quality control

OUTPUT:
- "/neurospin/brainomics/2013_imagen_bmi/data/Imagen_mainSulcalMorphometry/
    full_sulci/more_clinics_745sulci.csv"
    complete clinical data for subjects who passed the quality control on
    sulci

#############################################################################

01-a_univariate_bmi_full_sulci.py:

Univariate correlation between BMI and some sulci of interest on IMAGEN
subjects.
The selected sulci are particularly studied because of their robustness to
the segmentation process. These sulci are respectively split into various
subsamples by the segmentation process. As a results, they have previously
been gathered again.
Here, we select the central, precentral, collateral sulci and the calloso-
marginal fissure.
NB: Their features have previously been filtered by the quality control step.
(cf 00_quality_control.py)

The resort to sulci -instead of considering images of anatomical structures-
should prevent us from the artifacts that may be induced by the normalization
step of the segmentation process.

INPUT:
- /neurospin/brainomics/2013_imagen_bmi/data/Imagen_mainSulcalMorphometry/
  full_sulci/Quality_control/sulci_df_qc.csv:
    sulci features after quality control

- /neurospin/brainomics/2013_imagen_bmi/data/BMI.csv:
    BMI of the 1.265 subjects for which we also have neuroimaging data

METHOD: MUOLS

NB: Subcortical features, BMI and covariates are centered-scaled.

OUTPUT:
- /neurospin/brainomics/2013_imagen_bmi/data/Imagen_mainSulcalMorphometry/
  full_sulci/Results/MULM_bmi_full_sulci.txt:
    Results of MULM computation, i.e. p-value for each feature of interest,
    that the feature of interest of the selected sulci is significantly
    associated to BMI

- /neurospin/brainomics/2013_imagen_bmi/data/Imagen_mainSulcalMorphometry/
  full_sulci/Results/MULM_after_Bonferroni_correction.txt:
    Since we focus here on 85 sulci (after QC), and for each of them on
    6 features, we only keep the probability-values p < (0.05 / (6 * 85))
    that meet a significance threshold of 0.05 after Bonferroni correction.

- /neurospin/brainomics/2013_imagen_bmi/data/Imagen_mainSulcalMorphometry/
  full_sulci/Results/MUOLS_beta_values_df.csv:
    Beta values from the General Linear Model run on sulci features.

#############################################################################

01-b_univariate_bmi_full_sulci_norm-ob_groups.py:

Univariate correlation between BMI and robustly segmented sulci on IMAGEN
normal and obese subjects.

The selected sulci are particularly studied because of their robustness to
the segmentation process. These sulci are respectively split into various
subsamples by the segmentation process. As a results, they have previously
been gathered again.
NB: Their features have previously been filtered by the quality control step.
(cf 00-a_quality_control.py)

The resort to sulci -instead of considering images of anatomical structures-
should prevent us from the artifacts that may be induced by the normalization
step of the segmentation process.

In comparison to the 01-a_univariate_bmi_full_sulci.py script, we here focus
on both the normal and obese groups (in order to avoid 'mirror effects' from
the insuff and overweight groups).

INPUT:
- /neurospin/brainomics/2013_imagen_bmi/data/Imagen_mainSulcalMorphometry/
  full_sulci/Quality_control/sulci_df_qc.csv:
    sulci features after quality control

- /neurospin/brainomics/2013_imagen_bmi/data/BMI.csv:
    BMI of the 1265 subjects for which we also have neuroimaging data

METHOD: MUOLS

NB: Subcortical features, BMI and covariates are centered-scaled.

OUTPUT:
- /neurospin/brainomics/2013_imagen_bmi/data/Imagen_mainSulcalMorphometry/
  full_sulci/Results/MULM_after_Bonferroni_correction.txt:
    Since we focus here on 85 sulci (after QC), and for each of them on
    6 features, we only keep the probability-values p < (0.05 / (6 * 85))
    that meet a significance threshold of 0.05 after Bonferroni correction.

- /neurospin/brainomics/2013_imagen_bmi/data/Imagen_mainSulcalMorphometry/
  full_sulci/Results/MUOLS_beta_values_df.csv:
    Beta values from the General Linear Model run on sulci features.

#############################################################################

01-c_univariate_bmi_full_sulci-mean_pds.py:

Univariate correlation between BMI and sulci considering the mean pds on
IMAGEN subjects.

The selected sulci are particularly studied because of their robustness to
the segmentation process. These sulci are respectively split into various
subsamples by the segmentation process. As a results, they have previously
been gathered again.
NB: Their features have previously been filtered by the quality control step.
(cf 00_quality_control.py)

The resort to sulci -instead of considering images of anatomical structures-
should prevent us from the artifacts that may be induced by the normalization
step of the segmentation process.

INPUT:
- /neurospin/brainomics/2013_imagen_bmi/data/Imagen_mainSulcalMorphometry/
  full_sulci/Quality_control/sulci_df_qc.csv:
    sulci features after quality control

- /neurospin/brainomics/2013_imagen_bmi/data/BMI.csv:
    BMI of the 1265 subjects for which we also have neuroimaging data

METHOD: MUOLS
        Apply contrast only on the mean pds.

NB: Subcortical features, BMI and covariates are centered-scaled.

OUTPUT:
- /neurospin/brainomics/2013_imagen_bmi/data/Imagen_mainSulcalMorphometry/
  full_sulci/Results/MULM_after_Bonferroni_correction_mean_pds.txt:
    Since we focus here on 85 sulci (after QC), and for each of them on
    6 features, we only keep the probability-values p < (0.05 / (6 * 85))
    that meet a significance threshold of 0.05 after Bonferroni correction.

- /neurospin/brainomics/2013_imagen_bmi/data/Imagen_mainSulcalMorphometry/
  full_sulci/Results/MUOLS_beta_values_mean_pds_df.csv:
    Beta values from the General Linear Model run on sulci features for the
    mean pds.

#############################################################################

01-d_univariate_bmi_full_sulci_depthMax.py:

Univariate correlation between BMI and the maximal depth of sulci of interest
on IMAGEN subjects.
The selected sulci are particularly studied because of their robustness to
the segmentation process. These sulci are respectively split into various
subsamples by the segmentation process. As a results, they have previously
been gathered again.
NB: Their features have previously been filtered by the quality control step.
(cf 00-a_quality_control.py)

The resort to sulci -instead of considering images of anatomical structures-
should prevent us from the artifacts that may be induced by the normalization
step of the segmentation process.

INPUT:
- /neurospin/brainomics/2013_imagen_bmi/data/Imagen_mainSulcalMorphometry/
  full_sulci/Quality_control/sulci_df_qc.csv:
    sulci features after quality control

- /neurospin/brainomics/2013_imagen_bmi/data/BMI.csv:
    BMI of the 1.265 subjects for which we also have neuroimaging data

METHOD: MUOLS

NB: Subcortical features, BMI and covariates are centered-scaled.

OUTPUT:
- /neurospin/brainomics/2013_imagen_bmi/data/Imagen_mainSulcalMorphometry/
  full_sulci/Quality_control/sulci_depthMax_df.csv:
    sulci maximal depth after quality control

- /neurospin/brainomics/2013_imagen_bmi/data/Imagen_mainSulcalMorphometry/
  full_sulci/Results/MULM_depthMax_after_Bonferroni_correction.txt:
    Since we focus here on the maximal depth of 85 sulci (after QC), we only
    keep the probability-values p < (0.05 / 85) that meet a significance
    threshold of 0.05 after Bonferroni correction.

- /neurospin/brainomics/2013_imagen_bmi/data/Imagen_mainSulcalMorphometry/
  full_sulci/Results/MUOLS_depthMax_beta_values_df.csv:
    Beta values from the General Linear Model run on sulci maximal depth.

#############################################################################

01-e_univariate_SNPs_BMI.py:

Univariate correlation between BMI and the intercept of SNPs referenced in
the literature as associated to the BMI and SNPs read by the Illumina
platform on IMAGEN subjects.

INPUT:
- "/neurospin/brainomics/2013_imagen_bmi/data/clinic/population.csv"
    useful clinical data

- "/neurospin/brainomics/2013_imagen_bmi/data/BMI.csv"
    BMI of the 1.265 subjects for which we also have neuroimaging data

- "/neurospin/brainomics/2013_imagen_bmi/data/BMI_associated_SNPs_measures.csv"
    Genetic measures on SNPs of interest, that is SNPs at the intersection
    between BMI-associated SNPs referenced in the literature and SNPs read
    by the Illumina platform

METHOD: MUOLS

NB: Subcortical features and covariates are centered-scaled.

OUTPUT:
- "/neurospin/brainomics/2013_imagen_bmi/data/genetics/Results/
    MULM_BMI_SNPs_after_Bonferroni_correction.txt":
    Since we focus here on 22 SNPs, we keep the probability-values
    p < (0.05 / 22) that meet a significance threshold of 0.05 after
    Bonferroni correction.

- "/neurospin/brainomics/2013_imagen_bmi/data/genetics/Results/
    MUOLS_BMI_SNPs_beta_values_df.csv":
    Beta values from the General Linear Model run on SNPs for BMI.

#############################################################################

01-f_univariate_SNPs_from_BMI_genes_sulci.py:

Univariate correlation between BMI and the intercept of all SNPs included in
BMI-associated genes and SNPs read by the Illumina platform on IMAGEN subjects.

INPUT:
- "/neurospin/brainomics/2013_imagen_bmi/data/clinic/population.csv"
    useful clinical data

- "/neurospin/brainomics/2013_imagen_bmi/data/BMI.csv"
    BMI of the 1.265 subjects for which we also have neuroimaging data

- "/neurospin/brainomics/2013_imagen_bmi/data/
    SNPs_from_BMI-associated_genes_measures.csv"
    Genetic measures on SNPs of interest, that is SNPs at the intersection
    between all SNPs included in BMI-associated genes and SNPs read by the
    Illumina platform

METHOD: MUOLS

NB: Subcortical features and covariates are centered-scaled.

OUTPUT:
- "/neurospin/brainomics/2013_imagen_bmi/data/genetics/Results/
    MULM_BMI_SNPs_from_BMI_genes_after_Bonferroni_correction.txt":
    Since we focus here on 1615 SNPs, we keep the probability-values
    p < (0.05 / 1615) that meet a significance threshold of 0.05 after
    Bonferroni correction.

- "/neurospin/brainomics/2013_imagen_bmi/data/genetics/Results/
    MUOLS_SNPs_from_BMI_genes_beta_values.csv":
    Beta values from the General Linear Model run on SNPs for BMI.

#############################################################################

02_multivariate_bmi_sulci.py:

Multivariate correlation between BMI and one feature of interest along all
sulci on IMAGEN subjects:
   CV using mapreduce and ElasticNet between the feature of interest and BMI.

The selected sulci are particularly studied because of their robustness to
the segmentation process. These sulci are respectively split into various
subsamples by the segmentation process. As a results, they have previously
been gathered again.
Here, we select the central, precentral, collateral sulci and the calloso-
marginal fissure.
NB: Their features have previously been filtered by the quality control step.
(cf 00_quality_control.py)

The resort to sulci -instead of considering images of anatomical structures-
should prevent us from the artifacts that may be induced by the normalization
step of the segmentation process.

INPUT:
- /neurospin/brainomics/2013_imagen_bmi/data/Imagen_mainSulcalMorphometry/
  full_sulci/Quality_control/sulci_df_qc.csv:
    sulci features after quality control for all sulci of interest

- /neurospin/brainomics/2013_imagen_bmi/data/BMI.csv:
    BMI of the 1265 subjects for which we also have neuroimaging data

METHOD: Search for the optimal set of hyperparameters (alpha, l1_ratio) that
        maximizes the correlation between predicted BMI values via the Linear
        Model  and true ones using Elastic Net algorithm, mapreduce and
        cross validation.
--NB: Computation involves to send jobs to Gabriel.--

NB: Subcortical features, BMI and covariates are centered-scaled.

OUTPUT:
- the Mapper returns predicted and true values of BMI, model estimators.
- the Reducer returns R2 scores between prediction and true values.

#############################################################################

02-b_multivariate_sulci_SNPs_BMI.py:

Multivariate correlation between BMI and the concatenation of SNPs known to
be associated to the BMI and read by the Illumina chip and the maximal depth
along sulci on IMAGEN subjects:
   CV using mapreduce and ElasticNet between the features of interest and BMI.

NB: Sulci maximal depth has previously been filtered by the quality control
step. (cf 00-a_quality_control.py)

The resort to sulci -instead of considering images of anatomical structures-
should prevent us from the artifacts that may be induced by the normalization
step of the segmentation process.

The idea to go through a multivariate analysis lies within the scope of
investigating a better relevance and improved efficiency of the model, and
a stronger associativity.

INPUT:
- '/neurospin/brainomics/2013_imagen_bmi/data/Imagen_mainSulcalMorphometry/
  full_sulci/Quality_control/sulci_depthMax_df.csv'
    sulci maximal depth after quality control

- '/neurospin/brainomics/2013_imagen_bmi/data/BMI_associated_SNPs_measures.csv'
    Genetic measures on SNPs of interest, that is SNPs at the intersection
    between BMI-associated SNPs referenced in the literature and SNPs read
    by the Illumina platform

- '/neurospin/brainomics/2013_imagen_bmi/data/BMI.csv'
    BMI of the 1.265 subjects for which we also have neuroimaging data

METHOD: Search for the optimal set of hyperparameters (alpha, l1_ratio) that
        maximizes the correlation between predicted BMI values via the Linear
        Model  and true ones using Elastic Net algorithm, mapreduce and
        cross validation.
--NB: Computation involves to send jobs to Gabriel.--

NB: Subcortical features, BMI and covariates are centered-scaled.

OUTPUT:
- the Mapper returns predicted and true values of BMI, model estimators.
- the Reducer returns R2 scores between prediction and true values.

#############################################################################

03_mltv_beta_map_bmi_full_sulci_all_features.py:

Mapping of hot spots which have been revealed by high correlation ratio
after optimization of (alpha, l1) hyperparameters using Enet algorithm.
This time, we have determined the optimum hyperparameters, so that we can
run the model on the whole dataset.

The selected sulci are robust to the segmentation process. These sulci are
respectively split into various subsamples by the segmentation process. As
a results, they have previously been gathered again.
NB: Their features have previously been filtered by the quality control step.
(cf 00_quality_control.py)

The resort to sulci -instead of considering images of anatomical structures-
should prevent us from the artifacts that may be induced by the normalization
step of the segmentation process.

INPUT:
- /neurospin/brainomics/2013_imagen_bmi/data/Imagen_mainSulcalMorphometry/
  full_sulci/Quality_control/sulci_df_qc.csv:
    sulci features after quality control

- /neurospin/brainomics/2013_imagen_bmi/data/BMI.csv:
    BMI of the 1265 subjects for which we also have neuroimaging data

METHOD: multivariate GLM

NB: Subcortical features, BMI and covariates are centered-scaled.

OUTPUT:
- /neurospin/brainomics/2013_imagen_bmi/data/Imagen_mainSulcalMorphometry/
  full_sulci/Results/mltv_beta_values_df.csv:
    returns for each sulcus the beta value associated to the GLM
    determined by the selected optimized set of hyperparameters.

#############################################################################