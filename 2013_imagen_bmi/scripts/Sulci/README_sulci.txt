#############################################################################

00_quality_control.py:
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
Then, we get rid of outliers, that is we drop subjects for which more than
25% of the remaining robust sulci have not been detected.
Finally, we eliminate subjects for whom at least one measure is aberrant,
that is we filter subjects whose features lie outside the interval
' mean +/- 3 * sigma '.

#############################################################################

01_univariate_bmi_sulci.py:
Univariate correlation between residualized BMI and some sulci of interest
on IMAGEN subjects.

The selected sulci are particularly studied because of their robustness to
the segmentation process. These sulci are respectively split into several
subsamples by the segmentation process. As a results, Clara Fischer helped
us to re-assemble the main sulci before processing our data.

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
    BMI of the 1265 subjects for which we also have neuroimaging data

METHOD: MUOLS

NB: Subcortical features, BMI and covariates are centered-scaled.

OUTPUT: returns a probability that the feature of interest of the selected
        sulci is significantly associated to BMI.

#############################################################################

02_multivariate_bmi_sulci.py:
Multivariate correlation between residualized BMI and one feature of interest
along all sulci on IMAGEN subjects:
   CV using mapreduce and ElasticNet between the feature of interest and BMI.

The selected sulci are particularly studied because of their robustness to
the segmentation process. These sulci are respectively split into various
subsamples by the segmentation process. As a results, Clara Fischer helped
us to re-assemble the main sulci before processing our data.

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

03_mltv_beta_map_bmi_full_sulci_all_features.py:
Mapping of hot spots which have been revealed by high correlation ratio
after optimization of (alpha, l1) hyperparameters using Enet algorithm.
This time, we have determined the optimum hyperparameters, so that we can
run the model on the whole dataset.

#############################################################################