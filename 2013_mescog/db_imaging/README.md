CADASIL COHORT
==============

CAD_bioclinical_nifti/subject<ID>-M<month>-<modality>
-----------------------------------------------------
month : 0 | 18 | 36

modality:
- T1: native not registred (native)
- rT1: rT1 biolinical space "rT1"
- LL: lacune map in "rT1"
- rFLAIR: FLAIR biolinical space "rFLAIR"
- WMH: WMH map in space "rFLAIR"
- DTI: number of direction may vary
    * bval (txt file) values
    * bvec direction (div par 3)
No transformation to MNI


For Paris variable
Fro Munich if GH absent if PAS 2 x 20 direction

Meta-information
----------------

CAD_Visits_Munich
 - Data missing or not
CAD_DICOM_Munich
    if GH: scanner no diffusion

CAD_norm_M0/LL|WM_norm
LL ou WMH normalized in MNI
With FSL 4.1.8 (rT1_to_MNI_wrap or rFLAIR_to_rT1 + rT1_to_MNI_warp)


Normalization
-------------

CAD/subject<ID>/
~~~~~~~~~~~~~~~
rFLAIR_to_rT1.mat : Rigid boby transfor (FSL flirt 4.1.6)
rT1_to_MNI_warp.nii.gz: non linear tranfo
rT1_to_MNI.mat: Affine transfo (useless)

If one transformation is missing 
if problem WMH of lacune map bad QC then tr is missing

ASPS/subject<ID>/
~~~~~~~~~~~~~~~~
FLAIR_to_MTR
MTR_to_MNI

SPPSfamilly
~~~~~~~~~~~
FLAIR_to_T1
T1_to_MNI

ASPS_norm
---------
WMH map of ASPS in MNI

CAD_database_sources
--------------------
global imaging variables

WMHV : WMH Volume
~~~~

Bioclinica
~~~~~~~~~~~
LLV Volume, LL_count, MB_count / ICC Ictra cranial cavity

Sienax
~~~~~~
Uncorrected volume given FSL sienax / ICC = BPF

