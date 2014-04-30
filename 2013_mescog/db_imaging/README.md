NEUROIMAGING DATA
=================


- `original`: data received in March 2013 in Munich meeting
- `ressources`: other ressources: atlas etc.

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

MUNICH
======

ASPS_norm
---------

**Content**: WMH map of ASPS in MNI

**Organisation**: `ASPS_norm/ASPS_<subject_id>-WMH_norm.nii.gz`


CAD_bioclinical_nifti
---------------------

**Content**: CADASIL (Paris + Munich) Neuroimaging data.

**Organisation**: `CAD_bioclinical_nifti/<subject_id>/<subject_id>-M<month>-<modality>`

With:

- month : 0 or 18 or 36
- modality:
    * T1: native not registred (native)
    * rT1: T1 in biolinical space: `rT1`
    * LL: lacune map in `rT1`
    * rFLAIR: FLAIR biolinical space `rFLAIR`
    * WMH: WMH map in space `rFLAIR`
    * DTI: number of direction may vary
        + bval (txt file) values
        + bvec direction (div par 3)
   No transformation to MNI

?? For Munich if GH absent if PAS 2 x 20 direction


CAD_database_sources
--------------------

**Content**: global imaging variables for CADASIL

**Organisation**: `global imaging variables/CAD_M??_<modality>.txt`

With modality:

- Bioclinica:  `ID, M0_LLV, M0_LLcount, M0_MBcount, M0_ICC`. ICC: Intra cranial cavity
- Sienax: `ID, M0_SIENAX`. Brain volume given FSL: sienax / ICC = BPF
- WMHV: : `ID, M0_WMHV`

Also present:

- base_commun.xlsx: see clinical database.
- france2012.xlsx: see clinical database.


CAD_norm_M0
-----------

**Content**: CADASIL LL and WMH Maps in MNI at M0. Can be obtained by applying normaliszation onto maps: (`CAD_bioclinical_nifti`).

**Organisation**: `CAD_norm_M0/??_norm/<subject_id>-M0_norm.nii.gz`

With: ?? ::= `LL` or `WMH`


Normalization
-------------

**Content**: Normalization transformations from native space to MNI. 

**Organisation**: `Normalization/<base>/<subject_id>/<transoformation>`

With `<base>`:

- `ASPS/<subject_id>/<transformation>`:
    * FLAIR_to_MTR.mat
    * MTR_to_MNI2mn.mat
    * MTR_to_MNI2mm_warp.nii.gz

- `ASPSFamily/<subject_id>/<transformation>`:
    * FLAIR_to_MTR.mat
    * MTR_to_MNI2mn.mat
    * MTR_to_MNI2mm_warp.nii.gz

- `CAD/<subject_id>/<transformation>`:
    * rFLAIR_to_rT1.mat: Rigid boby transfor (FSL flirt 4.1.6)
    * rT1_to_MNI.mat: non linear tranfo
    * rT1_to_MNI_warp.nii.gz: Affine transfo (useless)

Note
Missing tranformation if problem with WMH or lacune map: bad QC.


Meta-information
----------------

- CAD_Visits_Munich
    * Data missing or not
- CAD_DICOM_Munich
    * if GH: scanner no diffusion

CAD_norm_M0/LL|WM_norm
LL ou WMH normalized in MNI
With FSL 4.1.8 (rT1_to_MNI_wrap or rFLAIR_to_rT1 + rT1_to_MNI_warp)

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

GRAZ
====


**Organisation**. All directory follow the same organisation: `/<modality>_<space>_<processing>/<cohort>/<subject_id>/<time_point>/`

- `modality`: `diff, flair, mtr, r2, t1, t2, t2star`
- `space`: `n`: native.
- `processing`: `raw`: no processing, `sub-FSL`:, `reg-FSLflirt`:

DataGraz.xlsx

diff_n_raw
----------

**Content**: 

flair_n_raw
-----------

**Content**: 

mtr_n_raw
---------

**Content**: 

r2_n_sub-FSL
------------

**Content**: 

t1_n_raw
--------

**Content**: 

t1_n_reg-FSLflirt
-----------------

**Content**: 

t2_n_raw
--------

**Content**: 

t2star_n_raw
------------

**Content**: 
