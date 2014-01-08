================================================================================
== 2014                                                                       =
================================================================================

Clinical variables to predict
-----------------------------

TMTB (frontal function)                             : TMTBT42
MDRS_total:  Mattis dementia rating scale           : SCORETOT41
     (global cognitive status)
mRS: Modified Rankin Score  (global disability)     : ???    
Barthel Index (functional independency)             : ???



Baseline data as potential input predictors
-------------------------------------------

Demographic data	
~~~~~~~~~~~~~~~~

Age                                                 : AGE_AT_INCLUSION
Gender (F/M)                                        : SEXE
Education level                                     : NCULTUREL14


Vascular risk factors at baseline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Systolic blood pressure at baseline                 : PAS
Diastolic blood pressure at baseline                : PAD
Mean blood pressure at baseline                     : ???
∆ blood pressure at baseline (systolic – diastolic) : PAS - PAD
Diagnosis of hypertension at baseline               : HTA
Diagnosis of hypercholesterolemia at baseline       : HCHOLES
Smoking                                             : TABAC
HDL cholesterol                                     : CHOLHDL17
LDL cholesterol                                     : CHOLLDL17
Triglycerides                                       : TRIGLY17
Homocysteine                                        : HOMOCY17
HbA1c                                               : HBGLYCO17
CRP (C reactive protein)                            : CRP17
Glycemia at baseline                                : GLYC17
Alcohol consumption (>2 drinks)                     : CALCOOL

Clinical variables at baseline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Any history of migraine with aura                   : MIGAAURA
Any history of stroke                               : ???
Presence of gait disturbances                       : ???
Presence/absence of balance troubles                : ???
Presence/absence  of dementia                       : ???
mRS at baseline                                     : ???
TMTB at baseline                                    : TMTBT15
MMSE at baseline                                    : INDEXMMS
MDRS at baseline (mattis dementia rating scale)     : SCORETOT14
NIH stroke scale at baseline                        : ???
Barthel Index at baseline                           : ???

Global MRI data at baseline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CAD_database_soures/global_imaging_variables/CAD_M0_Bioclinica.txt

lacune_vol: lacune total volume and/or lacune number : M0_LLV
  (more reliable for clinical use)
lesion_vol: lesion total volume lesions SB           : LLcount * voxel size ???
mb_num: micro hemorrhage number                      : M0_MBcount
BPF: Brain Parenchymal Fraction (GM+WM/(GM+WM+CSF))  : The CAD_M0_Bioclinica.txt
  provide only the ICC.


Other data may be included in future research such as 
Subscales of MDRS (initiation/perseveration subscale)
MRI data including global sulcal informations or local sulcal informations
Location of WM hyperintensities using appropriate maps
Location of lacunar infarctions using appropriate maps

Other proposal (already discussed)
Use of data also obtained at M18 to consolidate the model and refine the trajectory of prediction using baseline data

================================================================================
== 2010                                                                       ==
================================================================================

Sillon:
  [1] "id"                           "label"                       
  [3] "side"                         
  
N=274
  "num" : time point (1,2,3,4) on ne travaiille que sur 1,2,3                         
Pour chaque sujet 60 (sillons) * 2 (side) * 3 (timepoint) mesure de chaque param

Clinic:
Certaines répétées <attri>_1|2

scores clinic non normaux


- Imagerie globale (time point = num)
    lacune_vol lesion_vol mb_num DWI_Peak BPF

- Imagerie pour chaque sillon (time point = num)
geodesicDepthMean     geodesicDepthMax (colinéarité ?) 
GM_thickness fold_opening surface.x


Question 1:
----------
M2 = f(M1)
devenir clinqique à M18 = f(clinic à M0 + radio à M0 + epidemio à M0)

Question 1:
-----------
M3=f(M2 + delta(M2,M1))

1

predict.t2
----------
RANKIN_2   ~ scores.t1 + clinical.t1 + imagery global at t1 + imagery sulci at t1
TMTBT_2    ~ scores.t1 + clinical.t1 + imagery global at t1 + imagery sulci at t1
SCORETOT_2 ~ scores.t1 + clinical.t1 + imagery global at t1 + imagery sulci at t1


predict.t3
----------
RANKIN_3 ~ scores.t2          + clinical.t2 + imagery global at t2       + imagery sulci at t2 +
           scores.delta.t2.t1 +             + imagery global.delta.t2.t1 + imagery sulci delta +
           
rem 
imagery global.delta.t2.t1 : delat classic sauf pour BPF où on prend Annual.PBVC

mesure.delta.t2.t1 : beta tel que;
    temps = ageirm1 et ageimr2
    mesure ~ temps * beta 

TMTBT_3
SCORETOT_3

Idées:
------
1 modèle prédictif d'un score
  généralisabilité inter centres / inter time-points
2 regarder étape par étape ce qui rajoute de l'info

Strategie 1
~~~~~~~~~~~

- Sur les FR une CV M36~M0 => oui/non je peux frabriquer une prédicteur de M36
- On applique au ALL M36~M0
- On applique à FR M54~M18 & ALL M54~M18
=> need scores 54 ALL et FR 

RANKIN_3   ~ scores.t1 + clinical.t1 + imagery global at t1 + imagery sulci at t1
TMTBT_3    ~ scores.t1 + clinical.t1 + imagery global at t1 + imagery sulci at t1
SCORETOT_3 ~ scores.t1 + clinical.t1 + imagery global at t1 + imagery sulci at t1


Strategie 2
~~~~~~~~~~~

- Sur les FR une CV M18~M0 => oui/non je peux frabriquer une prédicteur de M18
- On fabrique le prédicteur sur les les tous les FR M18~M0
  => on généralise  ALL M18~M0 et FR M36~M18 et ALL M36~M18

Virer
-----

"CHOLHDL", "CHOLTOT",
    "TRIGLY"
    
label sillons virer ventricule

================================================================================
== 23/04/2012
================================================================================

2011-02_results_splitFR-D

Variables
=========

Réponses à prédire:
-------------------
RANKIN : score moteur de 0 (aucun handicap) à 5 (grabataire)
TMTB : cognitive  (fonctions executives) chronomètre le temps
SCORETOT : échelle globale cognitive (mémoire, plannification)

RANKIN_[1|2|3] : score de handicap in 0-5
TMTBT_[1|2|3]  : temps en s de réponse
TOTAL_1,SCORETOT_[2|3]: score cognitif in démence 0-144 (normal)

RANKIN      mRS         Modified Rankin Score   (score de handicap in 0-5, motor disbility)
TMTBT       TMTB        Trial making test B (time s) (cognitive)
SCORETOT    MDRS_total  Mattis dementia rating scale) (cognitive)

Prédicteurs clinic
------------------

- Les variables du temps précédent
- Clinic + demographic
NCULTUREL_1 education rem ne bouge pas
AGEIRM_ age
SEXE.x  sex

PAS_1   SYS_BP  Systolic blood pressure
PAD_1   DIA_BP  Diastolic blood pressure
TABAC   
CHOLHDL HDL
CHOLTOT CHOL
TRIGLY  
HOMOCY  HOMOCYSTEIN (marqueur biol sanguin)
HBGLYCO HBA1C (HbA1c in mg/dl)
CRP     C-reactive Proteins (marqueur sanguin de l'inflamation)


Predicteur imagerie
-------------------
lacune_vol  lacune total vomule
lesion_vol  lesion total vomule
mb_num      micro hemorrhage number
DWI_Peak    Peak of ADC histogram
BPF         Brain Parenchimal Fraction (GM+WM/(GM+WM+CSF))

Explications

mb_num : micro saignement (pas un trou = micro hémoragie en T2*)
lacune : trou artère qui se bouche (trombose ou autre puis  ischémie)


trombose artère qui se bouche
ischémie : manquer d'oxygène
- ischémie fonctionnelle temporaire
- ischémie trop longtemps => mort cellulaire => infarctus => trou = lacune
Tous les infactus ne donent pas un trou mais dans plupart des cas on a un trou

Dans CADASIL 90% des patient ont des LI
- WML sur T2 hyper-signaux du WM: ischémie chronique infarcus incomplet (WMLesions) (lesion vol)
  * hétérogène neuro-patholologie transition vers l'infarctus ou démiélénisation ?
  *
- Lacune homogène = trou = infarctus avec cavitation (lacune vol)
- (Micro-hemorage) Microsaignements T2* pas des trous (mb_num)


Dans ASPS 90% des patients n'en ont pas, hyper-signaux

Nouvelles expérimentations
==========================

=> Essayer uniquement avec l'imagerie globale
M38~M0 + imagery global
M38~M0 + imagery global + age + sex + niveau culurel
"NCULTUREL_1", "SEXE.x" "AGEIRM_1"

- Focalise sur learn sur FR => predict sur D
Essayer learn sur D => predict sur FR
Test significativité entre les 2 modèles RANKIN1 vs CLINIC + IMAGERY global 37 vs 54

Idée :
======
Context:
CADASIL est une maladie monogénique sans ttt (traitement) connu
Evolution de la maladie doit être la même partout
?Trouver les bon marqueurs biologique pour faire un essaye clinique


Objectif:
Trouver des marqueurs biologique prédictifs et robustes d'un centre à l'autre
    si ce n'est pas la maladie qui varie c'est la qualité du marqueur
preditif + robustesse = prédire sur une base provenant d'un autre site


Outcome measures:
L'évolution = les variables mésurée moteur + cognitif
Les variables prédictives clinique (moteur+cognitif) + biologie + imagerie
Quels sont les bon prédicteurs ?

Results:




Les trois scores
================

Juste en prenant les 3 scores comme regresseur

RANKIN
======

TMTB
====

discussion prédiction TMTB décroit
----------------------------------

La prédiction du TMTB décroit quand on rajoute la clinique,
En réalité clinique = clinique + biologique
=> Quelles est la variables qui fait décroitre les résultats ?
AGEIRM_1 ou CRP
BPF = age du cerveau pourrait se subsituer à l'age
CRP ?? variabilité inter-centre

SCORETOT
========

L'ajout de mesures biologiques n'améliore la prédiction


Mon Jan 28 2013
===============

Virer DWI_PEAK
Apprendre sur D généraliser sur Fr
Fournir N pour D

