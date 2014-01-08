#Clinical variables to predict
#-----------------------------

#TMTB (frontal function)                             : TMTBT42
#MDRS_total:  Mattis dementia rating scale           : SCORETOT41
#     (global cognitive status)
#mRS: Modified Rankin Score  (global disability)     : ???    
#Barthel Index (functional independency)             : ???
target=c(
"TMTBT42",
"SCORETOT41")

#Baseline data as potential input predictors
#-------------------------------------------

#Demographic data	
#~~~~~~~~~~~~~~~~

#Age                                                 : AGE_AT_INCLUSION
#Gender (F/M)                                        : SEXE
#Education level                                     : NCULTUREL14
demographic=c(
"AGE_AT_INCLUSION",
"SEXE",
"NCULTUREL14")

#Vascular risk factors at baseline
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Systolic blood pressure at baseline                 : PAS
#Diastolic blood pressure at baseline                : PAD
#Mean blood pressure at baseline                     : ???
#∆ blood pressure at baseline (systolic – diastolic) : PAS - PAD
#Diagnosis of hypertension at baseline               : HTA
#Diagnosis of hypercholesterolemia at baseline       : HCHOLES
#Smoking                                             : TABAC
#HDL cholesterol                                     : CHOLHDL17
#LDL cholesterol                                     : CHOLLDL17
#Triglycerides                                       : TRIGLY17
#Homocysteine                                        : HOMOCY17
#HbA1c                                               : HBGLYCO17
#CRP (C reactive protein)                            : CRP17
#Glycemia at baseline                                : GLYC17
#Alcohol consumption (>2 drinks)                     : CALCOOL

vascular=c(
"PAS",
"PAD",
"???",
"PAS - PAD",
"HTA",
"HCHOLES",
"TABAC",
"CHOLHDL17",
"CHOLLDL17",
"TRIGLY17",
"HOMOCY17",
"HBGLYCO17",
"CRP17",
"GLYC17",
"CALCOOL")

#Clinical variables at baseline
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Any history of migraine with aura                   : MIGAAURA
#Any history of stroke                               : ???
#Presence of gait disturbances                       : ???
#Presence/absence of balance troubles                : ???
#Presence/absence  of dementia                       : ???
#mRS at baseline                                     : ???
#TMTB at baseline                                    : TMTBT15
#MMSE at baseline                                    : INDEXMMS
#MDRS at baseline (mattis dementia rating scale)     : SCORETOT14
#NIH stroke scale at baseline                        : ???
#Barthel Index at baseline                           : ???
clinical=c(
"MIGAAURA",
"???",
"???",
"???",
"???",
"???",
"TMTBT15",
"INDEXMMS",
"SCORETOT14",
"???",
"???")

#Global MRI data at baseline
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~
