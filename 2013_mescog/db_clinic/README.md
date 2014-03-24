COMON DATABASE
==============

The common database of clinical variables:
- `commondb_clinic_cadasil-asps-aspfs_<date>.csv`

A summary of common variables:
- `commondb_clinic_cadasil-asps-aspfs_mapping-summary_<date>.csv`
This file contains the mapping between the cadasil/asps, and some statistics (mean, sd, min, max, etc.) for a basic QC for each variable and for each cohorts (cadasil, asps, aspfs).

HISTORY OF THE COMMON DB
========================

DB_Mapping.*
------------
- `DB_Mapping_Longit_Last_EJ_2013-05-08.xlx` <=> `DB_Mapping_Longit_Last_EJ_2013-05-08.csv`
- `DB_Mapping_Longit_Last_EJ_20131007`. Recode EMBOLICDESEASE => EMBOLICDISEASE

CADASIL
-------
- `base_commun.xlsx` => `base_commun.csv`
- `base_commun_20131008.csv` Add DATEINCL and DATENAIS in CADASIL (See 00_MISSING_CADASIL.py)
- `france2012.csv`
- `CAD_Munich_Dates.txt`
  Date birth/inclusion for munich patients

ASP(F)S_klinVariables_*
---------------------
- `ASPS_klinVariables_20130806.sav` received from Hofer Edith => `ASPS_klinVariables_20130806.csv`.
    Individual 1607 is actually 53 (not 0).
    Individual 1911 is 54 (and not 34 as in the file I sent you). =>
- `ASPS_klinVariables_20131015.csv`
- `ASPFS_klinVariables_20130711.sav` received from Hofer Edith => `ASPFS_klinVariables_20130711.csv`


--------------------------------------------------------------------------------
TODO Update from Edith Hofer:
18/12/2013
Just a reminder: for ASPS Family we have two different IDs:
ID: this is the ID which was in the file with ASPS Family clinical variables which I already sent to you
mr_id: this is the Id of our mr images. I now added this id to the files


ASPS_klinVariables_20131218.sav:
This is the file I already sent you but I added 2 new Variables:


corresponding_ASPS_Family_ID:  it has a value only if the same indiviual also exists in the ASPS Family database, the value is the ASPS Family ID


corresponging_ASPS_Family_mr_nr: it has a value only if the same indiviual also exists in the ASPS Family database, the value is the ASPS Family mr_nr


ASPS_Family_klinVariables_20131218.sav:
This file is sligthly different than the file I already sent to you because I removed all the individuals for which we do not have any MR data (and therefore they also do not have an mr_nr)


I added 2 new Variables:


corresponding_ASPS_ID: it has a value only if the same indiviual also exists in the ASPS database, the value is the ASPS ID


mr_nr: this is the ASPS Family mr number


Please do not hesitate to ask if you have any questions!


Best wishes,
Edith
--------------------------------------------------------------------------------


MISSING DATA
============

Add DATEINCL and DATENAIS an compute AGE_AT_INCLUSION in CADASIL subjects

**Script**

https://github.com/neurospin/scripts/blob/master/2013_mescog/db_clinic/00_MISSING_CADASIL.py
Access is retricted, ask edouard.duchesnay@cea.fr

**Input**

- `base_commun.csv`
- `france2012.csv` => date DATEINCL and DATENAIS for french
- `CAD_Munich_Dates.txt` => date DATEINCL and DATENAIS for german

**Output**

- `base_commun_20140109.csv` == `base_commun.csv` + Date (from `france2012.csv` + `CAD_Munich_Dates.txt`)

QC for CDADASIL
===============
Check if `france2012.csv` and `base_commun_20131003.csv` are simillar.
Manualy correct some mistakes (unit and values)

**Script**

https://github.com/neurospin/scripts/blob/master/2013_mescog/db_clinic/01_QC_CADASIL.py
Access is retricted, ask edouard.duchesnay@cea.fr


**Input**

- `france2012.csv`
- `base_commun_20140109.csv`

**Output**

- `QC/cadasil_qc.csv`
- `QC/cadasil_qc.html`
- `base_commun_20140109.csv`

where:
For each variable in base_commun.csv (378 samples):
- in_france2012   : is the variable in france2012.csv (249 samples)
- diff            : if numeric the maximum difference, if symbolic the number of diff
- n_missing       : nb of missing values (in base_commun.csv)
- n_missing_base_commun_but_not_infr2012 : nb of missing values (in base_commun.csv) but not in france2012.csv


MERGE BASES
===========

**Script**

https://github.com/neurospin/scripts/blob/master/2013_mescog/db_clinic/02_merge_CADASIL-ASPS.py
Access is retricted, ask edouard.duchesnay@cea.fr


**Input**

- `base_commun_20140109.csv`
- `ASPS_klinVariables_20131015.csv`
- `ASPFS_klinVariables_20130711.csv`
- `DB_Mapping_Longit_Last_EJ_20131007.csv`

**Output**

1. Summary of common DB
    `db_clinic_cadasil-asps_mapping_summary.csv`
    `db_clinic_cadasil-asps_mapping_summary.html`
2. Common DB
    `db_clinic_cadasil-asps-common.csv`


ISSUES ON CADASIL BASE
======================

Unit problems
-------------

LEUKO_COUNT,, LEUCO17
Problem: Unit are clearly not compatible.
ASPS/ASPFS seem to be in cells per cubic millimeter of blood (mean ASPS=5991.79).
CADASIL seems to be be in 109 cells per litre PLEASE CONFIRM (mean=6.67248)
Proposition if confirmed convert CADSASIL into


Remarks
-------


*** GLYC17 ***
ID:1095
    Problem: GLYC17C == '%': it is a mistake ?
    Proposition: '%' => MMOL/L
ID:2101
    Problem: GLYC17 == 13: Non realistic value
    Proposition: suppression
IDs:1117, 1119, 1151, 1155, 1156, 1157, 1172, 1179
    Problem: GLYC17C == 'G/L'. Convertion to MG/DL (* 100)
    lead to realistic values: (mean:100.37, std:15.33, min:78.99, max:130.99)
    => values are really in G/L.
    Proposition: Simply convert into MG/DL.
GLYC17C units: set([nan, 'MG/DL'])

*** CHOLTOT17 ***
For many patients:
    Problem: CHOLTOT17C == 'MMOL/', "L" is missing.
    Proposition: MMOL/ => MMOL/L
IDs:1117, 1119, 1145, 1150, 1151, 1152, 1153, 1155, 1156, 1157, 1159, 1179, 1227
    Problem: CHOLTOT17C == 'G/L'. Convertion to MG/DL (* 100)
    lead to realistic values: (mean:201.0, std:31.307, min:155.0, max:254.0)
    => values are really in G/L.
    Proposition: Simply convert into MG/DL.
CHOLTOT17 units: set([nan, 'MG/DL'])

*** CHOLHDL17 ***
2004
    Problem: CHOLHDL17C == 'MM1.STUNDE', value==53
    Proposition: MM1.STUNDE => 'MG/DL'
IDs:1117, 1119, 1145, 1150, 1151, 1152, 1153, 1155, 1156, 1157, 1159, 1179, 1227
    Problem: CHOLHDL17C == 'G/L'. Convertion to MG/DL (* 100)
    lead to realistic values: 'mean:55.15, std:13.71, min:37.00, max:82.00'
    => values are really in G/L.
    Proposition: Simply convert into MG/DL.
CHOLHDL17 units: set([nan, 'MG/DL'])

*** CHOLLDL17 ***
For many patients:
    Problem: CHOLLDL17C == 'MMOL/', "L" is missing.
    Proposition: MMOL/ => MMOL/L
IDs:1117, 1119, 1145, 1150, 1151, 1152, 1153, 1155, 1156, 1157, 1159, 1179, 1227
    Problem: CHOLLDL17C  == 'G/L'. Convertion to MG/DL (* 100)
    lead to realistic values: 'mean:124.92, std:26.07, min:81.00, max:166.00'
    => values are really in G/L.
    Proposition: Simply convert into MG/DL.
CHOLLDL17 units: set([nan, 'MG/DL'])

*** TRIGLY17 ***
For many patients:
    Problem: TRIGLY17C == 'MMOL/', "L" is missing.
    Proposition: MMOL/ => MMOL/L
IDs:1117, 1119, 1145, 1150, 1151, 1152, 1153, 1155, 1156, 1157, 1159, 1179, 1227
    Problem: TRIGLY17C  == 'G/L'. Convertion to MG/DL (* 100)
    lead to realistic values:'mean:103.92, std:46.09, min:44.00, max:202.00'
    => values are really in G/L.
    Proposition: Simply convert into MG/DL.
TRIGLY17 units: set([nan, 'MG/DL'])

**HEMO17**
IDs:1002
    Problem: HEMO17C == 'G/L'. However, the value: 15.3 seems to be in MG/DL
    Proposition: Correct error G/L => MG/DL.
HEMO17C units: set([nan, 'G/DL'])

*** CRP17 ***
all in MG/DL
*** MIGSSAURA
MIGAAURA


Marco algo to update:
Using base_commun, please do the following:
- When Cephalees == “2" (NO), then set also Migssaura, Migaura, Cephalete, Cephaletc, Cephalautre as “2"
- When Cephalees == “1" and none of the 5 abovementioned variables is “1", then set all variables to “NA”
- When Cehpalees == “1" and at least one of the 5 is “1", then set all empty variables to “2".

  QC: No CEPHALEES and MIGSSAURA: 0
  QC: No CEPHALEES and MIGAAURA: 0
  QC: No CEPHALEES and CEPHALETE: 0
  QC: No CEPHALEES and CEPHALETC: 0
  QC: No CEPHALEES and CEPHALEAUTRE: 0
When Cephalees == “2" (NO), then set also Migssaura, Migaura, Cephalete, Cephaletc, Cephalautre as "2".
When Cephalees == “1" and none of the 5 abovementioned variables is “1", then set all variables to “NA”.
Nb time this case occure: 0
When Cehpalees == “1" and at least one of the 5 is “1", then set all empty variables to “2".
Nb time this case occure: 266
MIGSSAURA values: set([1.0, 2.0])
MIGAAURA values: set([1.0, 2.0])
CEPHALETE values: set([1.0, 2.0])
CEPHALETC values: set([1.0, 2.0])
CEPHALEAUTRE values: set([1.0, 2.0])


*** MIGSSAURA26 ***
*** MIGAAURA26 ***

  QC: No CEPHALEES26 and MIGSSAURA26: 0
  QC: No CEPHALEES26 and MIGAAURA26: 0
  QC: No CEPHALEES26 and CEPHALETE26: 0
  QC: No CEPHALEES26 and CEPHALETC26: 0
  QC: No CEPHALEES26 and CEPHALEAUTRE26: 0
When Cephalees == “2" (NO), then set also Migssaura, Migaura, Cephalete, Cephaletc, Cephalautre as "2".
Nb time this case occure: 152
When Cephalees == “1" and none of the 5 abovementioned variables is “1", then set all variables to “NA”.
Nb time this case occure: 1
When Cehpalees == “1" and at least one of the 5 is “1", then set all empty variables to “2".
Nb time this case occure: 138
MIGSSAURA26 values: set([nan, 1.0, 2.0, nan, ...])
MIGAAURA26 values: set([nan, 1.0, 2.0, nan, ...])
CEPHALETE26 values: set([nan, 1.0, 2.0,  nan, ...])
CEPHALETC26 values: set([nan, nan, 2.0, nan, 1.0,  nan, ...])
CEPHALEAUTRE26 values: set([nan, 1.0, 2.0,  nan, ...])

***
MIGSSAURA39
MIGAAURA39

  QC: No CEPHALEES39 and MIGSSAURA39: 0
  QC: No CEPHALEES39 and MIGAAURA39: 0
  QC: No CEPHALEES39 and CEPHALETE39: 0
  QC: No CEPHALEES39 and CEPHALETC39: 0
  QC: No CEPHALEES39 and CEPHALEAUTRE39: 0
When Cephalees == “2" (NO), then set also Migssaura, Migaura, Cephalete, Cephaletc, Cephalautre as "2".
Nb time this case occure: 134
When Cephalees == “1" and none of the 5 abovementioned variables is “1", then set all variables to “NA”.
Nb time this case occure: 0
When Cehpalees == “1" and at least one of the 5 is “1", then set all empty variables to “2".
Nb time this case occure: 102
MIGSSAURA39 values: set([nan, 1.0, 2.0,  nan, ...])
MIGAAURA39 values: set([nan, 1.0, 2.0,  nan, ...])
CEPHALETE39 values: set([nan, 1.0, 2.0,  nan, ...])
CEPHALETC39 values: set([nan, 1.0, 2.0,  nan, ...])
CEPHALEAUTRE39 values: set([nan, nan, 2.0, nan, 1.0,  nan, ...)

**LEUCO17**

ID:Many subjects
    Problem: Unit is "10E" seems to be compatible with G/L
    Proposition: Correct error 10E => G/L ?

ID: CADASIL vs ASPS
    Problem: Unit are clearly not compatible.
    DB_Mapping_Longit_Last_EJ_2013-05-08 indicates "same unit" ie.: leukocutes 
    per microliter (mcL). However CADASIL seems to use G/L
    mean CADASIL = 6.67 mean ASPS=5991.79. Clearly  ASPS reaaly use count per
    per microliter. So do you have the leukocyte mass or any rule to convert
    G/L to count per microliter ?

nan mean:nan, std:nan, min:nan, max:nan
G/L mean:7.00, std:1.99, min:3.40, max:18.00
% mean:5.70, std:0.00, min:5.70, max:5.70
10E mean:6.51, std:3.32, min:3.20, max:49.90

**FIBRINO17**
nan mean:nan, std:nan, min:nan, max:nan
G/L mean:3.42, std:0.74, min:1.98, max:6.37
MG/DL mean:337.00, std:75.65, min:20.60, max:547.00
NF mean:nan, std:nan, min:nan, max:nan
FIBRINO17C units: set([nan, 'G/L'])

Save cadasil
/neurospin/mescog/clinic/base_commun_20140109.csv
