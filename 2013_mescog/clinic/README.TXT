FILES HISTORY
=============

DB_Mapping.*
------------

DB_Mapping_Longit_Last_EJ_2013-05-08.xlx <=> DB_Mapping_Longit_Last_EJ_2013-05-08.csv
DB_Mapping_Longit_Last_EJ_20131007
    EMBOLICDESEASE => EMBOLICDISEASE

CADASIL
-------
- base_commun.xlsx
    => base_commun.csv
    Add DATEINCL and DATENAIS in CADASIL
    => base_commun_20131008.csv. See 00_MISSING_CADASIL.py

- france2012.csv

- CAD_Munich_Dates.txt
  Date birth/inclusion for munich patients

ASP(F)S_klinVariables_*
---------------------
ASP(F)S_klinVariables_*.sav received from Hofer Edith => ASP(F)S_klinVariables_*.csv

COMON DATABASE
--------------

- db_clinic_cadasil-asps-common.*
- db_clinic_cadasil-asps_mapping_summary_*


MISSING DATA
============

Add DATEINCL and DATENAIS an compute AGE_AT_INCLUSION in CADASIL subjects

SCRIPT
------

00_MISSING_CADASIL.py

INPUT
-----

"base_commun.csv"
"france2012.csv" => date DATEINCL and DATENAIS for french
"CAD_Munich_Dates.txt" => date DATEINCL and DATENAIS for german

OUTPUT
------

"base_commun_20131003.csv" == "base_commun.csv" + Date (from "france2012.csv" + CAD_Munich_Dates.txt)

QC for CDADASIL
===============
Check if france2012 and base_commun_20131003.csv are simillar

SCRIPT
------

00_QC_CADASIL.py

INPUT
-----
france2012.csv
base_commun_20131003.csv

OUTPUT
------
/neurospin/mescog/clinic/QC/cadasil_qc.csv
/neurospin/mescog/clinic/QC/cadasil_qc.html

where:
For each variable in base_commun.csv (378 samples):
in_france2012   : is the variable in france2012.csv (249 samples)
diff            : if numeric the maximum difference, if symbolic the number of diff
n_missing       : nb of missing values (in base_commun.csv)
n_missing_base_commun_but_not_infr2012 : nb of missing values (in base_commun.csv) but not in france2012.csv



MERGING BASES
=============

SCRIPT
------

01_merge_CADASIL-ASPS.py


INPUT
------

1) base_commun_20131003.csv
2) ASPS_klinVariables_20130806.csv
3) DB_Mapping_Longit_Last_EJ_20131007.csv

OUTPUT
------

1) Summary of common DB
    "db_clinic_cadasil-asps_mapping_summary.csv"
    "db_clinic_cadasil-asps_mapping_summary.html"
2) Common DB
    "db_clinic_cadasil-asps-common.csv"

================================================================================
TO BE CHECKED

See blue cells in db_clinic_cadasil-asps_mapping_summary_20130620.xls

AGE_AT_INCLUSION
    ASPS_min = 0
=> One ASPS "AGE_AT_INCLUSION" is == 0.

MIGRAINE_*:
CADASIL, Every non missing are "yes" (1), no false found,
=> is it normal ?

FAST_GLUC:
Very different MEAN and STD, I checked the conversion from  mmol/l to mg/dl, it seems ok.
=> Check "RECODING" column

Details:
mmoll_to_mgdl(cadasil_base[cadasil_name], mm=180)
Where mmoll_to_mgdl is the function:
def mmoll_to_mgdl(mmolpl, mm):
    return mmolpl * mm / 10.0

CHOL, LDL*, HDL*:
Again, MEAN and STD are different, I checked the conversion from  mmol/l to mg/dl, it seems ok.
=> Check RECODING column, molar mass of CHOL=386.65

PROBABLE ERRORS

See the red cell in db_clinic_cadasil-asps_mapping_summary_20130620.xls

LEUKO_COUNT
The ASPS and CADASIL units are supposed to be the same. However, the means and std are **very** different:
mean cadasil=6.67248, while mean ASPS=5991.7923115327

MAil de Marco
============
     I think I might know the reason for some strange lab values. Within the CADASIL cohort, units between Paris and Munich can differ. E.g. glucose is recorded in mM at Paris, but mg/dl in Munich. An exception is subject 2101, which was also recorded in mM. 
     It seems that Munich values have also been converted, this is why there are some sky high values (up to 4000) for Munich subjects.
Yes, but Glyc17c indicates MG/DL what is the correct unit for this subject according to you ?

=> I propose:
1) To adapt the conversion according to the unit mentioned column "Glyc17c" in base_commun.xls file. This should fix most of the problem like Munich vs Paris unit, and some exceptions (Paris subjects 1117, 1119, 1155, 1156, 1157 and Munich 2101).
2) Recompute the quality control scores to detect some outliers as the subject 2124 (the one with sky high value).
3) Get a feedback from you, Stefan and Eric


    The same seems to be true for cholesterol (Paris mM, Munich mg/dl).

I propose the same procedure:
1) Look <VARIABLE_NAME>c example CHOLTOT17c
2) quality control => outliers
3) feedback


    Do you have the complete table with units? If not, I can send it to you.

Yes I would like. However it seems that I can use <VARIABLE_NAME>c. Is it correct ?

 

    P.S. Making some very fast plausibility checks, we found that there are more problems, i.e. errors in the CADASIL database. 


    Look at glucose values for subjects 1155, 1156, 1157. The values make no sense (way too low). I looked in the original file "base_commun" Excel file.
    Here for these subjects the unit is given as G/L, which does not make sense for glucose.

=> This will be fixed by the procedure evoked previously

    in my base_commun the Glyc17c for 2101 is µMOL/L. Still, the value of 13 is also not plausible, even with µM. I suggest to delete this value.
=> OK

For the rest, the procedure suggested by you is good.

Still, I think glucose for 1155, 1156 and 1157 make no sense. This has to be checked by the Paris group.


base_commun_20131008.tsv
MMOL/L
col:GLYC17C l:96 % => MMOL/L


================================================================================
**GLYC17**
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
GLYC17C units: set([nan, 'MG/DL', 'MMOL/L'])
Paris unit: set([nan, 'MG/DL', 'MMOL/L'])
Munich unit: set([nan, 'MG/DL'])
**CHOLTOT17**
For many patients:
    Problem: CHOLTOT17C == 'MMOL/', "L" is missing.
    Proposition: MMOL/ => MMOL/L
IDs:1117, 1119, 1145, 1150, 1151, 1152, 1153, 1155, 1156, 1157, 1159, 1179, 1227
    Problem: CHOLTOT17C == 'G/L'. Convertion to MG/DL (* 100)
    lead to realistic values: (mean:201.0, std:31.307, min:155.0, max:254.0)
    => values are really in G/L.
    Proposition: Simply convert into MG/DL.
CHOLTOT17 units: set([nan, 'MG/DL', 'MMOL/L'])
**CHOLHDL17**
2004
    Problem: CHOLHDL17C == 'MM1.STUNDE', value==53
    Proposition: MM1.STUNDE => 'MG/DL'
IDs:1117, 1119, 1145, 1150, 1151, 1152, 1153, 1155, 1156, 1157, 1159, 1179, 1227
    Problem: CHOLHDL17C == 'G/L'. Convertion to MG/DL (* 100)
    lead to realistic values: 'mean:55.15, std:13.71, min:37.00, max:82.00'
    => values are really in G/L.
    Proposition: Simply convert into MG/DL.
CHOLHDL17 units: set([nan, 'MG/DL', 'MMOL/L'])
**CHOLLDL17**
For many patients:
    Problem: CHOLLDL17C == 'MMOL/', "L" is missing.
    Proposition: MMOL/ => MMOL/L
IDs:1117, 1119, 1145, 1150, 1151, 1152, 1153, 1155, 1156, 1157, 1159, 1179, 1227
    Problem: CHOLLDL17C  == 'G/L'. Convertion to MG/DL (* 100)
    lead to realistic values: 'mean:124.92, std:26.07, min:81.00, max:166.00'
    => values are really in G/L.
    Proposition: Simply convert into MG/DL.
CHOLLDL17 units: set([nan, 'MG/DL', 'MMOL/L'])
**TRIGLY17**
For many patients:
    Problem: TRIGLY17C == 'MMOL/', "L" is missing.
    Proposition: MMOL/ => MMOL/L
IDs:1117, 1119, 1145, 1150, 1151, 1152, 1153, 1155, 1156, 1157, 1159, 1179, 1227
    Problem: TRIGLY17C  == 'G/L'. Convertion to MG/DL (* 100)
    lead to realistic values:'mean:103.92, std:46.09, min:44.00, max:202.00'
    => values are really in G/L.
    Proposition: Simply convert into MG/DL.
TRIGLY17 units: set([nan, 'MG/DL', 'MMOL/L'])
**HEMO17**
IDs:1002
    Problem: HEMO17C == 'G/L'. However, the value: 15.3 seems to be in MG/DL
    Proposition: Correct error G/L => MG/DL.
HEMO17C units: set([nan, 'G/DL'])
**LEUCO17**
nan mean:nan, std:nan, min:nan, max:nan
G/L mean:7.00, std:1.99, min:3.40, max:18.00
% mean:5.70, std:0.00, min:5.70, max:5.70
10E mean:6.51, std:3.32, min:3.20, max:49.90
================================================================================
