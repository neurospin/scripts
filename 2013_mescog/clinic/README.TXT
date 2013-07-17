QC
==

CDADASIL
--------
For each variable in base_commun.csv (378 samples):
in_france2012   : is the variable in france2012.csv (249 samples)
diff            : if numeric the maximum difference, if symbolic the number of diff
n_missing       : nb of missing values (in base_commun.csv)
n_missing_base_commun_but_not_infr2012 : nb of missing values (in base_commun.csv) but not in france2012.csv


MERGING BASES
=============
ADASIL
-------

/neurospin/mescog/clinic
- base_commun.csv from base_commun.xls : M0 18 and 36 for CADASIL both Munich & Paris
- france2012.csv from france2012.xls: M0 18 and 36 54 for Paris

ASPS
----
Stefan R: I could not find the clinical data in the archive you gave us at Munich, could you send it to me please ?

COMON VRIABLES
--------------

CADASIL
~~~~~~~
Marco: Does "AGE_AT_INCLUSION" correspond to "Ts" column in "base_commun.xlsx" ?
Eric :  Should "DATEINCL" and "DATENAIS" be taken in "france2012.xls" ?

All of you: since we have few shared follow up variables (SYS_BP, DIA_BP, WEIGHT, BMI) I will suffix them by "__M36".

              NEW_NAME                CADASIL_NAME             ASPS_NAME  CADASIL_COL
0     AGE_AT_INCLUSION  FORM = (DATEINCL-DATENAIS)                 ALTER           -1 => If Ts missing use DATEINCL-DATENAIS from france2012
1               SYS_BP                         PAS               MEANSYS          325
2          SYS_BP__M36                       PAS40              MEANSYS1          935
3               DIA_BP                         PAD               MEANDIA          326
4          DIA_BP__M36                       PAD40              MEANDIA1          936
5                  SEX                        SEXE                   SEX            5
6               HEIGHT                      TAILLE               GROESSE          324
7               WEIGHT                       POIDS               GEWICHT          323
8          WEIGHT__M36                     POIDS40              GEWICHT1          934
9     MIGRAINE_WO_AURA                   MIGSSAURA               MIGOFNS           79
10  MIGRAINE_WITH_AURA                    MIGAAURA               MIGMFNS           80
11        MIGRAINE_AGE                AGEMIGRAINES             ANFALLVON           87
12       MIGRAINE_FREQ               FREQMIGRAINES              FREQUMTH           88
13        MIGRAINE_MED             SEE MEDICATIONS                 MEDTH           -1 => POIDS/TAILLE^2
14                 BMI  TO DEFINE (POIDS/TAILLE^2)                   BMI           -1 => POIDS40/TAILLE^2
15            BMI__M36            POIDS40/TAILLE^2                  BMI1           -1
16             SMOKING                       TABAC                  NIKO           51
17             ALCOHOL                     CALCOOL               ALCOHOL           55
18        HYPERTENSION                         HTA          HYPERTENSION           56
19            DIABETES                        DIAB              DIABETES           62
20           HYPERCHOL                     HCHOLES  HYPERCHOLESTEROLEMIA           59
21     PERIPH_VASC_DIS                  ARTERITEMI       VASCULARDISEASE           39
22          VENOUS_DIS                PATHOTHROMBO        EMBOLICDESEASE           40
23           FAST_GLUC                      GLYC17               BLZWERT          489
24               HBA1C                   HBGLYCO17                 HBA1C          487
25                CHOL                   CHOLTOT17                  CHOL          481
26                 HDL                   CHOLHDL17                   HDL          477
27                 LDL                   CHOLLDL17                   LDL          479
28              TRIGLY                    TRIGLY17                   TRI          483
29          HEMOGLOBIN                      HEMO17               HAEMOGL          469
30         LEUKO_COUNT                     LEUCO17                 LEUKO          471
31       THROMBO_COUNT                      PLAQ17               THROMBO          473
32          FIBRINOGEN                   FIBRINO17                   FIB          493
33             AF_HIST                FAURICULAIRE               VORHBEK           38
