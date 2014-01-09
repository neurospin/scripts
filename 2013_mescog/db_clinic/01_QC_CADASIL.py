# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 13:39:45 2013

@author: ed203246

QC for CDADASIL
===============
Check if france2012 and base_commun_20131003.csv are simillar
Manualy correct some mistakes (unit and values)

SCRIPT
------

01_QC_CADASIL.py

INPUT
-----
france2012.csv
base_commun_20140109.csv

OUTPUT
------
QC/cadasil_qc.csv
QC/cadasil_qc.html
base_commun_20140109.csv

where:
For each variable in base_commun.csv (378 samples):
in_france2012   : is the variable in france2012.csv (249 samples)
diff            : if numeric the maximum difference, if symbolic the number of diff
n_missing       : nb of missing values (in base_commun.csv)
n_missing_base_commun_but_not_infr2012 : nb of missing values (in base_commun.csv) but not in france2012.csv

"""

import os.path
import pandas as pd
import numpy as np

## I/O
WD = "/neurospin/mescog"
INPUT_cadasil_base_commun_filepath = os.path.join(WD, "clinic", "base_commun_20140109.csv")
INPUT_cadasil_france2012_filepath = os.path.join(WD, "clinic", "france2012.csv")
OUTPUT_cadasil_qc = os.path.join(WD, "clinic", "QC", "cadasil_qc")
OUTPUT_cadasil_base_commun_filepath = os.path.join(WD, "clinic", "base_commun_20140109.csv")

cadasil_base_commun = pd.read_table(INPUT_cadasil_base_commun_filepath, header=0, sep=",").replace("-", np.nan)
cadasil_france2012 = pd.read_table(INPUT_cadasil_france2012_filepath, header=0).replace("-", np.nan)

## ================================================================================
# 1, 2, 3) Look for differences between cadasil_base_commun & cadasil_france2012
cadasil_base_commun.columns = [s.upper() for s in cadasil_base_commun.columns]
cadasil_france2012.columns = [s.upper() for s in cadasil_france2012.columns]

qc_cada_common_var = list()
for i in xrange(len(cadasil_base_commun.columns)):
    var = cadasil_base_commun.columns[i].upper()
    if var == 'ID':
        continue
    in_france2012 = 0
    n_missing = diff = n_missing_base_commun_but_not_infr2012 = 0
    if var in cadasil_france2012.columns:
        in_france2012 = 1
    base_com = cadasil_base_commun[['ID', var]]
    n_missing = base_com[var].isnull().sum()
    if n_missing == base_com[var].shape[0]:
        continue
    if in_france2012:
        fr2012 = cadasil_france2012[['ID', var]]
        n_france2012 = fr2012.shape[0]
        #merge = pd.merge(base_com, fr2012)
        merge = pd.merge(base_com, fr2012, on="ID", suffixes=["_base_commun", "_france2012"])
        n_missing_base_commun_but_not_infr2012 = np.sum(merge.icol(1).isnull() & merge.icol(2).notnull())
        try:
            diff = np.max(np.abs(merge.icol(1) - merge.icol(2)))
        except:
            try:
                merge = merge.fillna("")
                diff = np.sum(merge.icol(1) != merge.icol(2))
            except:
                diff = "NOT COMPARABLE"
    qc_cada_common_var.append((var, in_france2012,
                               diff, n_missing, n_missing_base_commun_but_not_infr2012))

qc_cada_common_var = pd.DataFrame(qc_cada_common_var,
    columns=['VAR', 'in_france2012', 'diff', 'n_missing', 'n_missing_base_commun_but_not_infr2012'])

print qc_cada_common_var.to_string()

qc_cada_common_var.to_csv(OUTPUT_cadasil_qc+".csv", sep=",", index=False)
qc_cada_common_var.to_html(OUTPUT_cadasil_qc+".html")


## ================================================================================
# 4) Mannualy correct some mistakes (unit and values) => base_commun_20131009.csv

def to_mgdl(x, unit, mm):
    out = list()
    for i in xrange(len(unit)):
        #unit[i], x[i]
        if pd.isnull(x[i]):
            mgdl = np.nan
        elif unit[i] == 'G/L':
            mgdl = x[i] * 100.
        elif unit[i] == 'MG/DL':
            mgdl = x[i]
        elif unit[i] == '\xc2\xb5MOL/L':  #µMOL/L
            mgdl = x[i] * mm / 100.0
        elif unit[i] == 'MMOL/L':
            mgdl = x[i] * mm / 10.0
        else:
            raise ValueError("??%s %i" % (unit[i], i))
        out.append(mgdl)
    return out

def set_nullunit_for_missing_value(d, val_col, unit_col):
    for i in xrange(d.shape[0]):
        if pd.isnull(d[val_col][i]):
            d[unit_col][i] = np.nan
    return d

def stats(x):
    return 'mean:%.2f, std:%.2f, min:%.2f, max:%.2f' % (np.mean(x), np.std(x), np.min(x), np.max(x))

d = cadasil_base_commun

# Check and fix unit and value ====================================================
print "*** GLYC17 ***"

print """ID:1095
    Problem: GLYC17C == '%': it is a mistake ?
    Proposition: '%' => MMOL/L"""
d.ID[d.GLYC17C == '%'].tolist()
#1095
if np.where(d.GLYC17C == '%')[0]==94:
    d.GLYC17C[94] = 'MMOL/L'

print """ID:2101
    Problem: GLYC17 == 13: Non realistic value
    Proposition: suppression"""
# Marco: in my base_commun the Glyc17c for 2101 is µMOL/L. Still, the value of 13 is also not plausible, even with µM. I suggest to delete this value.
d.ID[d.GLYC17 == 13].tolist()
if np.where(d.GLYC17 == 13)[0] == 349:
    d.GLYC17[349] = np.nan

print """IDs:1117, 1119, 1151, 1155, 1156, 1157, 1172, 1179
    Problem: GLYC17C == 'G/L'. Convertion to MG/DL (* 100)
    lead to realistic values: (mean:100.37, std:15.33, min:78.99, max:130.99)
    => values are really in G/L.
    Proposition: Simply convert into MG/DL."""

d.ID[d.GLYC17C == 'G/L'].tolist()
#'mean:100.37, std:15.34, min:79.00, max:131.00'
stats(d.GLYC17[d.GLYC17C == 'G/L'] * 100)
#'mean:100.37, std:15.34, min:79.00, max:131.00'
stats(d.GLYC17[d.GLYC17C == 'MG/DL'])
#'mean:111.09, std:42.63, min:78.00, max:228.00'
# => Seems OK recode G/L => MG/L

d.GLYC17[d.GLYC17C == 'G/L'] *= 100
d.GLYC17C[d.GLYC17C == 'G/L'] = 'MG/DL'

# Marco Still, I think glucose for 1155, 1156 and 1157 make no sense. This has to be checked by the Paris group.

d = set_nullunit_for_missing_value(d, "GLYC17", "GLYC17C")
# conv all to MG/DL
d.GLYC17 = to_mgdl(x=d.GLYC17, unit=d.GLYC17C, mm=180)
d.GLYC17C[d.GLYC17C == 'MMOL/L'] = 'MG/DL'

print "GLYC17C units:", set(d.GLYC17C)


print "*** CHOLTOT17 ***"

print """For many patients:
    Problem: CHOLTOT17C == 'MMOL/', "L" is missing.
    Proposition: MMOL/ => MMOL/L"""
#IDs: [1117, 1119, 1145, 1150, 1151, 1152, 1153, 1155, 1156, 1157, 1159, 1179, 1227]
# Fix error in unit MMOL/ instead of MMOL/L
d.CHOLTOT17C[d.CHOLTOT17C == 'MMOL/'] = 'MMOL/L'

print """IDs:1117, 1119, 1145, 1150, 1151, 1152, 1153, 1155, 1156, 1157, 1159, 1179, 1227
    Problem: CHOLTOT17C == 'G/L'. Convertion to MG/DL (* 100)
    lead to realistic values: (mean:201.0, std:31.307, min:155.0, max:254.0)
    => values are really in G/L.
    Proposition: Simply convert into MG/DL."""

d.ID[d.CHOLTOT17C == 'G/L'].tolist()
stats(d.CHOLTOT17[d.CHOLTOT17C == 'G/L'] * 100)
#'mean:201.00, std:31.31, min:155.00, max:254.00'
stats(d.CHOLTOT17[d.CHOLTOT17C == 'MG/DL'])
#'mean:210.78, std:43.44, min:0.50, max:338.00'
# => Seems OK recode G/L => MG/L
d.CHOLTOT17[d.CHOLTOT17C == 'G/L'] *= 100
d.CHOLTOT17C[d.CHOLTOT17C == 'G/L'] = 'MG/DL'
d = set_nullunit_for_missing_value(d, "CHOLTOT17", "CHOLTOT17C")
# conv all to MG/DL
d.CHOLTOT17 = to_mgdl(x=d.CHOLTOT17, unit=d.CHOLTOT17C, mm=386.65)
d.CHOLTOT17C[d.CHOLTOT17C == 'MMOL/L'] = 'MG/DL'
print "CHOLTOT17 units:", set(d.CHOLTOT17C)

print "*** CHOLHDL17 ***"

print """2004
    Problem: CHOLHDL17C == 'MM1.STUNDE', value==53
    Proposition: MM1.STUNDE => 'MG/DL'"""

d.ID[d.CHOLHDL17C == 'MM1.STUNDE']
d.CHOLHDL17[d.CHOLHDL17C == 'MM1.STUNDE'] # 53
stats(d.CHOLHDL17[d.CHOLHDL17C == 'MG/DL'])
#'mean:57.12, std:16.30, min:24.00, max:117.00'
# OK MM1.STUNDE => 'MG/DL'
d.CHOLHDL17C[d.CHOLHDL17C == 'MM1.STUNDE'] = 'MG/DL'

print """IDs:1117, 1119, 1145, 1150, 1151, 1152, 1153, 1155, 1156, 1157, 1159, 1179, 1227
    Problem: CHOLHDL17C == 'G/L'. Convertion to MG/DL (* 100)
    lead to realistic values: 'mean:55.15, std:13.71, min:37.00, max:82.00'
    => values are really in G/L.
    Proposition: Simply convert into MG/DL."""

d.ID[d.CHOLHDL17C == 'G/L'].tolist()
#[1117, 1119, 1145, 1150, 1151, 1152, 1153, 1155, 1156, 1157, 1159, 1179, 1227]
stats(d.CHOLHDL17[d.CHOLHDL17C == 'G/L'] * 100)
#'mean:55.15, std:13.71, min:37.00, max:82.00'
stats(d.CHOLHDL17[d.CHOLHDL17C == 'MG/DL'])
#'mean:57.09, std:16.24, min:24.00, max:117.00'
# => Seems OK recode G/L => MG/L
d.CHOLHDL17[d.CHOLHDL17C == 'G/L'] *= 100
d.CHOLHDL17C[d.CHOLHDL17C == 'G/L'] = 'MG/DL'
d = set_nullunit_for_missing_value(d, "CHOLTOT17", "CHOLTOT17C")
# conv all to MG/DL
d.CHOLHDL17 = to_mgdl(x=d.CHOLHDL17, unit=d.CHOLHDL17C, mm=386.65)
d.CHOLHDL17C[d.CHOLHDL17C == 'MMOL/L'] = 'MG/DL'
print "CHOLHDL17 units:", set(d.CHOLHDL17C)


print "*** CHOLLDL17 ***"

print """For many patients:
    Problem: CHOLLDL17C == 'MMOL/', "L" is missing.
    Proposition: MMOL/ => MMOL/L"""
#IDs: [1117, 1119, 1145, 1150, 1151, 1152, 1153, 1155, 1156, 1157, 1159, 1179, 1227]
# Fix error in unit MMOL/ instead of MMOL/L
d.CHOLLDL17C[d.CHOLLDL17C == 'MMOL/'] = 'MMOL/L'


print """IDs:1117, 1119, 1145, 1150, 1151, 1152, 1153, 1155, 1156, 1157, 1159, 1179, 1227
    Problem: CHOLLDL17C  == 'G/L'. Convertion to MG/DL (* 100)
    lead to realistic values: 'mean:124.92, std:26.07, min:81.00, max:166.00'
    => values are really in G/L.
    Proposition: Simply convert into MG/DL."""
d.ID[d.CHOLLDL17C == 'G/L'].tolist()
#[1117, 1119, 1145, 1150, 1151, 1152, 1153, 1155, 1156, 1157, 1159, 1179, 1227]
stats(d.CHOLLDL17[d.CHOLLDL17C == 'G/L'] * 100)
stats(d.CHOLLDL17[d.CHOLLDL17C == 'MG/DL'])
# => Seems OK recode G/L => MG/L
d.CHOLLDL17[d.CHOLLDL17C == 'G/L'] *= 100
d.CHOLLDL17C[d.CHOLLDL17C == 'G/L'] = 'MG/DL'
d = set_nullunit_for_missing_value(d, "CHOLTOT17", "CHOLTOT17C")
# conv all to MG/DL
d.CHOLLDL17 = to_mgdl(x=d.CHOLLDL17, unit=d.CHOLLDL17C, mm=386.65)
d.CHOLLDL17C[d.CHOLLDL17C == 'MMOL/L'] = 'MG/DL'
print "CHOLLDL17 units:", set(d.CHOLLDL17C)


print "*** TRIGLY17 ***"

print """For many patients:
    Problem: TRIGLY17C == 'MMOL/', "L" is missing.
    Proposition: MMOL/ => MMOL/L"""
# Fix error in unit MMOL/ instead of MMOL/L
d.TRIGLY17C[d.TRIGLY17C == 'MMOL/'] = 'MMOL/L'


print """IDs:1117, 1119, 1145, 1150, 1151, 1152, 1153, 1155, 1156, 1157, 1159, 1179, 1227
    Problem: TRIGLY17C  == 'G/L'. Convertion to MG/DL (* 100)
    lead to realistic values:'mean:103.92, std:46.09, min:44.00, max:202.00'
    => values are really in G/L.
    Proposition: Simply convert into MG/DL."""
d.ID[d.TRIGLY17C == 'G/L'].tolist()
#[1117, 1119, 1145, 1150, 1151, 1152, 1153, 1155, 1156, 1157, 1159, 1179, 1227]
stats(d.TRIGLY17[d.TRIGLY17C == 'G/L'] * 100)
#'mean:103.92, std:46.09, min:44.00, max:202.00'
stats(d.TRIGLY17[d.TRIGLY17C == 'MG/DL'])
#'mean:150.08, std:95.73, min:32.00, max:685.00'
# => Seems OK recode G/L => MG/L
d.TRIGLY17[d.TRIGLY17C == 'G/L'] *= 100
d.TRIGLY17C[d.TRIGLY17C == 'G/L'] = 'MG/DL'
d = set_nullunit_for_missing_value(d, "CHOLTOT17", "CHOLTOT17C")
# conv all to MG/DL
d.TRIGLY17 = to_mgdl(x=d.TRIGLY17, unit=d.TRIGLY17C, mm=875.)
d.TRIGLY17C[d.TRIGLY17C == 'MMOL/L'] = 'MG/DL'
print "TRIGLY17 units:", set(d.TRIGLY17C)

print "**HEMO17**"
print """IDs:1002
    Problem: HEMO17C == 'G/L'. However, the value: 15.3 seems to be in MG/DL
    Proposition: Correct error G/L => MG/DL."""

d.ID[d.HEMO17C == 'G/L'].tolist()
#[1002]
d.HEMO17[d.HEMO17C == 'G/L']
#15.3
stats(d.HEMO17[d.HEMO17C == 'G/DL'])
#'mean:14.20, std:1.29, min:9.20, max:18.40'
# just an error => recode 'G/L' => G/DL
d.HEMO17C[d.HEMO17C == 'G/L'] = 'G/DL'
print "HEMO17C units:", set(d.HEMO17C)

print "*** CRP17 ***"
print "all in MG/DL"
d.CRP17[d.CRP17C == 'MG/L'] /= 10 
d.CRP17C[d.CRP17C == 'MG/L'] = 'MG/DL'

print """*** MIGSSAURA
MIGAAURA
"""

print """
Marco algo to update:
Using base_commun, please do the following:
- When Cephalees == “2" (NO), then set also Migssaura, Migaura, Cephalete, Cephaletc, Cephalautre as “2"
- When Cephalees == “1" and none of the 5 abovementioned variables is “1", then set all variables to “NA”
- When Cehpalees == “1" and at least one of the 5 is “1", then set all empty variables to “2".
"""

print "  QC: No CEPHALEES and MIGSSAURA:", np.sum((d.CEPHALEES == 2) & (d.MIGSSAURA == 1))
print "  QC: No CEPHALEES and MIGAAURA:", np.sum((d.CEPHALEES == 2) & (d.MIGAAURA == 1))
print "  QC: No CEPHALEES and CEPHALETE:", np.sum((d.CEPHALEES == 2) & (d.CEPHALETE == 1))
print "  QC: No CEPHALEES and CEPHALETC:", np.sum((d.CEPHALEES == 2) & (d.CEPHALETC == 1))
print "  QC: No CEPHALEES and CEPHALEAUTRE:", np.sum((d.CEPHALEES == 2) & (d.CEPHALEAUTRE == 1))

print """When Cephalees == “2" (NO), then set also Migssaura, Migaura, Cephalete, Cephaletc, Cephalautre as "2"."""
d.MIGSSAURA[(d.CEPHALEES == 2)]    = 2
d.MIGAAURA[(d.CEPHALEES == 2)]     = 2
d.CEPHALETE[(d.CEPHALEES == 2)]    = 2
d.CEPHALETC[(d.CEPHALEES == 2)]    = 2
d.CEPHALEAUTRE[(d.CEPHALEES == 2)] = 2

print """When Cephalees == “1" and none of the 5 abovementioned variables is “1", then set all variables to “NA”."""
cephalees_but_nothing_else = (d.CEPHALEES == 1) & (d.MIGSSAURA != 1) & (d.MIGAAURA != 1) & (d.CEPHALETE !=1) & (d.CEPHALETC !=1)  & (d.CEPHALEAUTRE !=1)
print "Nb time this case occure:", np.sum(cephalees_but_nothing_else) 
d.MIGSSAURA[cephalees_but_nothing_else]    = np.nan
d.MIGAAURA[cephalees_but_nothing_else]     = np.nan
d.CEPHALETE[cephalees_but_nothing_else]    = np.nan
d.CEPHALETC[cephalees_but_nothing_else]    = np.nan
d.CEPHALEAUTRE[cephalees_but_nothing_else] = np.nan

print """When Cehpalees == “1" and at least one of the 5 is “1", then set all empty variables to “2"."""

cephalees_and_any_other = (d.CEPHALEES == 1) & ((d.MIGSSAURA == 1) | (d.MIGAAURA == 1) | (d.CEPHALETE ==1) | (d.CEPHALETC ==1) | (d.CEPHALEAUTRE ==1))
print "Nb time this case occure:", np.sum(cephalees_and_any_other) 
d.MIGSSAURA[cephalees_and_any_other    & pd.isnull(d.MIGSSAURA)]    = 2
d.MIGAAURA[cephalees_and_any_other     & pd.isnull(d.MIGAAURA)]     = 2
d.CEPHALETE[cephalees_and_any_other    & pd.isnull(d.CEPHALETE)]    = 2
d.CEPHALETC[cephalees_and_any_other    & pd.isnull(d.CEPHALETC)]    = 2
d.CEPHALEAUTRE[cephalees_and_any_other & pd.isnull(d.CEPHALEAUTRE)] = 2

print "MIGSSAURA values:", set(d.MIGSSAURA)
print "MIGAAURA values:", set(d.MIGAAURA)
print "CEPHALETE values:", set(d.CEPHALETE)
print "CEPHALETC values:", set(d.CEPHALETC)
print "CEPHALEAUTRE values:", set(d.CEPHALEAUTRE)

print """***
MIGSSAURA26
MIGAAURA26
"""
print "  QC: No CEPHALEES26 and MIGSSAURA26:", np.sum((d.CEPHALEES26 == 2) & (d.MIGSSAURA26 == 1))
print "  QC: No CEPHALEES26 and MIGAAURA26:", np.sum((d.CEPHALEES26 == 2) & (d.MIGAAURA26 == 1))
print "  QC: No CEPHALEES26 and CEPHALETE26:", np.sum((d.CEPHALEES26 == 2) & (d.CEPHALETE26 == 1))
print "  QC: No CEPHALEES26 and CEPHALETC26:", np.sum((d.CEPHALEES26 == 2) & (d.CEPHALETC26 == 1))
print "  QC: No CEPHALEES26 and CEPHALEAUTRE26:", np.sum((d.CEPHALEES26 == 2) & (d.CEPHALEAUTRE26 == 1))

print """When Cephalees == “2" (NO), then set also Migssaura, Migaura, Cephalete, Cephaletc, Cephalautre as "2"."""
print "Nb time this case occure:", np.sum(d.CEPHALEES26 == 2)
d.MIGSSAURA26[(d.CEPHALEES26 == 2)]    = 2
d.MIGAAURA26[(d.CEPHALEES26 == 2)]     = 2
d.CEPHALETE26[(d.CEPHALEES26 == 2)]    = 2
d.CEPHALETC26[(d.CEPHALEES26 == 2)]    = 2
d.CEPHALEAUTRE26[(d.CEPHALEES26 == 2)] = 2

print """When Cephalees == “1" and none of the 5 abovementioned variables is “1", then set all variables to “NA”."""
cephalees_but_nothing_else = (d.CEPHALEES26 == 1) & (d.MIGSSAURA26 != 1) & (d.MIGAAURA26 != 1) & (d.CEPHALETE26 !=1) & (d.CEPHALETC26 !=1)  & (d.CEPHALEAUTRE26 !=1)
print "Nb time this case occure:", np.sum(cephalees_but_nothing_else)
d.MIGSSAURA26[cephalees_but_nothing_else]    = np.nan
d.MIGAAURA26[cephalees_but_nothing_else]     = np.nan
d.CEPHALETE26[cephalees_but_nothing_else]    = np.nan
d.CEPHALETC26[cephalees_but_nothing_else]    = np.nan
d.CEPHALEAUTRE26[cephalees_but_nothing_else] = np.nan

print """When Cehpalees == “1" and at least one of the 5 is “1", then set all empty variables to “2"."""

cephalees_and_any_other = (d.CEPHALEES26 == 1) & ((d.MIGSSAURA26 == 1) | (d.MIGAAURA26 == 1) | (d.CEPHALETE26 ==1) | (d.CEPHALETC26 ==1) | (d.CEPHALEAUTRE26 ==1))
print "Nb time this case occure:", np.sum(cephalees_and_any_other)
d.MIGSSAURA26[cephalees_and_any_other    & pd.isnull(d.MIGSSAURA26)]    = 2
d.MIGAAURA26[cephalees_and_any_other     & pd.isnull(d.MIGAAURA26)]     = 2
d.CEPHALETE26[cephalees_and_any_other    & pd.isnull(d.CEPHALETE26)]    = 2
d.CEPHALETC26[cephalees_and_any_other    & pd.isnull(d.CEPHALETC26)]    = 2
d.CEPHALEAUTRE26[cephalees_and_any_other & pd.isnull(d.CEPHALEAUTRE26)] = 2

print "MIGSSAURA26 values:", set(d.MIGSSAURA26)
print "MIGAAURA26 values:", set(d.MIGAAURA26)
print "CEPHALETE26 values:", set(d.CEPHALETE26)
print "CEPHALETC26 values:", set(d.CEPHALETC26)
print "CEPHALEAUTRE26 values:", set(d.CEPHALEAUTRE26)

print """***
MIGSSAURA39
MIGAAURA39
"""
print "  QC: No CEPHALEES39 and MIGSSAURA39:", np.sum((d.CEPHALEES39 == 2) & (d.MIGSSAURA39 == 1))
print "  QC: No CEPHALEES39 and MIGAAURA39:", np.sum((d.CEPHALEES39 == 2) & (d.MIGAAURA39 == 1))
print "  QC: No CEPHALEES39 and CEPHALETE39:", np.sum((d.CEPHALEES39 == 2) & (d.CEPHALETE39 == 1))
print "  QC: No CEPHALEES39 and CEPHALETC39:", np.sum((d.CEPHALEES39 == 2) & (d.CEPHALETC39 == 1))
print "  QC: No CEPHALEES39 and CEPHALEAUTRE39:", np.sum((d.CEPHALEES39 == 2) & (d.CEPHALEAUTRE39 == 1))

print """When Cephalees == “2" (NO), then set also Migssaura, Migaura, Cephalete, Cephaletc, Cephalautre as "2"."""
print "Nb time this case occure:", np.sum(d.CEPHALEES39 == 2)
d.MIGSSAURA39[(d.CEPHALEES39 == 2)]    = 2
d.MIGAAURA39[(d.CEPHALEES39 == 2)]     = 2
d.CEPHALETE39[(d.CEPHALEES39 == 2)]    = 2
d.CEPHALETC39[(d.CEPHALEES39 == 2)]    = 2
d.CEPHALEAUTRE39[(d.CEPHALEES39 == 2)] = 2

print """When Cephalees == “1" and none of the 5 abovementioned variables is “1", then set all variables to “NA”."""
cephalees_but_nothing_else = (d.CEPHALEES39 == 1) & (d.MIGSSAURA39 != 1) & (d.MIGAAURA39 != 1) & (d.CEPHALETE39 !=1) & (d.CEPHALETC39 !=1)  & (d.CEPHALEAUTRE39 !=1)
print "Nb time this case occure:", np.sum(cephalees_but_nothing_else) 
d.MIGSSAURA39[cephalees_but_nothing_else]    = np.nan
d.MIGAAURA39[cephalees_but_nothing_else]     = np.nan
d.CEPHALETE39[cephalees_but_nothing_else]    = np.nan
d.CEPHALETC39[cephalees_but_nothing_else]    = np.nan
d.CEPHALEAUTRE39[cephalees_but_nothing_else] = np.nan

print """When Cehpalees == “1" and at least one of the 5 is “1", then set all empty variables to “2"."""

cephalees_and_any_other = (d.CEPHALEES39 == 1) & ((d.MIGSSAURA39 == 1) | (d.MIGAAURA39 == 1) | (d.CEPHALETE39 ==1) | (d.CEPHALETC39 ==1) | (d.CEPHALEAUTRE39 ==1))
print "Nb time this case occure:", np.sum(cephalees_and_any_other) 
d.MIGSSAURA39[cephalees_and_any_other    & pd.isnull(d.MIGSSAURA39)]    = 2
d.MIGAAURA39[cephalees_and_any_other     & pd.isnull(d.MIGAAURA39)]     = 2
d.CEPHALETE39[cephalees_and_any_other    & pd.isnull(d.CEPHALETE39)]    = 2
d.CEPHALETC39[cephalees_and_any_other    & pd.isnull(d.CEPHALETC39)]    = 2
d.CEPHALEAUTRE39[cephalees_and_any_other & pd.isnull(d.CEPHALEAUTRE39)] = 2

print "MIGSSAURA39 values:", set(d.MIGSSAURA39)
print "MIGAAURA39 values:", set(d.MIGAAURA39)
print "CEPHALETE39 values:", set(d.CEPHALETE39)
print "CEPHALETC39 values:", set(d.CEPHALETC39)
print "CEPHALEAUTRE39 values:", set(d.CEPHALEAUTRE39)


print "**LEUCO17**"
"""ID:1083
    Problem: unit is '%', value is 5.70 compatible with G/L (mean:7.00, std:1.99)
    Proposition: Correct error % => G/L"""
print """
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
"""
d.ID[d.LEUCO17C == '%']
d.ID[d.LEUCO17C == '10E']

#1083
for v in set(d.LEUCO17C):
    print v, stats(d.LEUCO17[d.LEUCO17C == v])
#nan mean:nan, std:nan, min:nan, max:nan
#G/L mean:7.00, std:1.99, min:3.40, max:18.00
#% mean:5.70, std:0.00, min:5.70, max:5.70
#10E mean:6.51, std:3.32, min:3.20, max:49.90

## % seems to be an error => recode 10E
#d.LEUCO17C[d.LEUCO17C == '%'] = 'G/L'
#
#"""ID:1001 ... 1250
#    Problem: unit is "10E". Values (mean:6.51, std:3.32, min:3.20, max:49.90)
#    seem to be compatible with G/L.
#    Proposition: Correct error 10E => G/L"""
#d.LEUCO17C[d.LEUCO17C == '10E'] = 'G/L'


print "**FIBRINO17**"
for v in set(d.FIBRINO17C):
    print v, stats(d.FIBRINO17[d.FIBRINO17C == v])
#    nan mean:nan, std:nan, min:nan, max:nan
#G/L mean:3.42, std:0.74, min:1.98, max:6.37
#MG/DL mean:337.00, std:75.65, min:20.60, max:547.00
#NF mean:nan, std:nan, min:nan, max:nan
# Some are in already "MG/DL", G/L ecpected convert
d.FIBRINO17[d.FIBRINO17C == "MG/DL"] /= 100
d.FIBRINO17C[d.FIBRINO17C == "MG/DL"] = 'G/L'
d = set_nullunit_for_missing_value(d, "FIBRINO17", "FIBRINO17C")

print "FIBRINO17C units:", set(d.FIBRINO17C)

# SAVE
print "Save cadasil\n%s" % OUTPUT_cadasil_base_commun_filepath
d.to_csv(OUTPUT_cadasil_base_commun_filepath, sep=",", index=False)
