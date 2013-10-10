# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 13:39:45 2013

@author: ed203246

## Quality control CADASIL
For each variable:
1) Check presence in france2012.csv
2) Compare base_commun.csv vs france2012.csv
3) Count missing in base_commun.csv

4) Manuualy correct some mistakes (unit and values) => base_commun_20131009.csv

INPUT
-----

base_commun_20131008.csv
france2012.csv

OUTPUT
------
base_commun_20131009.csv
"QC/cadasil_qc.csv/html"

"""

import os.path
import pandas as pd
import numpy as np

## I/O
WD = "/neurospin/mescog"
INPUT_cadasil_base_commun_filepath = os.path.join(WD, "clinic", "base_commun_20131009.csv")
INPUT_cadasil_france2012_filepath = os.path.join(WD, "clinic", "france2012.csv")
OUTPUT_cadasil_qc = os.path.join(WD, "clinic", "QC", "cadasil_qc")
OUTPUT_cadasil_base_commun_filepath = os.path.join(WD, "clinic", "base_commun_20131009.csv")

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
# 4) Manuualy correct some mistakes (unit and values) => base_commun_20131009.csv
def set_nullunit_for_missing_value(d, val_col, unit_col):
    for i in xrange(d.shape[0]):
        if pd.isnull(d[val_col][i]):
            d[unit_col][i] = np.nan
    return d

def stats(x):
    return np.mean(x), np.std(x), np.min(x), np.max(x)
    
d = cadasil_base_commun

# Check and fix unit and value
print "Fix: GLYC17"
## => MAIL ERIC

if np.where(d.GLYC17C == '%')[0]==94:
    d.GLYC17C[94] = 'MMOL/L'
# Marco: in my base_commun the Glyc17c for 2101 is µMOL/L. Still, the value of 13 is also not plausible, even with µM. I suggest to delete this value.
if np.where(d.GLYC17 == 13)[0] == 349:
    d.GLYC17[349] = np.nan

# Some Paris subject are in G/L, but values do seem not possible
np.sum(d.GLYC17C == 'G/L')  # 8
d.ID[d.GLYC17C == 'G/L'].tolist()
#[1117, 1119, 1151, 1155, 1156, 1157, 1172, 1179]

stats(d.GLYC17[d.GLYC17C == 'G/L'] * 100)
#Out[52]: (100.374960899353, 15.337335298418084, 78.999996185303004, 130.999946594238)

stats(d.GLYC17[d.GLYC17C == 'MG/DL'])
#Out[53]: (111.09090909090909, 42.626186030856552, 78.0, 228.0)

# => Seems OK recode G/L => MG/L
d.GLYC17[d.GLYC17C == 'G/L'] *= 100
d.GLYC17C[d.GLYC17C == 'G/L'] = 'MG/DL'

# Marco Still, I think glucose for 1155, 1156 and 1157 make no sense. This has to be checked by the Paris group.

d = set_nullunit_for_missing_value(d, "GLYC17", "GLYC17C")
print "GLYC17C units:", set(d.GLYC17C)
# set([nan, 'MG/DL', 'MMOL/L'])
print "All Paris unit:", set(d.GLYC17C[d.ID<2000])
print "All Munich 'MG/DL'?", set(d.GLYC17C[d.ID>2000])




print "Fix: CHOLTOT17"
## => MAIL ERIC

# Fix error in unit MMOL/ instead of MMOL/L
d.CHOLTOT17C[d.CHOLTOT17C == 'MMOL/'] = 'MMOL/L'
# 13 Paris subject are in G/L, but values do seem not possible
d.ID[d.CHOLTOT17C == 'G/L'].tolist() # 13
# [1117, 1119, 1145, 1150, 1151, 1152, 1153, 1155, 1156, 1157, 1159, 1179, 1227]

stats(d.CHOLTOT17[d.CHOLTOT17C == 'G/L'] * 100)
#(201.0, 31.307408806125206, 155.0, 254.0)

stats(d.CHOLTOT17[d.CHOLTOT17C == 'MG/DL'])
#(210.78174603174602, 43.441919629297672, 0.5, 338.0)
# => Seems OK recode G/L => MG/L
d.CHOLTOT17[d.CHOLTOT17C == 'G/L'] *= 100
d.CHOLTOT17C[d.CHOLTOT17C == 'G/L'] = 'MG/DL'

print "CHOLTOT17 units:", set(d.CHOLTOT17C)
d = set_nullunit_for_missing_value(d, "CHOLTOT17", "CHOLTOT17C")

print "Fix: CHOLHDL17"
d.CHOLHDL17
set(d.CHOLHDL17C)
d.CHOLHDL17[d.CHOLHDL17C == 'MM1.STUNDE'] # 53
stats(d.CHOLHDL17[d.CHOLHDL17C == 'MG/DL'])
#(57.119999999999997, 16.299496924752006, 24.0, 117.0)
# OK MM1.STUNDE => 'MG/DL'
d.CHOLHDL17C[d.CHOLHDL17C == 'MM1.STUNDE'] = 'MG/DL'

stats(d.CHOLHDL17[d.CHOLHDL17C == 'G/L'] * 100)
#(55.153824732853764, 13.710442273167219, 36.999988555907997, 81.999969482422003)
stats(d.CHOLHDL17[d.CHOLHDL17C == 'MG/DL'])
#(57.087301587301589, 16.23880314675943, 24.0, 117.0)
# => Seems OK recode G/L => MG/L
d.CHOLHDL17[d.CHOLHDL17C == 'G/L'] *= 100
d.CHOLHDL17C[d.CHOLHDL17C == 'G/L'] = 'MG/DL'
d = set_nullunit_for_missing_value(d, "CHOLTOT17", "CHOLTOT17C")
print "CHOLHDL17 units:", set(d.CHOLHDL17C)


print "Fix: CHOLLDL17"
set(d.CHOLLDL17C)

stats(d.CHOLLDL17[d.CHOLLDL17C == 'G/L'] * 100)
(124.92307692307692, 26.069320425657825, 81.0, 166.0)
stats(d.CHOLLDL17[d.CHOLLDL17C == 'MG/DL'])
(133.02936507936508, 35.152725892768423, 2.7000000000000002, 249.0)
# => Seems OK recode G/L => MG/L
d.CHOLLDL17[d.CHOLLDL17C == 'G/L'] *= 100
d.CHOLLDL17C[d.CHOLLDL17C == 'G/L'] = 'MG/DL'
d = set_nullunit_for_missing_value(d, "CHOLTOT17", "CHOLTOT17C")
print "CHOLLDL17 units:", set(d.CHOLLDL17C)


print "Fix: TRIGLY17"
set(d.TRIGLY17C)

stats(d.TRIGLY17[d.TRIGLY17C == 'G/L'] * 100)
(103.92302916600146, 46.094294818847324, 43.99998188018801, 201.99985504150399)
stats(d.TRIGLY17[d.TRIGLY17C == 'MG/DL'])
(150.07936507936509, 95.732315551577031, 32.0, 685.0)
# => Seems OK recode G/L => MG/L
d.TRIGLY17[d.TRIGLY17C == 'G/L'] *= 100
d.TRIGLY17C[d.TRIGLY17C == 'G/L'] = 'MG/DL'
d = set_nullunit_for_missing_value(d, "CHOLTOT17", "CHOLTOT17C")
print "TRIGLY17 units:", set(d.TRIGLY17C)

print "Fix: HEMO17C"
#d.HEMO17C
d.ID[d.HEMO17C == 'G/L'].tolist()
#[1002]
d.HEMO17[d.HEMO17C == 'G/L']
#15.3
np.mean(d.HEMO17[d.HEMO17C == 'G/DL'])
#14.20109625668449
# just an error => recode 'G/L' => G/DL
d.HEMO17C[d.HEMO17C == 'G/L'] = 'G/DL'

print "Fix: LEUCO17"

set(d.LEUCO17C)

for v in set(d.LEUCO17C):
    print v, stats(d.LEUCO17[d.LEUCO17C == v])nan (nan, nan, nan, nan)
10
#G/L (6.9952755905511843, 1.9874748854042865, 3.3999999999999999, 18.0)
#% (5.7000000000000002, 0.0, 5.7000000000000002, 5.7000000000000002)
#10E (6.510445344129554, 3.3216657514598942, 3.2000000000000002, 49.899999999999999)
# % seems to be an error => recode 10E
d.LEUCO17C[d.LEUCO17C == '%'] = '10E'
# Seems to be G/L
