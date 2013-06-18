# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 13:39:45 2013

@author: ed203246
"""

import os.path
import pandas as pd
import numpy as np

WD = "/neurospin/mescog"

##############################################################################
## Quality control CADASIL
# 1) Check presence in france2012.csv
# 2) Compare base_commun.csv vs france2012.csv
# 3) Count missing in base_commun.csv
##############################################################################

cadasil_base_commun_filepath = os.path.join(WD, "clinic", "base_commun.csv")
cadasil_france2012_filepath = os.path.join(WD, "clinic", "france2012.csv")

cadasil_base_commun = pd.read_table(cadasil_base_commun_filepath, header=0).replace("-", np.nan)
cadasil_france2012 = pd.read_table(cadasil_france2012_filepath, header=0).replace("-", np.nan)


# Look for differences between cadasil_base_commun & cadasil_france2012
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

qc_cada_common_var.to_csv(os.path.join(WD, "clinic", "QC", "cadasil_qc.csv"), sep="\t", index=False)
qc_cada_common_var.to_html(os.path.join(WD, "clinic", "QC", "cadasil_qc.html"))

