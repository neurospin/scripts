# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 10:44:49 2013

Add DATEINCL and DATENAIS an compute AGE_AT_INCLUSION in CADASIL subjects

INPUT
-----

"base_commun.csv"
"france2012.csv" => date DATEINCL and DATENAIS for french
"CAD_Munich_Dates.txt" => date DATEINCL and DATENAIS for german

OUTPUT
------

"base_commun_20140109.csv" == "base_commun.csv" + Date (from "france2012.csv" + CAD_Munich_Dates.txt)

"""

##############################################################################
## Add DATEINCL and DATENAIS in CADASIL
# 1) Check presence in france2012.csv
# 2) Compare base_commun.csv vs france2012.csv
# 3) Count missing in base_commun.csv
##############################################################################

import os.path
import string
import pandas as pd
import numpy as np

WD = "/neurospin/mescog"

INPUT_cadasil_base_commun_filepath = os.path.join(WD, "clinic", "base_commun.csv")
INPUT_cadasil_france2012_filepath = os.path.join(WD, "clinic", "france2012.csv")
INPUT_cadasil_munich_date_filepath = os.path.join(WD, "clinic", "CAD_Munich_Dates.txt")
OUTPUT = os.path.join(WD, "clinic", "base_commun_20140109.csv")

cadasil_base_commun = pd.read_table(INPUT_cadasil_base_commun_filepath, header=0, sep=",").replace("-", np.nan)
cadasil_france2012 = pd.read_table(INPUT_cadasil_france2012_filepath, header=0, sep=",").replace("-", np.nan)
cadasil_munich_date = pd.read_table(INPUT_cadasil_munich_date_filepath, header=0)

# Look for differences between cadasil_base_commun & cadasil_france2012
cadasil_base_commun.columns = [s.upper() for s in cadasil_base_commun.columns]
cadasil_france2012.columns = [s.upper() for s in cadasil_france2012.columns]

# drop DATEINCL and DATENAIS of cadasil_base_commun
cadasil_base_commun = cadasil_base_commun.drop(["DATEINCL", "DATENAIS"], 1)

# 1) Recode date in cadasil_france2012
# 1/28/10 12:00 AM => 2010-28-01
def format_date(dates, pref="20"):
    out = list()
    for l in dates:
        date = l.split()[0].split("/")
        out.append('%s%s-%s-%s' % (pref, date[0], "{:0>2d}".format(int(date[1])), "{:0>2d}".format(int(date[2]))))
    return out

cadasil_france2012.ID
cadasil_france2012.DATEINCL = format_date(cadasil_france2012.DATEINCL, pref="")
cadasil_france2012.DATENAIS = format_date(cadasil_france2012.DATENAIS, pref="")

cadasil_france2012[["ID", "DATENAIS", "DATEINCL"]]
# 2) reformat cadasil_munich_date.ID
#cadasil_munich_date.ID contains "CAD_" remove, it will be added later
cadasil_munich_date.ID = [int(s.replace('CAD_', '')) for s in cadasil_munich_date.ID]
cadasil_munich_date.ID

# 3) Merge to cadasil_base_commun
cada_dates = cadasil_france2012[["ID", "DATENAIS", "DATEINCL"]].append(cadasil_munich_date, ignore_index=True)

age = pd.DataFrame(dict(AGE_AT_INCLUSION=[int(cada_dates.DATEINCL[i].split("-")[0]) - int(cada_dates.DATENAIS[i].split("-")[0])
for i in xrange(cada_dates.shape[0])]))

cada_dates = pd.concat([cada_dates, age], axis=1)

print cada_dates.to_string()
# Merge with cadasil_base_commun and save

merge = pd.merge(cada_dates, cadasil_base_commun, on="ID")
merge.to_csv(OUTPUT, sep=",", index=False)
