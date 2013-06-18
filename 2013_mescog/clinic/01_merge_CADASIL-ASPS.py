# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 13:39:45 2013

@author: ed203246
"""

import os.path
import pandas as pd
import numpy as np

WD = "/neurospin/mescog"
OUTPUT_FILEPATH = os.path.join(WD, "clinic", "DB_Mapping_summary") #.csv | .html
mapping_filepath = os.path.join(WD, "clinic",
                                "DB_Mapping_Longit_Last_EJ_2013-05-08.csv")

cadasil_base_commun_filepath = os.path.join(WD, "clinic", "base_commun.csv")
cadasil_base_commun = pd.read_table(cadasil_base_commun_filepath, header=0).replace("-", np.nan)

cadasil_france2012_filepath = os.path.join(WD, "clinic", "france2012.csv")
cadasil_france2012 = pd.read_table(cadasil_france2012_filepath, header=0).replace("-", np.nan)

# MAPPING
mapping = pd.read_table(mapping_filepath, header=0).replace("-", np.nan)

##############################################################################
## IN_BOTH variables output => common_df
## Merged clinical base will contain all variables with a  "new common name"
## in DB_Mapping_Longit_Last_EJ_2013-05-08.csv
## Standardized variables:
## If a variable is IN_BOTH at M0, all time point should be standardized
## Other are simply copied
##############################################################################
# If present in both, keep all time points even if not present in ASPS
# Upper case cadasil_base_commun.columns
cadasil_base_commun.columns = [s.upper() for s in cadasil_base_commun.columns]
cadasil_france2012.columns = [s.upper() for s in cadasil_france2012.columns]

# Check no doublons
print len(set(cadasil_base_commun.columns)) == len(cadasil_base_commun.columns)
print len(set(cadasil_france2012.columns)) == len(cadasil_france2012.columns)

"""
def recoding(new_var):
    recode_srt, remark_str = "", ""

    return recode_srt, remark_str

col = cadasil_base_commun["Sexe"]
exec(recode_srt)
replace(col, cada)
del cada
"""
def replace(values, rep):
    for l in set(values):
        if not l in rep:
            rep[l] = l
    return [rep[v] for v in values]


# IN_BOTH : present in both cadasil and asps
# IN_COMMON_BD: present in the common database. If the baseline is in both
# dDB (IN_BOTH=1) all the next time-point will be in the common DB even if they
# are missing in one DB.
variables = list()
common_db = dict()
for i in xrange(mapping.shape[0]):
    #i = 0
    l = mapping.ix[i, :]
    if l['time point'] == 'M0' or pd.isnull(l['time point']):  # Reset IN_COMMON_BD flag
        IN_COMMON_BD = 0
    cadasil_name = asps_name = new_name = recode_srt = remark_str = ""
    IN_BOTH = in_cadasil_base_commun = in_cadasil_france2012 = 0
    if pd.notnull(l["new common name"]):  # could be null since noty repeated for time points
        curr = l['new common name']
        new_name = curr
    elif (l["CADASIL.given"] == 1 or l["ASPS.given"] ==1) and pd.notnull(l['time point']):
        new_name = curr + "@" + l['time point']
    if l["CADASIL.given"] == 1:
        cadasil_name = l["CADASIL.name"].upper()
    if l["ASPS.given"] == 1:
        asps_name = l["ASPS.name"]
    if cadasil_name and asps_name:
        IN_BOTH = 1
        IN_COMMON_BD = 1 # Set IN_COMMON_BD flag
    if IN_COMMON_BD:
        in_cadasil_base_commun = int(cadasil_name in cadasil_base_commun.columns)
        in_cadasil_france2012 = int(cadasil_name in cadasil_france2012.columns)
        if in_cadasil_base_commun:
            var_cada = cadasil_base_commun[cadasil_name].tolist()
        else:
            var_cada = []
        var_asps = []#replace(asps_base[asps_name], asps)
        if new_name == "AGE_AT_INCLUSION":
            recode_srt = "DATEINCL-DATENAIS"
            remark_str = "PROBLEM: DATEINCL, DATENAIS not found in base_commun, What is Ts in base_commun only for german patient"
        if new_name == "SEX":
            recode_srt = "cada={1:'M', 2:'F'}; asps={1:'F', 2:'M'}"
            exec(recode_srt)
            var_cada = replace(var_cada, cada)
            var_asps = []#replace(asps_base[asps_name], asps)
        if (new_name.find('MIGRAINE_WO_AURA') != -1) or (new_name.find('MIGRAINE_WITH_AURA') != -1) :
            recode_srt = "cada={1:1, 2:0}; asps={}"
            exec(recode_srt)
            var_cada = replace(var_cada, cada)
            var_asps = []#replace(asps_base[asps_name], asps)
        if new_name == "MIGRAINE_AGE":
            recode_srt = "cada={1:((5+15)/2.), 2:((16+30)/2.), 3:((31+40)/2.), 4:((41+50)/2.), 5:((51+60)/2.), 6:70}; asps={}"
            exec(recode_srt)
            var_cada = replace(var_cada, cada)
            var_asps = []#replace(asps_base[asps_name], asps)
        if new_name == "MIGRAINE_FREQ":
            remark_str = "STANDARDIZE: CADASIL:during 2 last years : 1 (>1/week) ; 2 (>1/month < 1/week) ; 3 (>1/3 months < 1/month) ; 4 (>1 /2years < 1/3 months) ; 5 (none) WITH ASPS:migraine attacks frequency with therapy"
            var_asps = []#replace(asps_base[asps_name], asps)
        if new_name == "MIGRAINE_MED":
            remark_str = "STANDARDIZE: CADASIL: See Medications WITH ASPS: 1=true, 0=false"
        if new_name == "BMI":
            recode_srt = 'var_cada = cadasil_base_commun["POIDS"] / cadasil_base_commun["TAILLE"]**2'
            exec(recode_srt)
            var_cada = var_cada.tolist()
        if new_name == "BMI@M18":
            recode_srt = 'var_cada = cadasil_base_commun["POIDS27"] / cadasil_base_commun["TAILLE"]**2'
            exec(recode_srt)
            var_cada = var_cada.tolist()
        if new_name == "BMI@M36":
            recode_srt = 'var_cada = cadasil_base_commun["POIDS40"] / cadasil_base_commun["TAILLE"]**2'
            exec(recode_srt)
            var_cada = var_cada.tolist()
        if new_name == "BMI@M54":
            recode_srt = 'var_cada = cadasil_base_commun["POIDS54"] / cadasil_base_commun["TAILLE"]**2'
            exec(recode_srt)
            var_cada = var_cada.tolist()
        if new_name == "SMOKING":
            cadasil_base_commun["TABAC"][cadasil_base_commun["ANCIENFUM"] == 1] = 3
            recode_srt = 'cada={1:"current", 2:"never", 3:"former"}; asps={1:"current", 0:"never", 2:"former"}'
            exec(recode_srt)
            var_cada = replace(cadasil_base_commun["TABAC"].tolist(), cada)
        
        common_db[new_name] = pd.Series(var_cada + var_asps)
    if new_name:  # if new common name has been difined keep it
#        if new_name == "SEX":
#            break
        #recode_srt, remark_str = recoding(new_name=new_name,
        #    in_cadasil_base_commun, in_cadasil_france2012)
        variables.append((new_name, IN_BOTH, IN_COMMON_BD, cadasil_name, asps_name,
                          in_cadasil_base_commun, in_cadasil_france2012,
                          recode_srt, remark_str))

mapping_summary =  pd.DataFrame(variables,
             columns=['NEW_NAME', 'IN_BOTH', 'IN_COMMON_BD', 'CADASIL_NAME', 'ASPS_NAME',
                      'cada_base_commun', 'cada_france2012',
                      'recoding', 'remark'])


mapping_summary = pd.concat([mapping_summary[mapping_summary['IN_COMMON_BD'] == 1],
mapping_summary[mapping_summary['IN_COMMON_BD'] == 0]])
mapping_summary.index = range(mapping_summary.shape[0])
#print mapping_summary.to_string()
print mapping_summary[mapping_summary['IN_COMMON_BD'] == 1].to_string()






mapping_summary.to_csv(OUTPUT_FILEPATH+".csv", sep="\t", index=False)
mapping_summary.to_html(OUTPUT_FILEPATH+".html")



##############################################################################
## Standardize IN_BOTH variables
##############################################################################
new_var = 'AGE_AT_INCLUSION'
cadasil_base_commun.DATEINCL