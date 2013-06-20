# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 13:39:45 2013

Produce two files:
- db_clinic_cadasil-asps_mapping_summary.csv
  Contain a summary of the variables within the merged DB.

  Fields:

      NEW_NAME: The new name in the merged DB

      IN_BOTH : Is the variable present in both DB ?

      IN_COMMON_BD: Is the variable present in the common DB?
      A variable is present in the common DB if it is in both DB at baseline
      and it is least present in one two DB for other time points.

      CADASIL_NAME & ASPS_NAME: original names

      cada_base_commun: is the variable present in the file "base_commun.xls" ?

      cada_france2012: is the variable present in the file "france2012.xls" ?

      recoding: The re-coding python code if relevent. 3 types of re-coding may occcur:
          - change discrete value: ex: " cada={1:'M', 2:'F'};" means that 1
          are re-coded 'M' and 2 'F'
          - Creation of new variables using original ones. Ex for BMI
              var_cada = cadasil_base["POIDS"] / cadasil_base["TAILLE"]**2
          - Unit convertion. Exa: var_cada = mmoll_to_mgdl(cadasil_base[cadasil_name], mm=386.65)

      remark: 
    
- db_clinic_cadasil-asps-common.csv
  the common DB

@author: edouard.duchesnay@cea.fr
"""

print __doc__

import os.path
import pandas as pd
import numpy as np

WD = "/neurospin/mescog"
WD = "/home/edouard/data/2013_mescog"

MAPPING_SUMMARY_FILEPATH = os.path.join(WD, "clinic", "db_clinic_cadasil-asps_mapping_summary") #.csv | .html
MERGE_CADASIL_ASPS_FILEPATH = os.path.join(WD, "clinic", "db_clinic_cadasil-asps-common") #.csv | .html

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

# cadasil_base is cadasil_base_commun
cadasil_base = cadasil_base_commun
##############################################################################
## Fake ASPS base
n_asps = 100
asps_dict = {mapping["ASPS.name"][i]: [np.nan] * n_asps for i in \
    xrange(mapping.shape[0]) if mapping["ASPS.given"][i]}
asps_dict["ID"] = range(n_asps)
asps_base = pd.DataFrame(asps_dict)

def replace(values, rep):
    return [rep[v] if v in rep else v for v in values]

def mmoll_to_mgdl(mmolpl, mm):
    return mmolpl * mm / 10.0

#mmol_per_liter_to_mg_per_del(53, mm=180)

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
    IN_BOTH = in_cadasil_base = in_cadasil_france2012 = 0
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
        IN_COMMON_BD = 1
    if IN_COMMON_BD and (cadasil_name or asps_name):
        print i, "new_name", new_name, "cada:", cadasil_name, "asps:", asps_name, "time", l['time point']
        in_cadasil_base = int(cadasil_name in cadasil_base.columns)
        in_cadasil_france2012 = int(cadasil_name in cadasil_france2012.columns)
        if in_cadasil_base:
            var_cada = cadasil_base[cadasil_name].tolist()
        else:
            var_cada = [np.nan] * cadasil_base.shape[0]
        in_asps_base = int(asps_name in asps_base.columns)
        if in_asps_base:
            var_asps = asps_base[asps_name].tolist()
        else:
            var_asps = [np.nan] * asps_base.shape[0]

        # Start re-coding
        if new_name == "AGE_AT_INCLUSION":
            recode_srt = """var_cada=cadasil_base["DATEINCL"]-cadasil_base["DATENAIS"]"""
            remark_str = "PROBLEM: DATEINCL, DATENAIS not found in base_commun, What is Ts in base_commun only for german patient"

        if new_name.find("SYS_BP") == 0:
            remark_str = "ok"

        if new_name.find("DIA_BP") == 0:
            remark_str = "ok"

        if new_name == "SEX":
            recode_srt = "cada={1:'M', 2:'F'}; asps={1:'F', 2:'M'}"
            exec(recode_srt)
            var_cada = replace(var_cada, cada)
            var_asps = replace(var_asps, asps)
            remark_str = "recode"

        if new_name == "HEIGHT":
            remark_str = "ok"

        if new_name.find("WEIGHT") == 0:
            remark_str = "ok"

        if (new_name.find('MIGRAINE_WO_AURA') == 0) or (new_name.find('MIGRAINE_WITH_AURA') == 0) :
            recode_srt = "cada={2:0}; asps={}"
            remark_str = "recode"
            exec(recode_srt)
            var_cada = replace(var_cada, cada)
            var_asps = replace(var_asps, asps)

        if new_name == "MIGRAINE_AGE":
            recode_srt = "cada={1:((5+15)/2.), 2:((16+30)/2.), 3:((31+40)/2.), 4:((41+50)/2.), 5:((51+60)/2.), 6:70}; asps={}"
            remark_str = "CHECK: standardization proposition"
            exec(recode_srt)
            var_cada = replace(var_cada, cada)
            ## var_asps = replace(var_asps, asps)

        if new_name == "MIGRAINE_FREQ":
            remark_str = "STANDARDIZE: CADASIL:during 2 last years : 1 (>1/week) ; 2 (>1/month < 1/week) ; 3 (>1/3 months < 1/month) ; 4 (>1 /2years < 1/3 months) ; 5 (none) WITH ASPS:migraine attacks frequency with therapy"
            recode_srt = """cada={}; asps={}"""
            exec(recode_srt)
            var_cada = replace(var_cada, cada)
            var_asps = replace(var_asps, asps)

        if new_name == "MIGRAINE_MED":
            remark_str = "STANDARDIZE: CADASIL: See Medications WITH ASPS: 1=true, 0=false"
            recode_srt = """cada={}; asps={}"""
            exec(recode_srt)
            var_cada = replace(var_cada, cada)
            var_asps = replace(var_asps, asps)

        if new_name == "BMI":
            recode_srt = 'var_cada = cadasil_base["POIDS"] / cadasil_base["TAILLE"]**2'
            remark_str = "ok"
            exec(recode_srt)
            var_cada = var_cada.tolist()

        if new_name == "BMI@M18":
            recode_srt = 'var_cada = cadasil_base["POIDS27"] / cadasil_base["TAILLE"]**2'
            remark_str = "ok"
            exec(recode_srt)
            var_cada = var_cada.tolist()

        if new_name == "BMI@M36":
            recode_srt = 'var_cada = cadasil_base["POIDS40"] / cadasil_base["TAILLE"]**2'
            remark_str = "ok"
            exec(recode_srt)
            var_cada = var_cada.tolist()

        if new_name == "BMI@M54":
            recode_srt = 'var_cada = cadasil_base["POIDS54"] / cadasil_base["TAILLE"]**2'
            remark_str = "PROBLEM: POIDS54 not in cadasil_base"
            #exec(recode_srt)
            #var_cada = var_cada.tolist()

        if new_name == "BMI@M72":
            remark_str = "ok"

        if new_name == "SMOKING":
            recode_srt = """cadasil_base["TABAC"][cadasil_base["ANCIENFUM"] == 1] = 3;\
            cada={1:"current", 2:"never", 3:"former"}; asps={0:"never", 1:"current", 2:"former"}"""
            remark_str = "ok"
            exec(recode_srt)
            var_cada = replace(cadasil_base["TABAC"].tolist(), cada)
            var_asps = replace(var_asps, asps)

        if new_name == "SMOKING@M36" or new_name == "SMOKING@M72":
            recode_srt = """cada={}; asps={0:"never", 1:"current", 2:"former"}"""
            remark_str = "ok"
            var_asps = replace(var_asps, asps)

        if new_name == "ALCOHOL":
            remark_str = "STANDARDIZE:  CADASIL: 1 = none, 2 = < 2 drinks a day ; 3 = > 2 drinks a day ASPS: alcohol in drinks per day"
            recode_srt = """cada={}; asps={}"""
            exec(recode_srt)
            var_cada = replace(var_cada, cada)
            var_asps = replace(var_asps, asps)

        if new_name == "ALCOHOL@M36" or new_name == "ALCOHOL@M72":
            remark_str = "STANDARDIZE:  See ALCOHOL"
            recode_srt = """cada={}; asps={}"""
            exec(recode_srt)
            #var_cada = replace(var_cada, cada)
            var_asps = replace(var_asps, asps)

        if new_name.find("HYPERTENSION") == 0 :
            remark_str = "ok"
            recode_srt = """cada={2:0}; asps={}"""
            exec(recode_srt)
            var_cada = replace(var_cada, cada)
            #var_asps = replace(var_asps, asps)

        if new_name.find("DIABETES") == 0 :
            remark_str = "ok"
            recode_srt = """cada={2:0}; asps={}"""
            exec(recode_srt)
            var_cada = replace(var_cada, cada)
            #var_asps = replace(var_asps, asps)

        if new_name.find("HYPERCHOL") == 0 :
            remark_str = "ok"
            recode_srt = """cada={2:0}; asps={}"""
            exec(recode_srt)
            var_cada = replace(var_cada, cada)
            #var_asps = replace(var_asps, asps)

        if new_name.find("PERIPH_VASC_DIS") == 0 :
            remark_str = "ok"
            recode_srt = """cada={2:0}; asps={}"""
            exec(recode_srt)
            var_cada = replace(var_cada, cada)
            #var_asps = replace(var_asps, asps)

        if new_name.find("VENOUS_DIS") == 0 :
            remark_str = "ok"
            recode_srt = """cada={2:0}; asps={}"""
            exec(recode_srt)
            var_cada = replace(var_cada, cada)
            #var_asps = replace(var_asps, asps)

        if new_name.find("FAST_GLUC") == 0 :
            remark_str = "ok, conv cadasil to mg/dl. 1 mmol/l = 180/10 mg/dL. (mm=180 for glucose)"
            recode_srt = """var_cada = mmoll_to_mgdl(cadasil_base[cadasil_name], mm=180)"""
            if cadasil_name in cadasil_base.columns:
                exec(recode_srt)
                var_cada = var_cada.tolist()
            #var_asps = replace(var_asps, asps)

        if new_name.find("HBA1C") == 0 :
            remark_str = "STANDARDIZE: How to conv cadasil % to mg/dl"
            recode_srt = """"""
            #exec(recode_srt)
            #var_cada = var_cada.tolist()
            #var_asps = replace(var_asps, asps)

        if (new_name.find("CHOL") == 0) or (new_name.find("HDL") == 0) \
            or (new_name.find("LDL") == 0):
            remark_str = "ok, conv cadasil to mg/dl. 1 mmol/L = 386.65/10 mg/dL. (mm=386.65 for C27H46O, HDL and LDL)"
            recode_srt = """var_cada = mmoll_to_mgdl(cadasil_base[cadasil_name], mm=386.65)"""
            if cadasil_name in cadasil_base.columns:
                exec(recode_srt)
                var_cada = var_cada.tolist()
            #var_asps = replace(var_asps, asps)

        if (new_name.find("TRIGLY") == 0):
            remark_str = "ok, conv cadasil to mg/dl. 1 mmol/L = 875/10 mg/dL. (mm=875 for triglycerides)"
            recode_srt = """var_cada = mmoll_to_mgdl(cadasil_base[cadasil_name], mm=875)"""
            if cadasil_name in cadasil_base.columns:
                exec(recode_srt)
                var_cada = var_cada.tolist()
            #var_asps = replace(var_asps, asps)

        if (new_name.find("HEMOGLOBIN") == 0) \
            or (new_name.find("LEUKO_COUNT") == 0) \
            or new_name.find("THROMBO_COUNT") == 0:
            remark_str = "ok"

        if new_name.find("FIBRINOGEN") == 0 :
            remark_str = "ok, conv cadasil from g/L to mg/dL"
            if cadasil_name in cadasil_base.columns:
                recode_srt = """var_cada = cadasil_base[cadasil_name] * 100"""
                exec(recode_srt)
                var_cada = var_cada.tolist()

        if new_name == "AF_HIST":
            recode_srt = """cada={2:0}; asps={}"""
            exec(recode_srt)
            var_cada = replace(var_cada, cada)
            #var_asps = replace(var_asps, asps)

        ## Add to common DB
        common_db[new_name] = pd.Series(var_cada + var_asps)
    if new_name:  # if new common name has been difined keep it
        variables.append((new_name, IN_BOTH, IN_COMMON_BD, cadasil_name, asps_name,
                          in_cadasil_base, in_cadasil_france2012,
                          recode_srt, remark_str))

common_db["ID"] = pd.Series(cadasil_base["ID"].tolist() + asps_base["ID"].tolist())
common_db["BASE"] = pd.Series(["cadasil"]*len(cadasil_base["ID"]) + ["asps"]*len(asps_base["ID"]))


## Mapping summary re-order IN_COMMON_BD first
mapping_summary = pd.DataFrame(variables,
                               columns=['NEW_NAME', 'IN_BOTH', 'IN_COMMON_BD', 'CADASIL_NAME', 'ASPS_NAME',
                      'cada_base_commun', 'cada_france2012',
                      'recoding', 'remark'])

mapping_summary = pd.concat([mapping_summary[mapping_summary['IN_COMMON_BD'] == 1],
mapping_summary[mapping_summary['IN_COMMON_BD'] == 0]])
mapping_summary.index = range(mapping_summary.shape[0])
#print mapping_summary.to_string()
print mapping_summary[mapping_summary['IN_COMMON_BD'] == 1].to_string()

mapping_summary.to_csv(MAPPING_SUMMARY_FILEPATH+".csv", sep="\t", index=False)
mapping_summary.to_html(MAPPING_SUMMARY_FILEPATH+".html")

mapping_summary.to_csv(MAPPING_SUMMARY_FILEPATH+".csv", sep="\t", index=False)

## COMMON_DB
db_columns = ["ID", "BASE"] + mapping_summary[mapping_summary.IN_COMMON_BD==1]["NEW_NAME"].tolist()

common_db = pd.DataFrame(common_db, columns=db_columns)
common_db.to_csv(MERGE_CADASIL_ASPS_FILEPATH+".csv", sep="\t", index=False)

mapping_summary = pd.read_table(MAPPING_SUMMARY_FILEPATH+".csv", header=0)
s = mapping_summary.to_string()
f = open(MERGE_CADASIL_ASPS_FILEPATH+".txt", "w")
f.write(s)
f.close()
