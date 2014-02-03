# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 13:39:45 2013

Create the CADASIL/ASP(F)S common DB
A variable is present in the common DB if it is in both DB at baseline
and it is least present in one two DB for other time points.

INPUT:
------

base_commun_20140109.csv
ASPS_klinVariables_20131015.csv
ASPFS_klinVariables_20130711.csv
DB_Mapping_Longit_Last_EJ_20131007.csv

OUTPUT:
-------

1) Summary of common DB
~~~~~~~~~~~~~~~~~~~~~~~
"commondb_clinic_cadasil-asps-aspfs_mapping-summary_20140109 #.csv | .html
"commondb_clinic_cadasil-asps-aspfs_20140109" #.csv | .html

    Fields:

      NEW_NAME: The new name in the merged DB

      CADASIL_NAME & ASPS_NAME: original names

      IN_CAD: was the variable found in CADASIL?

      IN_ASPS: was the variable found in ASPS?

      UNIT:

      CAD/ASPS_NaN: number of missing data.

      CAD/ASPS_mean: Means, for numerical variables only.

      CAD/ASPS_std: Std-dev, for numerical variables only.

      CAD/ASPS_min/max: Min/max, for numerical variables only.

      CAD/ASPS_count: Count of each level of categorial variables

      RECODING: The re-coding python code if relevent. 3 types of re-coding may occcur:
          - change discrete value: ex: " cada={1:'M', 2:'F'};" means that 1
          are re-coded 'M' and 2 'F'
          - Creation of new variables using original ones. Ex for BMI
              var_cada = cadasil_base["POIDS"] / cadasil_base["TAILLE"]**2
          - Unit convertion. Exa: var_cada = mgdl(cadasil_base[cadasil_name], cadasil_base[cadasil_name+"C"], mm=386.65)

      remark: 

2) Common DB
~~~~~~~~~~~~

    "commondb_clinic_cadasil-asps_20130811.csv"


@author: edouard.duchesnay@cea.fr
"""

print __doc__

import os.path
import pandas as pd
import numpy as np

WD = "/neurospin/mescog"
#WD = "/home/edouard/data/2013_mescog"

###################################################################################
# INPUTS
INPUT_mapping_filepath = os.path.join(WD, "clinic",
                                "DB_Mapping_Longit_Last_EJ_20131007.csv")
INPUT_cadasil_base_commun_filepath = os.path.join(WD, "clinic", "base_commun_20140109.csv")
INPUT_asps_filepath = os.path.join(WD, "clinic", "ASPS_klinVariables_20131015.csv")
INPUT_aspfs_filepath = os.path.join(WD, "clinic", "ASPFS_klinVariables_20130711.csv")


OUTPUT_MAPPING_SUMMARY_FILEPATH = os.path.join(WD, "clinic", "commondb_clinic_cadasil-asps-aspfs_mapping-summary_20140109") #.csv | .html
OUTPUT_MERGE_CADASIL_ASPS_FILEPATH = os.path.join(WD, "clinic", "commondb_clinic_cadasil-asps-aspfs_20140109") #.csv | .html

cadasil_base = pd.read_table(INPUT_cadasil_base_commun_filepath, header=0, sep=',').replace("-", np.nan)
asps_base = pd.read_table(INPUT_asps_filepath, header=0, sep=',')
aspfs_base = pd.read_table(INPUT_aspfs_filepath, header=0)
mapping = pd.read_table(INPUT_mapping_filepath, header=0).replace("-", np.nan)

# Upper case cadasil_base_commun.columns
cadasil_base.columns = [s.upper() for s in cadasil_base.columns]
#asps.columns = [s.upper() for s in asps.columns]  ## ASPS already in upper case

# Check no doublons
print len(set(cadasil_base.columns)) == len(cadasil_base.columns)

# cadasil_base_commun

###################################################################################
# UTILS
def replace(values, rep):
    return [rep[v] if v in rep else v for v in values]

#def mmoll_to_mgdl(mmolpl, mm):
#    return mmolpl * mm / 10.0

#cadasil_name='GLYC17'
#unit = cadasil_base[cadasil_name+"C"]
#x = cadasil_base[cadasil_name]
#mm=180

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

#def convert_to_mgdl(x)
#set([nan, 'G/L', 'DM', '%', 'NF', 'MG/DL', '\xc2\xb5MOL/L', 'MMOL/L'])

def do_stat(var_cada, var_asps, var_aspfs):
    """Do some statistics on variable. Return a tuble with:
    "CAD_N",  "ASPS_N", ASPFS_N # of missing

    for quantititative variables:
        "CAD_mean", "ASPS_mean", "CAD_std", "ASPS_std": basic stats
        "CAD_min", "ASPS_min", "CAD_max", "ASPS_max": basic stats

    for qulitative variables:
        "CAD_count", "ASPS_count"
    """
#    def is_numeric(obj):
#        attrs = ['__add__', '__sub__', '__mul__', '__div__', '__pow__']
#        return all(hasattr(obj, attr) for attr in attrs)
    c = pd.Series(var_cada)
    a = pd.Series(var_asps)
    f = pd.Series(var_aspfs)
    c_na = pd.isnull(c)
    a_na = pd.isnull(a)
    f_na = pd.isnull(f)
    if len(set(c[np.logical_not(c_na)]).union(set(a[np.logical_not(a_na)]))) > 6:  
        stats = [
        len(c_na) - c_na.sum(), len(a_na) - a_na.sum(), len(f_na) - f_na.sum(),
        np.mean(c[np.logical_not(c_na)]), np.mean(a[np.logical_not(a_na)]), np.mean(f[np.logical_not(f_na)]),
        np.std(c[np.logical_not(c_na)]), np.std(a[np.logical_not(a_na)]), np.std(f[np.logical_not(f_na)]),
        np.nan if not len(c[np.logical_not(c_na)]) else np.min(c[np.logical_not(c_na)]),
        np.nan if not len(a[np.logical_not(a_na)]) else np.min(a[np.logical_not(a_na)]),
        np.nan if not len(f[np.logical_not(f_na)]) else np.min(f[np.logical_not(f_na)]),
        np.nan if not len(c[np.logical_not(c_na)]) else np.max(c[np.logical_not(c_na)]),
        np.nan if not len(a[np.logical_not(a_na)]) else np.max(a[np.logical_not(a_na)]),
        np.nan if not len(f[np.logical_not(f_na)]) else np.max(f[np.logical_not(f_na)]),
        np.nan, np.nan, np.nan]
    else:
        c_counts = ['%s:%i' % (str(l), (c[np.logical_not(c_na)]==l).sum()) for 
                l in set(c[np.logical_not(c_na)])]
        a_counts = ['%s:%i' % (str(l), (a[np.logical_not(a_na)]==l).sum()) for 
                l in set(a[np.logical_not(a_na)])]
        f_counts = ['%s:%i' % (str(l), (f[np.logical_not(f_na)]==l).sum()) for 
                l in set(f[np.logical_not(f_na)])]
        stats = [
            len(c_na) - c_na.sum(), len(a_na) - a_na.sum(), len(f_na) - f_na.sum(),
            np.nan, np.nan, np.nan,
            np.nan, np.nan, np.nan,
            np.nan, np.nan, np.nan,
            np.nan, np.nan, np.nan,
            " ".join(c_counts), " ".join(a_counts), " ".join(f_counts)]
    return stats

stat_colnames = ["CAD_N",  "ASPS_N", "ASPFS_N", 
            "CAD_MEAN", "ASPS_MEAN", "ASPFS_MEAN", "CAD_STD", "ASPS_STD", "ASPFS_STD",
            "CAD_MIN", "ASPS_MIN", "ASPFS_MIN", "CAD_MAX", "ASPS_MAX", "ASPFS_MAX",
            "CAD_COUNT", "ASPS_COUNT", "ASPFS_COUNT"]

###################################################################################
# DO THE MAPPING
variables = list()
common_db = dict()
for i in xrange(mapping.shape[0]):
    #i = 0
    l = mapping.ix[i, :]
    if l['time point'] == 'M0' or pd.isnull(l['time point']):  # Reset IN_COMMON flag
        IN_COMMON = 0
    cadasil_name = asps_name = aspfs_name = new_name = unit_str = recode_srt = remark_str = ""
    in_cadasil_base = in_asps_base = in_aspfs_base = 0
    if pd.notnull(l["new common name"]):  # could be null since noty repeated for time points
        curr = l['new common name']
        new_name = curr
    elif (l["CADASIL.given"] == 1 or l["ASPS.given"] ==1 or l["ASPFS.given"] ==1) \
        and pd.notnull(l['time point']):
        new_name = curr + "." + l['time point']
    if l["CADASIL.given"] == 1:
        cadasil_name = l["CADASIL.name"].upper()
        in_cadasil_base = int(cadasil_name in cadasil_base.columns)
    if l["ASPS.given"] == 1:
        asps_name = l["ASPS.name"]
        in_asps_base = int(asps_name in asps_base.columns)
    if l["ASPFS.given"] == 1:
        aspfs_name = l["ASPFS.name"]
        in_aspfs_base = int(aspfs_name in aspfs_base.columns)
    if cadasil_name and asps_name and aspfs_name:
        IN_COMMON = 1
    if IN_COMMON and (in_cadasil_base or in_asps_base or in_aspfs_base):
        print i, "new_name:", new_name, "cada:", cadasil_name, "asps:", asps_name, "aspfs:", aspfs_name, "time", l['time point']
        #in_cadasil_france2012 = int(cadasil_name in cadasil_france2012.columns)
        if in_cadasil_base:
            var_cada = cadasil_base[cadasil_name].tolist()
        else:
            var_cada = [np.nan] * cadasil_base.shape[0]
        if in_asps_base:
            var_asps = asps_base[asps_name].tolist()
        else:
            var_asps = [np.nan] * asps_base.shape[0]
        if in_aspfs_base:
            var_aspfs = aspfs_base[aspfs_name].tolist()
        else:
            var_aspfs = [np.nan] * aspfs_base.shape[0]

        # Recode variables
        if new_name.find("SYS_BP") == 0:
            remark_str = "ok"

        if new_name.find("DIA_BP") == 0:
            remark_str = "ok"

        if new_name == "AGE_AT_INCLUSION":
            unit_str = "year"

        if new_name == "SEX":
            unit_str = "m, f"
            recode_srt = "cada={1:'m', 2:'f'}; asps={1:'f', 2:'m'}; aspfs={1:'f', 2:'m'}"
            exec(recode_srt)
            var_cada = replace(var_cada, cada)
            var_asps = replace(var_asps, asps)
            var_aspfs = replace(var_aspfs, aspfs)
            remark_str = "recode"

        if new_name == "HEIGHT":
            unit_str = "cm"
            remark_str = "ok"

        if new_name.find("WEIGHT") == 0:
            unit_str = "kg"
            remark_str = "ok"

        if (new_name.find('MIGRAINE_WO_AURA') == 0) or (new_name.find('MIGRAINE_WITH_AURA') == 0) :
            unit_str = "0:no, 1:yes"
            recode_srt = """cada={2:0}; asps={"nein":0, "ja":1}; aspfs={"nein":0, "ja":1}"""
            remark_str = "recode"
            exec(recode_srt)
            var_cada = replace(var_cada, cada)
            var_asps = replace(var_asps, asps)
            var_aspfs = replace(var_aspfs, aspfs)

        if new_name == "MIGRAINE_AGE":
            unit_str = "1:5-15(years), 2:16-30, 3:31-40, 4:41-50, 5:51-60, 6:>60"
            recode_srt = "cada={1:((5+15)/2.), 2:((16+30)/2.), 3:((31+40)/2.), 4:((41+50)/2.), 5:((51+60)/2.), 6:70}; asps={}"
            remark_str = """map cadasil to asps ANFALLVON using the average age of the range"""
            exec(recode_srt)
            var_cada = replace(var_cada, cada)

        if new_name == "MIGRAINE_FREQ":
            IN_COMMON = 0
            remark_str = "Difficulties to standardize, removed"

        if new_name == "MIGRAINE_MED":
            IN_COMMON = 0
            remark_str = "Difficulties to standardize, removed"

        if new_name == "BMI":
            unit_str = "kg/m2"
            recode_srt = 'var_cada = cadasil_base["POIDS"] / (cadasil_base["TAILLE"] / 100.)**2'
            remark_str = "ok"
            exec(recode_srt)
            var_cada = var_cada.tolist()

        if new_name == "BMI.M18":
            unit_str = "kg/m2"
            recode_srt = 'var_cada = cadasil_base["POIDS27"] / (cadasil_base["TAILLE"] / 100.)**2'
            remark_str = "ok"
            exec(recode_srt)
            var_cada = var_cada.tolist()

        if new_name == "BMI.M36":
            unit_str = "kg/m2"
            recode_srt = 'var_cada = cadasil_base["POIDS40"] / (cadasil_base["TAILLE"] / 100.)**2'
            remark_str = "ok"
            exec(recode_srt)
            var_cada = var_cada.tolist()

        if new_name == "BMI.M54":
            unit_str = "kg/m2"
            recode_srt = 'var_cada = cadasil_base["POIDS54"] / (cadasil_base["TAILLE"] / 100.)**2'
            remark_str = "PROBLEM: POIDS54 not in cadasil_base"

        if new_name == "BMI.M72":
            unit_str = "kg/m2"
            remark_str = "ok"

        if new_name == "SMOKING":
            unit_str = "current, never, former"
            recode_srt = """cadasil_base["TABAC"][cadasil_base["ANCIENFUM"] == 1] = 3;\
            cada={1:"current", 2:"never", 3:"former"}; asps={0:"never", 1:"current", 2:"former"}; aspfs={0:"never", 1:"current", 2:"former"}"""
            remark_str = "ok"
            exec(recode_srt)
            var_cada = replace(cadasil_base["TABAC"].tolist(), cada)
            var_asps = replace(var_asps, asps)
            var_aspfs = replace(var_aspfs, aspfs)

        if new_name == "SMOKING.M36" or new_name == "SMOKING.M72":
            unit_str = "current, never, former"
            recode_srt = """cada={}; asps={0:"never", 1:"current", 2:"former"}"""
            remark_str = "ok"
            var_asps = replace(var_asps, asps)

        recode_srt_alcohol =\
        """def asps_to_cad(v):
                if np.isnan(v): return np.nan
                if v == 0: return 1
                if v <= 2: return 2
                return 3
var_asps = [asps_to_cad(v) for v in var_asps]
var_aspfs = [asps_to_cad(v) for v in var_aspfs]
        """
        if new_name == "ALCOHOL":
            unit_str = "1:none, 2:<=2 drinks a day, 3:>2 drinks a day"
            remark_str = "STANDARDIZE:  CADASIL: 1 = none, 2 = < 2 drinks a day ; 3 = > 2 drinks a day ASPS: alcohol in drinks per day"
            recode_srt = recode_srt_alcohol
            exec(recode_srt)
            
        if new_name == "ALCOHOL.M36" or new_name == "ALCOHOL.M72":
            unit_str = "1:none, 2:<=2 drinks a day, 3:>2 drinks a day"
            remark_str = "STANDARDIZED:  See ALCOHOL"
            recode_srt = recode_srt_alcohol
            exec(recode_srt)

        if new_name.find("HYPERTENSION") == 0 :
            unit_str = "0:no, 1:yes"
            remark_str = "ok"
            recode_srt = """cada={2:0}; asps={}"""
            exec(recode_srt)
            var_cada = replace(var_cada, cada)

        if new_name.find("DIABETES") == 0 :
            unit_str = "0:no, 1:yes"
            remark_str = "ok"
            recode_srt = """cada={2:0}; asps={}"""
            exec(recode_srt)
            var_cada = replace(var_cada, cada)

        if new_name.find("HYPERCHOL") == 0 :
            unit_str = "0:no, 1:yes"
            remark_str = "ok"
            recode_srt = """cada={2:0}; asps={}"""
            exec(recode_srt)
            var_cada = replace(var_cada, cada)

        if new_name.find("PERIPH_VASC_DIS") == 0 :
            unit_str = "0:no, 1:yes"
            remark_str = "ok"
            recode_srt = """cada={2:0}; asps={}"""
            exec(recode_srt)
            var_cada = replace(var_cada, cada)

        if new_name.find("VENOUS_DIS") == 0 :
            unit_str = "0:no, 1:yes"
            remark_str = "ok"
            recode_srt = """cada={2:0}; asps={}"""
            exec(recode_srt)
            var_cada = replace(var_cada, cada)

        if new_name.find("FAST_GLUC") == 0 :
            unit_str = "mg/dl"
            remark_str = "ok, conv cadasil to mg/dl. 1 mmol/l = 180/10 mg/dL. (mm=180 for glucose)"
            recode_srt = """var_cada = to_mgdl(cadasil_base[cadasil_name], cadasil_base[cadasil_name+"C"], mm=180)"""
            if cadasil_name in cadasil_base.columns:
                exec(recode_srt)

        if new_name.find("HBA1C") == 0 :
            unit_str = "% or mg/dl"
            remark_str = "WARNING!! NOT STANDARDIZED: % for cadasil %, mg/dl for asps"
            recode_srt = """"""

        if (new_name.find("CHOL") == 0) or (new_name.find("HDL") == 0) \
            or (new_name.find("LDL") == 0):
            unit_str = "mg/dl"
            remark_str = "ok, conv cadasil to mg/dl. 1 mmol/L = 386.65/10 mg/dL. (mm=386.65 for C27H46O, HDL and LDL)"
            recode_srt = """var_cada = to_mgdl(cadasil_base[cadasil_name], cadasil_base[cadasil_name+"C"], mm=386.65)"""
            if cadasil_name in cadasil_base.columns:
                exec(recode_srt)

        if (new_name.find("TRIGLY") == 0):
            unit_str = "mg/dl"
            remark_str = "ok, conv cadasil to mg/dl. 1 mmol/L = 875/10 mg/dL. (mm=875 for triglycerides)"
            recode_srt = """var_cada = to_mgdl(cadasil_base[cadasil_name], cadasil_base[cadasil_name+"C"], mm=875)"""
            if cadasil_name in cadasil_base.columns:
                exec(recode_srt)

        if (new_name.find("HEMOGLOBIN") == 0) \
            or (new_name.find("LEUKO_COUNT") == 0) \
            or new_name.find("THROMBO_COUNT") == 0:
            unit_str = "g/dl"
            remark_str = "ok"

        if new_name.find("FIBRINOGEN") == 0 :
            unit_str = "mg/dl"
            remark_str = "ok, conv cadasil from g/L to mg/dL"
            if cadasil_name in cadasil_base.columns:
                recode_srt = """var_cada = cadasil_base[cadasil_name] * 100"""
                exec(recode_srt)
                var_cada = var_cada.tolist()

        if new_name == "AF_HIST":
            unit_str = "0:no, 1:yes"
            recode_srt = """cada={2:0}; asps={}; aspfs={"nein":0, "ja":1}"""
            exec(recode_srt)
            var_cada = replace(var_cada, cada)
            var_aspfs = replace(var_aspfs, aspfs)
        
        if IN_COMMON:  # could have been set to 0
            stats = do_stat(var_cada, var_asps, var_aspfs)
            ## Add to common DB
            common_db[new_name] = pd.Series(var_cada + var_asps + var_aspfs)
            variables.append([new_name, cadasil_name, asps_name, aspfs_name,
                              in_cadasil_base, in_asps_base, in_aspfs_base, unit_str]
                              + stats + [recode_srt, remark_str])
if True:
    ## MAPPING    
    mapping_summary = pd.DataFrame(variables,
                                   columns=['NEW_NAME', 'CAD_NAME', 'ASPS_NAME',  'ASPFS_NAME', 
                          'IN_CAD', 'IN_ASPS', 'IN_ASPFS', 'UNIT'] + stat_colnames + ['RECODING', 'REMARKS'])
    mapping_summary.index = range(mapping_summary.shape[0])
    print "Save summary\n%s\n%s" % (OUTPUT_MAPPING_SUMMARY_FILEPATH+".csv", OUTPUT_MAPPING_SUMMARY_FILEPATH+".html")
    mapping_summary.to_csv(OUTPUT_MAPPING_SUMMARY_FILEPATH+".csv", sep=",", index=False)
    mapping_summary.to_html(OUTPUT_MAPPING_SUMMARY_FILEPATH+".html")
    
    
    ## COMMON_DB
    common_db["ID"] = ["CAD_%i" % ID for ID in cadasil_base.ID] + \
                      ["ASPS_%i" % ID for ID in asps_base.ID] + \
                      ["ASPFS_%i" % ID for ID in aspfs_base.ID]
    common_db["BASE"] = ["CAD"] * len(cadasil_base.ID) + \
                        ["ASPS"] * len(asps_base.ID) + \
                        ["ASPFS"] * len(aspfs_base.ID)
    db_columns = ["ID", "BASE"] + mapping_summary["NEW_NAME"].tolist()
    common_db = pd.DataFrame(common_db, columns=db_columns)
    print "Save common DB\n%s" % OUTPUT_MERGE_CADASIL_ASPS_FILEPATH+".csv"
    common_db.to_csv(OUTPUT_MERGE_CADASIL_ASPS_FILEPATH+".csv", sep=",", index=False)
