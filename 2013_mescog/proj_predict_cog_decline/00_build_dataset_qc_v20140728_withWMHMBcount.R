# install.packages("reshape")
# install.packages("ggplot2")
# install.packages("gdata")

SRC = paste(Sys.getenv("HOME"),"git/scripts/2013_mescog/proj_predict_cog_decline",sep="/")
BASEDIR = "/neurospin/mescog"
# Var to use ---
VARIABLES_PATH = paste(SRC,"variables.csv",sep="/")

# INPUT ---
CLINIC_PATH = paste(BASEDIR, "clinic", "base_commun_20140109.csv", sep="/")
CLINIC_MAPPING_PATH = paste(BASEDIR, "commondb_clinic_cadasil-asps-aspfs_mapping-summary_20131015.csv", sep="/")
# NI GLOBAL
##NIGLOB_PATH = paste(BASEDIR, "neuroimaging/original/global", "baseCADASIL_imagerie.csv", sep="/")
NIGLOBVOL_DIRPATH = paste(BASEDIR, "neuroimaging/original/munich/CAD_database_soures/global imaging variables", sep="/")
NIGLOB_Bioclinica_PATH = paste(NIGLOBVOL_DIRPATH, "CAD_M0_Bioclinica.txt", sep="/")
NIGLOB_Bioclinica_M36_PATH = paste(NIGLOBVOL_DIRPATH, "CAD_M36_Bioclinica.txt", sep="/")
NIGLOB_Sienax_PATH = paste(NIGLOBVOL_DIRPATH, "CAD_M0_Sienax.txt", sep="/")
NIGLOB_WMHV_PATH = paste(NIGLOBVOL_DIRPATH, "CAD_M0_WMHV.txt", sep="/")

# OUTPUT ---
#OUTPUT_PATH = "/neurospin/mescog/proj_predict_cog_decline/data/dataset_clinic_niglob_20140128"
#OUTPUT_PATH = "/neurospin/mescog/proj_predict_cog_decline/data/dataset_clinic_niglob_20140204"
# OUTPUT_PATH = "/neurospin/mescog/proj_predict_cog_decline/data/dataset_clinic_niglob_20140205"
OUTPUT_PATH = "/neurospin/mescog/proj_predict_cog_decline/data/dataset_clinic_niglob_20140728"

################################################################################################
## Names mapping
################################################################################################
# mapping = read.csv(CLINIC_MAPPING_PATH, header=TRUE, strip.white=TRUE, as.is=TRUE)[, c("NEW_NAME", "CAD_NAME")]
# vars = read.table(VARIABLES_PATH, sep =":", header=TRUE, strip.white=TRUE, as.is=TRUE)
## Find NEW_NAME using mapping
# new_name = c()
# for(v in vars$CAD_NAME){
#   if(v %in%  mapping$CAD_NAME) new_name = c(new_name, mapping[mapping$CAD_NAME==v, "NEW_NAME"]) else new_name = c(new_name, NA)
# }
# vars$NEW_NAME = new_name
# => Manual modification of "variables.csv"

################################################################################################
## READ CLINIC
################################################################################################
CLINIC = read.csv(CLINIC_PATH)
CLINIC$DELTA_BP =  CLINIC$PAS - CLINIC$PAD
vars = read.table(VARIABLES_PATH, sep =":", header=TRUE, strip.white=TRUE, as.is=TRUE)
CLINIC = CLINIC[, c("ID", vars$CAD_NAME)]
CLINIC$ID = paste("CAD_", as.character(CLINIC$ID), sep="")
#18         HOMOCY17        HOMOCYSTEIN      226      123  11.8454425  14.8065041   4.2199696  34.9019604  5.09000    5.2  26.5900  396.0
CLINIC$HOMOCY17[CLINIC$HOMOCY17 > 100] = NA
#CLINIC$HOMOCY17[CLINIC$HOMOCY17 > 100]

CLINIC$CHOLLDL17 = as.double(as.character(CLINIC$CHOLLDL17))

# USE NEW NAME
new_names= sapply(colnames(CLINIC), function(x){if(x %in% vars$CAD_NAME)return(vars$NEW_NAME[vars$CAD_NAME==x])else return(x)})
names(new_names) = NULL
print(data.frame(colnames(CLINIC), new_names))
colnames(CLINIC) = new_names
print(dim(CLINIC))
#378  20

################################################################################################
## NI GLOBAL Measurments
################################################################################################
NIGLOB_Bioclinica = read.table(NIGLOB_Bioclinica_PATH, header=TRUE, as.is=TRUE)
NIGLOB_Bioclinica_M36 = read.table(NIGLOB_Bioclinica_M36_PATH, header=TRUE, as.is=TRUE)
NIGLOB_Sienax     = read.table(NIGLOB_Sienax_PATH, header=TRUE, as.is=TRUE)
NIGLOB_WMHV       = read.table(NIGLOB_WMHV_PATH, header=TRUE, as.is=TRUE)
M = merge(merge(merge(NIGLOB_Bioclinica, NIGLOB_Bioclinica_M36, by="ID", all=TRUE), 
                NIGLOB_Sienax, by="ID", all=TRUE),
          NIGLOB_WMHV, by="ID", all=TRUE)

NIGLOB = data.frame(ID=M$ID,
                    LLV = M$M0_LLV,
                    #LLVn = M$M0_LLV / M$M0_ICC,
                    LLcount = M$M0_LLcount,
                    WMHV = M$M0_WMHV,
                    #WMHVn = M$M0_WMHV / M$M0_ICC,
                    MBcount = M$M0_MBcount,
                    BPF = M$M0_SIENAX / M$M0_ICC,
                    #BRAINVOL = M$M0_SIENAX,
                    "LLV.M36" = M$M36_LLV,
                    "LLcount.M36" = M$M36_LLcount,
                    "MBcount.M36" = M$M36_MBcount, check.names=FALSE)

print(dim(NIGLOB))
# 373   9

################################################################################################
## Merge, keep only those with NI
################################################################################################
DB = merge(CLINIC, NIGLOB, by="ID")
print(dim(DB))
# 372  28
id = sapply(strsplit(DB$ID, "_"), function(x)as.integer(x[[2]]))
DB$SITE=NA
DB$SITE[id < 2000] = "FR"
DB$SITE[id >= 2000] = "GE"


################################################################################################
## Impute missing data
################################################################################################
source(paste(SRC,"utils.R",sep="/"))

# "/neurospin/mescog/proj_predict_cog_decline/data/dataset_clinic_niglob_20140205"

VARIABLES = variables()
# Remove subject with missing LLV or BPF
DB = DB[!(is.na(DB$BPF) | is.na(DB$LLV)), unlist(VARIABLES)]

# Impute by site
Dfr = DB[DB$SITE=="FR", ]
Dge = DB[DB$SITE=="GE", ]

missing = data.frame(var=colnames(DB), FR=sapply(Dfr, function(x)sum(is.na(x))), GE=sapply(Dge, function(x)sum(is.na(x))))
rownames(missing)=NULL
print(missing)
# var  FR GE
# 1                ID   0  0
# 2              SITE   0  0
# 3     TMTB_TIME.M36 104 40
# 4    MDRS_TOTAL.M36  94 38
# 5           MRS.M36  74 38
# 6          MMSE.M36  83 38
# 7         TMTB_TIME  34  9
# 8        MDRS_TOTAL  13  0
# 9               MRS   0  0
# 10             MMSE   8  2
# 11              LLV   0  0
# 12              BPF   0  0
# 13             WMHV   1  1
# 14          MBcount   0  0
# 15 AGE_AT_INCLUSION   0  0
# 16              SEX   0  0
# 17        EDUCATION   4  0
# 18           SYS_BP   6  0
# 19           DIA_BP   6  0
# 20          SMOKING  18  5
# 21              LDL   3  3
# 22      HOMOCYSTEIN  16  7
# 23            HBA1C   3  3
# 24            CRP17   9  4
# 25          ALCOHOL  21  3

imput_missing<-function(D, var){
  imput_model = NULL
  # Impute baseline using other baselines
  Di = imput_missing_lm(D, id=D$ID, to_impute=var$col_baselines, predictors=var$col_baselines)
  imput_model = rbind(imput_model, attr(Di, "models"))
  # Impute clinic by mean
  Di = imput_missing_mean(Di, id=D$ID, to_impute=var$col_clinic)
  attr(Di, "models")$model = as.factor(attr(Di, "models")$model)
  imput_model = rbind(imput_model, attr(Di, "models"))
  # Impute NIGLOB (except) "LLV", "BPF" by mean
  niglob_to_imput = var$col_niglob_full[!(var$col_niglob_full %in% c("LLV", "BPF"))]
  Di = imput_missing_mean(Di, id=Di$ID, to_impute=niglob_to_imput)
  attr(Di, "models")$model = as.factor(attr(Di, "models")$model)
  imput_model = rbind(imput_model, attr(Di, "models"))
  print(data.frame(sapply(Di, function(x)sum(is.na(x)))))
  return(list(D=Di, imput_rules=imput_model))
}

fr = imput_missing(Dfr, VARIABLES)
ge = imput_missing(Dge, VARIABLES)

Di = rbind(fr$D, ge$D)
D_imput_rules = rbind(fr$imput_rules, ge$imput_rules)
data.frame(lapply(Di , function(x)sum(is.na(x))))
#ID SITE TMTB_TIME.M36 MDRS_TOTAL.M36 MRS.M36 MMSE.M36 TMTB_TIME MDRS_TOTAL MRS MMSE LLV BPF WMHV MBcount AGE_AT_INCLUSION SEX EDUCATION
# 0    0           144            132     112      121         0          0   0    0   0   0    0       0                0   0         0
#SYS_BP DIA_BP SMOKING LDL HOMOCYSTEIN HBA1C CRP17 ALCOHOL
#    0      0       0   0           0     0     0       0
write.csv(DB, paste(OUTPUT_PATH, "_nomissing_BPF-LLV.csv", sep=""), row.names=FALSE)
write.csv(Di, paste(OUTPUT_PATH, "_nomissing_BPF-LLV_imputed.csv", sep=""), row.names=FALSE)
write.csv(D_imput_rules, paste(OUTPUT_PATH, "_nomissing_BPF-LLV_imputed_rules.csv", sep=""), row.names=FALSE)

################################################################################################
## Descriptive before imputation
################################################################################################

tmp = DB
# SEX cada={1:'m', 2:'f'}
tmp$SEX = as.factor(sub(2, "F", sub(1, 'M', tmp$SEX)))
# CADASIL: 1 = none, 2 = < 2 drinks a day ; 3 = > 2 drinks a day
tmp$ALCOHOL = as.factor(sub(1, "None", sub(2, '2 drink/day', sub(3, '>2 drink/day', round(tmp$ALCOHOL)))))
# cada={1:"current", 2:"never", 3:"former"}
tmp$SMOKING = as.factor(sub(1, "Current", sub(2, "Never", sub(3, "Former", round(tmp$SMOKING)))))

write.csv(tmp, file = "/tmp/db.csv", row.names=FALSE)

cmd = paste('python ~/git/datamind/descriptive/descriptive_statistics.py -i /tmp/db.csv ',
            '-o ', OUTPUT_PATH, '_nomissing_BPF-LLV_notimputed_descriptive.xls', ' -e "ID"', sep="")
system(cmd)

################################################################################################
## Descriptive after imputation
################################################################################################

tmp = Di
# SEX cada={1:'m', 2:'f'}
tmp$SEX = as.factor(sub(2, "F", sub(1, 'M', tmp$SEX)))
# CADASIL: 1 = none, 2 = < 2 drinks a day ; 3 = > 2 drinks a day
tmp$ALCOHOL = as.factor(sub(1, "None", sub(2, '2 drink/day', sub(3, '>2 drink/day', round(tmp$ALCOHOL)))))
# cada={1:"current", 2:"never", 3:"former"}
tmp$SMOKING = as.factor(sub(1, "Current", sub(2, "Never", sub(3, "Former", round(tmp$SMOKING)))))

write.csv(tmp, file = "/tmp/db.csv", row.names=FALSE)

cmd = paste('python ~/git/datamind/descriptive/descriptive_statistics.py -i /tmp/db.csv ',
            '-o ', OUTPUT_PATH, '_nomissing_BPF-LLV_imputed_descriptive.xls', ' -e "ID"', sep="")
system(cmd)

# DB$SEX = as.factor(DB$SEX)
# # stat csv
# fr = DB$SITE == "FR"
# gr = DB$SITE == "GE"
# stat = NULL
# for(name in colnames(DB)){
#   #name = "TMTB_TIME@M36"
#   cat(name, "\n")
#   if(!(name %in% c("ID", "SITE"))){
#     v = DB[, name]
#     tuple = list(NAME=name, 
#     count_fr=sum(!is.na(v[fr])), count_gr=sum(!is.na(v[gr])),
#     mu_fr=mean(v[fr], na.rm=T), mu_gr=mean(v[gr], na.rm=T),
#     sd_fr=sd(v[fr], na.rm=T), sd_gr=sd(v[gr], na.rm=T),             
#     min_fr=min(v[fr], na.rm=T), min_gr=min(v[gr], na.rm=T),
#     max_fr=max(v[fr], na.rm=T), max_gr=max(v[gr], na.rm=T))
#     if(is.null(stat))stat=data.frame(tuple)else stat = rbind(stat, data.frame(tuple))
#   }
# }
# 
# options(width=200)
# print(stat)
# #library(gdata)
# write.csv(stat, paste(OUTPUT_PATH, "_qc_summary.csv", sep=""), row.names=FALSE)

