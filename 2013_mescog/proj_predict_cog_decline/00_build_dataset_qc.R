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
OUTPUT_PATH = "/neurospin/mescog/proj_predict_cog_decline/data/dataset_clinic_niglob_20140128"

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

# USE NEW NAME
new_names= sapply(colnames(CLINIC), function(x){if(x %in% vars$CAD_NAME)return(vars$NEW_NAME[vars$CAD_NAME==x])else return(x)})
names(new_names) = NULL
print(data.frame(colnames(CLINIC), new_names))
colnames(CLINIC) = new_names
print(dim(CLINIC))
#378  34

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
# 366   8

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
## QC FR vs GR
################################################################################################
# PLOT
library(reshape)
library(ggplot2)

Dm = melt(DB[, !(colnames(DB) %in% "ID")])
Dm$variable_by_site = interaction(Dm$variable, Dm$SITE)
Dm = Dm[order(as.character(Dm$variable_by_site)), ]

pdf(paste(OUTPUT_PATH, "_qc_boxplot.pdf", sep=""), width=5, height=50)
p = ggplot(Dm, aes(SITE, value))
p = p + geom_boxplot() + facet_grid(variable~., scales="free")#, space="free")
print(p)
dev.off()

# stat csv
fr = DB$SITE == "FR"
gr = DB$SITE == "GE"

stat = NULL
for(name in colnames(DB)){
  #name = "TMTB_TIME@M36"
  if(!(name %in% c("ID", "SITE"))){
    v = DB[, name]
    tuple = list(NAME=name, 
    count_fr=sum(!is.na(v[fr])), count_gr=sum(!is.na(v[gr])),
    mu_fr=mean(v[fr], na.rm=T), mu_gr=mean(v[gr], na.rm=T),
    sd_fr=sd(v[fr], na.rm=T), sd_gr=sd(v[gr], na.rm=T),             
    min_fr=min(v[fr], na.rm=T), min_gr=min(v[gr], na.rm=T),
    max_fr=max(v[fr], na.rm=T), max_gr=max(v[gr], na.rm=T))
    if(is.null(stat))stat=data.frame(tuple)else stat = rbind(stat, data.frame(tuple))
  }
}
options(width=200)
print(stat)
#library(gdata)
write.csv(stat, paste(OUTPUT_PATH, "_qc_summary.csv", sep=""), row.names=FALSE)

################################################################################################
## Impute missing data by mean
################################################################################################
source(paste(SRC,"utils.R",sep="/"))


skip = c(colnames(DB)[grep("M36",colnames(DB))], c("ID", "SITE"))
imput = imput_missing(DB, skip)

write.csv(DB, paste(OUTPUT_PATH, ".csv", sep=""), row.names=FALSE)
write.csv(imput$Dimputed, paste(OUTPUT_PATH, "_imputed.csv", sep=""), row.names=FALSE)
write.csv(imput$models, paste(OUTPUT_PATH, "_imputed_models.csv", sep=""), row.names=FALSE)
