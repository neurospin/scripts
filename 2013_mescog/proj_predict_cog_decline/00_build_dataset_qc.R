# install.packages("reshape")
# install.packages("ggplot2")
# install.packages("gdata")

WD  = paste(Sys.getenv("HOME"),"data/2014_mescog_predict_cog_decline",sep="/")
SRC = paste(Sys.getenv("HOME"),"git/scripts/2013_mescog/proj_predict_cog_decline",sep="/")

INPUT_PATH = "/neurospin/mescog"
CLINIC_PATH = paste(INPUT_PATH, "clinic", "base_commun_20140109.csv", sep="/")
CLINIC_MAPPING_PATH = paste(INPUT_PATH, "commondb_clinic_cadasil-asps-aspfs_mapping-summary_20131015.csv", sep="/")
NIGLOB_PATH = paste(INPUT_PATH, "neuroimaging/original", "baseCADASIL_imagerie.csv", sep="/")


VARIABLES_PATH = paste(SRC,"variables.csv",sep="/")

################################################################################################
## Names mapping
################################################################################################
mapping = read.csv(CLINIC_MAPPING_PATH, header=TRUE, strip.white=TRUE, as.is=TRUE)[, c("NEW_NAME", "CAD_NAME")]
vars = read.table(VARIABLES_PATH, sep =":", header=TRUE, strip.white=TRUE, as.is=TRUE)

## Find NEW_NAME using mapping
# new_name = c()
# for(v in vars$CAD_NAME){
#   if(v %in%  mapping$CAD_NAME) new_name = c(new_name, mapping[mapping$CAD_NAME==v, "NEW_NAME"]) else new_name = c(new_name, NA)
# }
# vars$NEW_NAME = new_name
# => Manual modification of "variables.csv"


################################################################################################
## QC FR vs GR
################################################################################################
vars = read.table(VARIABLES_PATH, sep =":", header=TRUE, strip.white=TRUE, as.is=TRUE)
D = read.csv(CLINIC_PATH)

################################################################################################
## Remove outliers 
################################################################################################
#18         HOMOCY17        HOMOCYSTEIN      226      123  11.8454425  14.8065041   4.2199696  34.9019604  5.09000    5.2  26.5900  396.0
D$HOMOCY17[D$HOMOCY17 > 100] = NA
#D$HOMOCY17[D$HOMOCY17 > 100]

D$DELTA_BP =  D$PAS - D$PAD
D = D[, c("ID", vars$CAD_NAME)]
D$SITE=NA
D$SITE[D$ID < 2000] = "FR"
D$SITE[D$ID >= 2000] = "GR"

# PLOT
library(reshape)
library(ggplot2)

Dm = melt(D[, !(colnames(D) %in% "ID")])
Dm$variable_by_site = interaction(Dm$variable, Dm$SITE)
Dm = Dm[order(as.character(Dm$variable_by_site)), ]

pdf(paste(WD, "data", "qc_dataset.pdf", sep="/"), width=5, height=50)
p = ggplot(Dm, aes(SITE, value))
p = p + geom_boxplot() + facet_grid(variable~., scales="free")#, space="free")
print(p)
dev.off()

# stat csv
fr = D$SITE=="FR"
gr = D$SITE=="GR"

stat = NULL
for(i in 1:nrow(vars)){
  if(vars[i, "CAD_NAME"] %in% colnames(D)){
    v = D[, vars[i, "CAD_NAME"]]
    tuple = list(CAD_NAME=vars[i, "CAD_NAME"], NAME=vars[i, "NEW_NAME"], 
    count_fr=sum(!is.na(v[fr])), count_gr=sum(!is.na(v[gr])),
    mu_fr=mean(v[fr], na.rm=T), mu_gr=mean(v[gr], na.rm=T),
    sd_fr=sd(v[fr], na.rm=T), sd_gr=sd(v[gr], na.rm=T),             
    min_fr=min(v[fr], na.rm=T), min_gr=min(v[gr], na.rm=T),
    max_fr=max(v[fr], na.rm=T), max_gr=max(v[gr], na.rm=T))
  }
  else{tuple=list(CAD_NAME=vars[i, "CAD_NAME"], NAME=vars[i, "NEW_NAME"], count=NA,
            mu_fr=NA, mu_gr=NA,
            min_fr=NA, min_gr=NA,
            max_fr=NA, max_gr=NA)}
  if(is.null(stat))stat=data.frame(tuple)else stat = rbind(stat, data.frame(tuple))
}
options(width=200)
print(stat)
library(gdata)
write.fwf(stat, paste(WD, "data", "qc_dataset.csv", sep="/"), rownames=FALSE)

################################################################################################
## Impute missing data by mean
################################################################################################
stat$CAD_NAME = as.character(stat$CAD_NAME)
stat$NAME = as.character(stat$NAME)

targets = stat$CAD_NAME[grep("@M36", stat$NAME)]
predictors = colnames(D)[!(colnames(D) %in% c("ID", "SITE", targets))]

for(v in predictors){
  D[is.na(D[,v]), v] = mean(D[,v], na.rm=T)
}

################################################################################################
## Impute missing data by mean
################################################################################################
IM = read.csv(NIGLOB_PATH, header=TRUE)[, c("Patient.Identifier" , "Time.Point", "Lesion.Volume..mm3.", "Lacunes.Volume..mm3.", "MB.Number")]
colnames(read.csv(NIGLOB_PATH, header=TRUE))
