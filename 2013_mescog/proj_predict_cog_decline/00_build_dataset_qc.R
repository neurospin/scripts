# install.packages("reshape")
# install.packages("ggplot2")

WD  = paste(Sys.getenv("HOME"),"data/2014_mescog_predict_cog_decline",sep="/")
SRC = paste(Sys.getenv("HOME"),"git/scripts/2013_mescog/proj_predict_cog_decline",sep="/")
INPUT_PATH = "/neurospin/mescog/clinic"
DATA_CADA_PATH = paste(INPUT_PATH, "base_commun_20131011.csv", sep="/")
MAPPING_PATH = paste(INPUT_PATH, "commondb_clinic_cadasil-asps-aspfs_mapping-summary_20131015.csv", sep="/")
VARIABLES_PATH = paste(SRC,"variables.csv",sep="/")

################################################################################################
## Names mapping
################################################################################################
mapping = read.csv(MAPPING_PATH, header=TRUE, strip.white=TRUE, as.is=TRUE)[, c("NEW_NAME", "CAD_NAME")]
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
D = read.csv(DATA_CADA_PATH)
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
write.csv(stat, paste(WD, "data", "qc_dataset.csv", sep="/"), row.names=FALSE)

PB:
           CAD_NAME               NAME count_fr count_gr      mu_fr       mu_gr     min_fr min_gr    max_fr max_gr
15        CHOLHDL17                HDL      247      126   4.275181  57.0873016  0.7399998   24.0  81.99997  117.0
17         TRIGLY17             TRIGLY      248      126   6.622711 150.0793651  0.3399999   32.0 201.99986  685.0
18         HOMOCY17        HOMOCYSTEIN      226      123  11.845442  14.8065041  5.0900000    5.2  26.59000  396.0 outlier replace per mean
20            CRP17              CRP17      234      125   5.542735   0.3296000  1.0000000    0.1  66.00000    6.1
21           GLYC17          FAST_GLUC      245       11   8.407344 111.0909091  3.6999989   78.0 130.99995  228.0
23         MIGAAURA MIGRAINE_WITH_AURA       90       49   1.000000   1.0000000  1.0000000    1.0   1.00000    1.0
32       INDEXNIHSS              NIHSS      248      129   1.669355   0.8527132  0.0000000    0.0  25.00000    7.0
