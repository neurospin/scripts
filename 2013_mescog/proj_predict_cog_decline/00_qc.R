WD  = paste(Sys.getenv("HOME"),"data/2014_mescog_predict_cog_decline",sep="/")
SRC = paste(Sys.getenv("HOME"),"git/scripts/2013_mescog/proj_predict_cog_decline",sep="/")
DATASET_PATH = paste(WD, "data", "base_commun_20131011.csv", sep="/")



## Load dataset
## ============
source(paste(SRC,"variables.R",sep="/"))
D = read.csv(DATASET_PATH)
allvar = c(target, demographic, vascular, clinical)
fr = D$ID < 2000
gr = D$ID >= 2000
var = "TMTBT42"
stat = NULL
for(var in allvar){
  if(var %in% colnames(D)){
    v = D[, var]
    tuple = list(var=var, count=sum(!is.na(v)),
    mu_fr=mean(v[fr], na.rm=T), mu_gr=mean(v[gr], na.rm=T),
    min_fr=min(v[fr], na.rm=T), min_gr=min(v[gr], na.rm=T),
    max_fr=max(v[fr], na.rm=T), max_gr=max(v[gr], na.rm=T))
  }
  else{tuple=list(var=var, count=NA,
            mu_fr=NA, mu_gr=NA,
            min_fr=NA, min_gr=NA,
            max_fr=NA, max_gr=NA)}
  if(is.null(stat))stat=data.frame(tuple)else stat = rbind(stat, data.frame(tuple))
}
  
colnames(D)
dim(D) # 378 1287
