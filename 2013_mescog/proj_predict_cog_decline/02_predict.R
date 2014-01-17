#install.packages("glmnet")
require(glmnet)
require(ggplot2)

SRC = paste(Sys.getenv("HOME"),"git/scripts/2013_mescog/proj_predict_cog_decline",sep="/")
BASE_DIR = "/neurospin/mescog/2014_mescog_predict_cog_decline"
#setwd(WD)
INPUT_DATA = paste(BASE_DIR, "data", "dataset_clinic_niglob_20140110.csv", sep="/")
source(paste(SRC,"utils.R",sep="/"))
OUTPUT = paste(BASE_DIR, "results_201401", sep="/")

OUTPUT_SUMMARY = paste(OUTPUT, "results_summary.csv")

################################################################################################
## READ INPUT
################################################################################################
db = read_db(INPUT_DATA)
dim(db$DB_FR)# 239  42
dim(db$DB_GR)# 126  42
# remove normalized niglob variables
db$col_niglob = db$col_niglob[!(db$col_niglob %in% c("LLVn", "WMHVn"))]

#lgenim()
#FORCE.ALPHA=1 #lasso
FORCE.ALPHA=.95 #enet
MOD.SEL.CV = "manual.cv.lambda.min"
#MOD.SEL.CV = "manual.cv.lambda.min.glm"
#MOD.SEL.CV = "manual.cv.lambda.1sd"
#MOD.SEL.CV = "manual.cv.lambda.1sd.glm"
#MOD.SEL.CV = "auto.max.r2"
#MOD.SEL.CV = "auto.max.cor"
#MOD.SEL.CV = "auto.min.mse"

BOOTSTRAP.NB=10; PERMUTATION.NB=10
#DATA_STR = "FR"
DATA_STR = "GR"
RESULTS = NULL


if(DATA_STR == "FR"){
  DBLEARN = db$DB_FR
  DBTEST = db$DB_GR
}
if(DATA_STR == "GR"){
  DBLEARN = db$DB_GR
  DBTEST = db$DB_FR
}



################################################################################################
## M36~BASELINE
################################################################################################

PREDICTORS_STR = "BASELINE_NOINTER"
for(TARGET in db$col_targets){
  #TARGET = "TMTB_TIME.M36"
  #TARGET = "MDRS_TOTAL.M36"
  PREDICTORS = strsplit(TARGET, "[.]")[[1]][1]
  PREFIX = paste(OUTPUT, "/", DATA_STR, "_GLM", TARGET, "~", PREDICTORS_STR, sep="")
  if (!file.exists(PREFIX)) dir.create(PREFIX)
  setwd(PREFIX)
  DBtr = DBLEARN[!is.na(DBLEARN[, TARGET]), c(TARGET, PREDICTORS)]
  DBte = DBTEST[!is.na(DBTEST[, TARGET]), c(TARGET, PREDICTORS)]
  res = do.a.lot.of.things.glm(DBtr, DBte, TARGET, FALSE, PREDICTORS, DATA_STR, PREDICTORS_STR,
                         BOOTSTRAP.NB, PERMUTATION.NB)
  if(is.null(RESULTS)) RESULTS = data.frame(as.list(res)) else RESULTS = rbind(RESULTS, data.frame(as.list(res)))
}
PREDICTORS_STR = "BASELINE_INTER"
for(TARGET in db$col_targets){
  #TARGET = "TMTB_TIME.M36"
  #TARGET = "MDRS_TOTAL.M36"
  PREDICTORS = strsplit(TARGET, "[.]")[[1]][1]
  PREFIX = paste(OUTPUT, "/", DATA_STR, "_GLM", TARGET, "~", PREDICTORS_STR, sep="")
  if (!file.exists(PREFIX)) dir.create(PREFIX)
  setwd(PREFIX)
  DBtr = DBLEARN[!is.na(DBLEARN[, TARGET]), c(TARGET, PREDICTORS)]
  DBte = DBTEST[!is.na(DBTEST[, TARGET]), c(TARGET, PREDICTORS)]
  res = do.a.lot.of.things.glm(DBtr, DBte, TARGET, TRUE, PREDICTORS, DATA_STR, PREDICTORS_STR,
                               BOOTSTRAP.NB, PERMUTATION.NB)
  if(is.null(RESULTS)) RESULTS = data.frame(as.list(res)) else RESULTS = rbind(RESULTS, data.frame(as.list(res)))
}

################################################################################################
## M36~clin
################################################################################################
PREDICTORS_STR = "BASELINE+CLINIC"

for(TARGET in db$col_targets){
  #TARGET = "TMTB_TIME.M36"
  PREDICTORS = db$col_clinic
  PREFIX = paste(OUTPUT, "/", DATA_STR, "_ENET", TARGET, "~", PREDICTORS_STR, sep="")
  if (!file.exists(PREFIX)) dir.create(PREFIX)
  setwd(PREFIX)
  DBtr = DBLEARN[!is.na(DBLEARN[, TARGET]), c(TARGET, PREDICTORS)]
  DBte = DBTEST[!is.na(DBTEST[, TARGET]), c(TARGET, PREDICTORS)]
  ## FR CV
  cv = cross_val(length(DBtr[,TARGET]),type="k-fold", k=10, random=TRUE, seed=97)
  dump("cv",file="cv.schema.R")

  Xtr = as.matrix(DBtr[, PREDICTORS])
  ytr = DBtr[, TARGET]
  Xte = as.matrix(DBte[, PREDICTORS])
  yte = DBte[, TARGET]
  #source(paste(SRC,"utils.R",sep="/"))
  #X=Xtr; y=ytr; log_file=LOG_FILE; bootstrap.nb=1;permutation.nb=1#TARGET FORCE.ALPHA, MOD.SEL.CV
  CV = do.a.lot.of.things.glmnet(X=Xtr, y=ytr, DATA_STR, TARGET, PREDICTORS_STR, FORCE.ALPHA, MOD.SEL.CV,
                                  bootstrap.nb=BOOTSTRAP.NB, permutation.nb=PERMUTATION.NB)
  #Xtr=Xtr; ytr=ytr; Xte=Xte; yte=yte; TARGET=TARGET; log_file=LOG_FILE; permutation.nb=PERMUTATION.NB
  TEST = generalize.on.test.dataset(Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, TARGET=TARGET,PREDICTORS_STR=PREDICTORS_STR,
                                    permutation.nb=PERMUTATION.NB)
  if(is.null(RESULTS)) RESULTS = data.frame(c(CV, TEST)) else RESULTS = rbind(RESULTS, data.frame(c(CV, TEST)))
}

################################################################################################
## M36~NIGLOB
################################################################################################
PREDICTORS_STR = "NIGLOB"

for(TARGET in db$col_targets){
  #TARGET =  "MMSE.M36"
  #TARGET = "TMTB_TIME.M36"
  #TARGET = "MMSE.M36"
  PREDICTORS = db$col_niglob
  PREFIX = paste(OUTPUT, "/", DATA_STR, "_ENET", TARGET, "~", PREDICTORS_STR, sep="")
  if (!file.exists(PREFIX)) dir.create(PREFIX)
  setwd(PREFIX)
  DBtr = DBLEARN[!is.na(DBLEARN[, TARGET]), c(TARGET, PREDICTORS)]
  DBte = DBTEST[!is.na(DBTEST[, TARGET]), c(TARGET, PREDICTORS)]
  ## FR CV
  y = DBtr[,TARGET]
  cv = cross_val(length(y),type="k-fold", k=10, random=TRUE, seed=97)
  dump("cv",file="cv.schema.R")
  
  Xtr = as.matrix(DBtr[, PREDICTORS])
  ytr = DBtr[, TARGET]
  Xte = as.matrix(DBte[, PREDICTORS])
  yte = DBte[, TARGET]
  CV = do.a.lot.of.things.glmnet(X=Xtr, y=ytr, DATA_STR, TARGET, PREDICTORS_STR, FORCE.ALPHA, MOD.SEL.CV,
                                 bootstrap.nb=BOOTSTRAP.NB, permutation.nb=PERMUTATION.NB)
  TEST = generalize.on.test.dataset(Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, TARGET=TARGET,PREDICTORS_STR=PREDICTORS_STR,
                                    permutation.nb=PERMUTATION.NB)
  if(is.null(RESULTS)) RESULTS = data.frame(c(CV, TEST)) else RESULTS = rbind(RESULTS, data.frame(c(CV, TEST)))
}

################################################################################################
## M36~BASELINE+NIGLOB
################################################################################################
PREDICTORS_STR = "BASELINE+NIGLOB"

for(TARGET in db$col_targets){
  #TARGET = "TMTB_TIME.M36"
  baseline = strsplit(TARGET, "[.]")[[1]][1]
  PREDICTORS = c(baseline, db$col_niglob)
  PREFIX = paste(OUTPUT, "/", DATA_STR, "_", TARGET, "~", PREDICTORS_STR, sep="")
  if (!file.exists(PREFIX)) dir.create(PREFIX)
  setwd(PREFIX)
  DBtr = DBLEARN[!is.na(DBLEARN[, TARGET]), c(TARGET, PREDICTORS)]
  DBte = DBTEST[!is.na(DBTEST[, TARGET]), c(TARGET, PREDICTORS)]
  ## FR CV
  y = DBtr[,TARGET]
  cv = cross_val(length(y),type="k-fold", k=10, random=TRUE, seed=97)
  dump("cv",file="cv.schema.R")
  
  Xtr = as.matrix(DBtr[, PREDICTORS])
  ytr = DBtr[, TARGET]
  Xte = as.matrix(DBte[, PREDICTORS])
  yte = DBte[, TARGET]
  CV = do.a.lot.of.things.glmnet(X=Xtr, y=ytr, DATA_STR, TARGET, PREDICTORS_STR, FORCE.ALPHA, MOD.SEL.CV,
                                 bootstrap.nb=BOOTSTRAP.NB, permutation.nb=PERMUTATION.NB)
  TEST = generalize.on.test.dataset(Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, TARGET=TARGET,PREDICTORS_STR=PREDICTORS_STR,
                                    permutation.nb=PERMUTATION.NB)
  if(is.null(RESULTS)) RESULTS = data.frame(c(CV, TEST)) else RESULTS = rbind(RESULTS, data.frame(c(CV, TEST)))
}

PREDICTORS_STR = "BASELINE+NIGLOB_NOINTER"
for(TARGET in db$col_targets){
  #TARGET = "TMTB_TIME.M36"
  #TARGET = "MDRS_TOTAL.M36"
  baseline = strsplit(TARGET, "[.]")[[1]][1]
  PREDICTORS = c(baseline, db$col_niglob)
  PREFIX = paste(OUTPUT, "/", DATA_STR, "_GLM", TARGET, "~", PREDICTORS_STR, sep="")
  if (!file.exists(PREFIX)) dir.create(PREFIX)
  setwd(PREFIX)
  DBtr = DBLEARN[!is.na(DBLEARN[, TARGET]), c(TARGET, PREDICTORS)]
  DBte = DBTEST[!is.na(DBTEST[, TARGET]), c(TARGET, PREDICTORS)]
  res = do.a.lot.of.things.glm(DBtr, DBte, TARGET, FALSE, PREDICTORS, DATA_STR, PREDICTORS_STR,
                               BOOTSTRAP.NB, PERMUTATION.NB)
  if(is.null(RESULTS)) RESULTS = data.frame(as.list(res)) else RESULTS = rbind(RESULTS, data.frame(as.list(res)))
}
PREDICTORS_STR = "BASELINE+NIGLOB_INTER"
for(TARGET in db$col_targets){
  #TARGET = "TMTB_TIME.M36"
  #TARGET = "MDRS_TOTAL.M36"
  baseline = strsplit(TARGET, "[.]")[[1]][1]
  PREDICTORS = c(baseline, db$col_niglob)
  PREFIX = paste(OUTPUT, "/", DATA_STR, "_GLM", TARGET, "~", PREDICTORS_STR, sep="")
  if (!file.exists(PREFIX)) dir.create(PREFIX)
  setwd(PREFIX)
  DBtr = DBLEARN[!is.na(DBLEARN[, TARGET]), c(TARGET, PREDICTORS)]
  DBte = DBTEST[!is.na(DBTEST[, TARGET]), c(TARGET, PREDICTORS)]
  res = do.a.lot.of.things.glm(DBtr, DBte, TARGET, TRUE, PREDICTORS, DATA_STR, PREDICTORS_STR,
                               BOOTSTRAP.NB, PERMUTATION.NB)
  if(is.null(RESULTS)) RESULTS = data.frame(as.list(res)) else RESULTS = rbind(RESULTS, data.frame(as.list(res)))
}

################################################################################################
## M36~CLINIC+NIGLOB
################################################################################################
PREDICTORS_STR = "BASELINE+CLINIC+NIGLOB"

for(TARGET in db$col_targets){
  #TARGET = "TMTB_TIME.M36"
  PREDICTORS = c(db$col_clinic, db$col_niglob)
  PREFIX = paste(OUTPUT, "/", DATA_STR, "_", TARGET, "~", PREDICTORS_STR, sep="")
  if (!file.exists(PREFIX)) dir.create(PREFIX)
  setwd(PREFIX)
  DBtr = DBLEARN[!is.na(DBLEARN[, TARGET]), c(TARGET, PREDICTORS)]
  DBte = DBTEST[!is.na(DBTEST[, TARGET]), c(TARGET, PREDICTORS)]
  ## FR CV
  y = DBtr[,TARGET]
  cv = cross_val(length(y),type="k-fold", k=10, random=TRUE, seed=97)
  dump("cv",file="cv.schema.R")
  
  Xtr = as.matrix(DBtr[, PREDICTORS])
  ytr = DBtr[, TARGET]
  Xte = as.matrix(DBte[, PREDICTORS])
  yte = DBte[, TARGET]
  CV = do.a.lot.of.things.glmnet(X=Xtr, y=ytr, DATA_STR, TARGET, PREDICTORS_STR, FORCE.ALPHA, MOD.SEL.CV,
                                 bootstrap.nb=BOOTSTRAP.NB, permutation.nb=PERMUTATION.NB)
  TEST = generalize.on.test.dataset(Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, TARGET=TARGET,PREDICTORS_STR=PREDICTORS_STR,
                                    permutation.nb=PERMUTATION.NB)
  if(is.null(RESULTS)) RESULTS = data.frame(c(CV, TEST)) else RESULTS = rbind(RESULTS, data.frame(c(CV, TEST)))
}

write.csv(RESULTS, OUTPUT_SUMMARY, row.names=FALSE)

# rsync -azvun --delete /neurospin/mescog/2014_mescog_predict_cog_decline ~/data/



# 
# for(TARGET in db$col_targets){
#     #TARGET = "TMTB_TIME.M36"
#     #TARGET = "MDRS_TOTAL.M36"
#     PREDICTORS = strsplit(TARGET, "[.]")[[1]][1]
#     PREFIX = paste(OUTPUT, "/", DATA_STR, "_", TARGET, "~", PREDICTORS_STR, sep="")
#     if (!file.exists(PREFIX)) dir.create(PREFIX)
#     setwd(PREFIX)
#     DBtr = DBLEARN[!is.na(DBLEARN[, TARGET]), c(TARGET, PREDICTORS)]
#     DBte = DBTEST[!is.na(DBTEST[, TARGET]), c(TARGET, PREDICTORS)]
#     ## FR CV
#     y = DBtr[,TARGET]
#     cv = cross_val(length(y),type="k-fold", k=10, random=TRUE, seed=97)
#     dump("cv",file="cv.schema.R")
#     
#     y.pred.cv = c();  y.true.cv = c()
#     for(test in cv){
#         DBtr_train = DBtr[-test,]
#         DBtr_test = DBtr[test,]
#         formula = formula(paste(TARGET,"~",PREDICTORS))
#         modlm = lm(formula, data = DBtr_train)
#         y.pred.cv = c(y.pred.cv, predict(modlm, DBtr_test))
#         y.true.cv = c(y.true.cv, DBtr_test[,TARGET])
#     }
#     loss.cv = round(loss.reg(y.true.cv, y.pred.cv, df2=2),digit=2)
#     ## CV predictions
#     ## ==============
# 
#     cat("\nCV predictions\n",file=LOG_FILE,append=TRUE)
#     cat("--------------\n",file=LOG_FILE,append=TRUE)
# 
#     sink(LOG_FILE, append = TRUE)
#     print(loss.cv)
#     sink()
# 
#     ## Fit and predict all (same train) data, (look for overfit)
#     ## =========================================================
# 
#     cat("\nFit and predict all (same train) data, (look for overfit)\n",file=LOG_FILE,append=TRUE)
#     cat("---------------------------------------------------------\n",file=LOG_FILE,append=TRUE)
#     #formula = formula(paste(TARGET,"~", PREDICTORS))
#     modlm.fr = lm(formula, data = DBtr)
#     y.pred.all = predict(modlm.fr, DBtr)
#     loss.all = round(loss.reg(y, y.pred.all, df=2),digit=2)
#     sink(LOG_FILE, append = TRUE)
#     print(loss.all)
#     sink()
#     
#     ## Plot true vs predicted
#     ## ======================
#     
#     pdf("cv_glm_true-vs-pred.pdf")
#     p = qplot(y.true.cv, y.pred.cv, geom = c("smooth","point"), method="lm",
#         main=paste(TARGET," - true vs. pred - CV - [R2_10CV=",loss.cv[["R2"]],"]",sep=""))
#     print(p); dev.off()
#   
#     # - true vs. pred (no CV)
#     
#     #svg("all.bestcv.glmnet.true-vs-pred.svg")
#     pdf("all_glm_true-vs-pred.pdf")
#     p = qplot(y, y.pred.all, geom = c("smooth","point"), method="lm",
#         main=paste(TARGET," - true vs. pred - no CV - [R2=", loss.all[["R2"]],"]",sep=""))
#     print(p); dev.off()
# 
#     cat("\nGeneralize on Test dataset:\n",file=LOG_FILE,append=TRUE)
#     cat("-------------------------------\n",file=LOG_FILE,append=TRUE)
#     y.preDBtr = predict(modlm.fr, DBtr)
#     y.preDBte  = predict(modlm.fr, DBte)
# 
#     loss.fr = round(loss.reg(DBtr[,TARGET], y.preDBtr, df2=2), digit=2)
#     loss.d  = round(loss.reg(DBte[,TARGET],  y.preDBte,  df2=2), digit=2)
#     
#     pdf("all_glm_true-vs-preDBte.pdf")
#     y.preDBte = as.vector(y.preDBte)
#     p = qplot(DBte[,TARGET], y.preDBte, geom = c("smooth","point"), method="lm",
#         main=paste(TARGET," - true vs. pred - TEST - [R2=", loss.d[["R2"]] ,"]",sep=""))
#     print(p);dev.off()    
#     
#     sink(LOG_FILE, append = TRUE)
#     print(rbind(c(center=1,loss.fr),c(center=2,loss.d)))
#     sink()
#     
#     res = data.frame(data=DATA_STR, target=TARGET, predictors="BASELINE", dim=paste(dim(DBtr)[1], dim(DBtr)[2]-1, sep="x"),
#       r2_cv=loss.cv["R2"], cor_cv=loss.cv["cor"], fstat_cv=loss.cv["fstat"],
#       r2_all=loss.all["R2"], cor_all=loss.all["R2"], r2_test=loss.d["R2"], cor_test=loss.d["cor"], fstat_test=loss.d["fstat"])
#     if(is.null(RESULTS)) RESULTS = res else RESULTS = rbind(RESULTS, res)
# }