#install.packages("glmnet")
require(glmnet)
require(ggplot2)

SRC = paste(Sys.getenv("HOME"),"git/scripts/2013_mescog/proj_predict_cog_decline",sep="/")
BASE_DIR = "/neurospin/mescog/2014_mescog_predict_cog_decline"
#setwd(WD)
INPUT_DATA = paste(BASE_DIR, "data", "dataset_clinic_niglob_20140110.csv", sep="/")
source(paste(SRC,"utils.R",sep="/"))
OUTPUT = paste(BASE_DIR, "results_201401", sep="/")
LOG_FILE = "log.txt"

OUTPUT_SUMMARY = paste(OUTPUT, "results_summary_with-norm-niglob.csv")

################################################################################################
## READ INPUT
################################################################################################
db = read_db(INPUT_DATA)
dim(db$DB_FR)# 239  42
dim(db$DB_GR)# 126  42

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
DATA_STR = "FR"
DBLEARN = db$DB_FR
DBTEST = db$DB_GR
RESULTS = NULL


################################################################################################
## M36~M0
################################################################################################

PREDICTORS_STR = "SIMPLE"

for(TARGET in db$col_targets){
    #TARGET = "TMTB_TIME.M36"
    PREDICTORS = strsplit(TARGET, "[.]")[[1]][1]
    PREFIX = paste(OUTPUT, "/", DATA_STR, "_", TARGET, "~", PREDICTORS_STR, sep="")
    if (!file.exists(PREFIX)) dir.create(PREFIX)
    setwd(PREFIX)
    D_LEARN = DBLEARN[!is.na(DBLEARN[, TARGET]), c(TARGET, PREDICTORS)]
    D_TEST = DBTEST[!is.na(DBTEST[, TARGET]), c(TARGET, PREDICTORS)]
    ## FR CV
    y = D_LEARN[,TARGET]
    cv = cross_val(length(y),type="k-fold", k=10, random=TRUE, seed=97)
    dump("cv",file="cv.schema.R")
    
    y.pred.cv = c();  y.true.cv = c()
    for(test in cv){
        D_LEARN_train = D_LEARN[-test,]
        D_LEARN_test = D_LEARN[test,]
        formula = formula(paste(TARGET,"~",PREDICTORS))
        modlm = lm(formula, data = D_LEARN_train)
        y.pred.cv = c(y.pred.cv, predict(modlm, D_LEARN_test))
        y.true.cv = c(y.true.cv, D_LEARN_test[,TARGET])
    }
    loss.cv = round(loss.reg(y.true.cv, y.pred.cv, df2=2),digit=2)
    ## CV predictions
    ## ==============

    cat("\nCV predictions\n",file=LOG_FILE,append=TRUE)
    cat("--------------\n",file=LOG_FILE,append=TRUE)

    sink(LOG_FILE, append = TRUE)
    print(loss.cv)
    sink()

    ## Fit and predict all (same train) data, (look for overfit)
    ## =========================================================

    cat("\nFit and predict all (same train) data, (look for overfit)\n",file=LOG_FILE,append=TRUE)
    cat("---------------------------------------------------------\n",file=LOG_FILE,append=TRUE)
    #formula = formula(paste(TARGET,"~", PREDICTORS))
    modlm.fr = lm(formula, data = D_LEARN)
    y.pred.all = predict(modlm.fr, D_LEARN)
    loss.all = round(loss.reg(y, y.pred.all, df=2),digit=2)
    sink(LOG_FILE, append = TRUE)
    print(loss.all)
    sink()
    
    ## Plot true vs predicted
    ## ======================
    
    pdf("cv_glm_true-vs-pred.pdf")
    p = qplot(y.true.cv, y.pred.cv, geom = c("smooth","point"), method="lm",
        main=paste(TARGET," - true vs. pred - CV - [R2_10CV=",loss.cv[["R2"]],"]",sep=""))
    print(p); dev.off()
  
    # - true vs. pred (no CV)
    
    #svg("all.bestcv.glmnet.true-vs-pred.svg")
    pdf("all_glm_true-vs-pred.pdf")
    p = qplot(y, y.pred.all, geom = c("smooth","point"), method="lm",
        main=paste(TARGET," - true vs. pred - no CV - [R2=", loss.all[["R2"]],"]",sep=""))
    print(p); dev.off()

    cat("\nGeneralize on German dataset:\n",file=LOG_FILE,append=TRUE)
    cat("-------------------------------\n",file=LOG_FILE,append=TRUE)
    y.preD_LEARN = predict(modlm.fr, D_LEARN)
    y.preD_TEST  = predict(modlm.fr, D_TEST)

    loss.fr = round(loss.reg(D_LEARN[,TARGET], y.preD_LEARN, df2=2), digit=2)
    loss.d  = round(loss.reg(D_TEST[,TARGET],  y.preD_TEST,  df2=2), digit=2)
    
    pdf("all_glm_true-vs-pred_test.pdf")
    y.preD_TEST = as.vector(y.preD_TEST)
    p = qplot(D_TEST[,TARGET], y.preD_TEST, geom = c("smooth","point"), method="lm",
        main=paste(TARGET," - true vs. pred - D - [R2=", loss.d[["R2"]] ,"]",sep=""))
    print(p);dev.off()    
    
    sink(LOG_FILE, append = TRUE)
    print(rbind(c(center=1,loss.fr),c(center=2,loss.d)))
    sink()
    
    res = data.frame(data=DATA_STR, target=TARGET, predictors="SIMPLE", dim=paste(dim(D_LEARN)[1], dim(D_LEARN)[2]-1, sep="x"),
      r2_cv=loss.cv["R2"], cor_cv=loss.cv["cor"], fstat_cv=loss.cv["fstat"],
      r2_all=loss.all["R2"], cor_all=loss.all["R2"], r2_test=loss.d["R2"], cor_test=loss.d["cor"], fstat_test=loss.d["fstat"])
    if(is.null(RESULTS)) RESULTS = res else RESULTS = rbind(RESULTS, res)
}


################################################################################################
## M36~clin
################################################################################################
PREDICTORS_STR = "CLINIC"

for(TARGET in db$col_targets){
  #TARGET = "TMTB_TIME.M36"
  PREDICTORS = db$col_clinic
  PREFIX = paste(OUTPUT, "/", DATA_STR, "_", TARGET, "~", PREDICTORS_STR, sep="")
  if (!file.exists(PREFIX)) dir.create(PREFIX)
  setwd(PREFIX)
  D_LEARN = DBLEARN[!is.na(DBLEARN[, TARGET]), c(TARGET, PREDICTORS)]
  D_TEST = DBTEST[!is.na(DBTEST[, TARGET]), c(TARGET, PREDICTORS)]
  ## FR CV
  cv = cross_val(length(D_LEARN[,TARGET]),type="k-fold", k=10, random=TRUE, seed=97)
  dump("cv",file="cv.schema.R")

  Xtr = as.matrix(D_LEARN[, PREDICTORS])
  ytr = D_LEARN[, TARGET]
  Xte = as.matrix(D_TEST[, PREDICTORS])
  yte = D_TEST[, TARGET]
  #source(paste(SRC,"utils.R",sep="/"))
  #X=Xtr; y=ytr; log_file=LOG_FILE; bootstrap.nb=1;permutation.nb=1#TARGET FORCE.ALPHA, MOD.SEL.CV
  CV = do.a.lot.of.things.glmnet(X=Xtr, y=ytr, DATA_STR, TARGET, PREDICTORS_STR, LOG_FILE, FORCE.ALPHA, MOD.SEL.CV,
                                  bootstrap.nb=BOOTSTRAP.NB, permutation.nb=PERMUTATION.NB)
  #Xtr=Xtr; ytr=ytr; Xte=Xte; yte=yte; TARGET=TARGET; log_file=LOG_FILE; permutation.nb=PERMUTATION.NB
  TEST = generalize.on.test.dataset(Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, TARGET=TARGET, log_file=LOG_FILE,
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
  PREFIX = paste(OUTPUT, "/", DATA_STR, "_", TARGET, "~", PREDICTORS_STR, sep="")
  if (!file.exists(PREFIX)) dir.create(PREFIX)
  setwd(PREFIX)
  D_LEARN = DBLEARN[!is.na(DBLEARN[, TARGET]), c(TARGET, PREDICTORS)]
  D_TEST = DBTEST[!is.na(DBTEST[, TARGET]), c(TARGET, PREDICTORS)]
  ## FR CV
  y = D_LEARN[,TARGET]
  cv = cross_val(length(y),type="k-fold", k=10, random=TRUE, seed=97)
  dump("cv",file="cv.schema.R")
  
  Xtr = as.matrix(D_LEARN[, PREDICTORS])
  ytr = D_LEARN[, TARGET]
  Xte = as.matrix(D_TEST[, PREDICTORS])
  yte = D_TEST[, TARGET]
  CV = do.a.lot.of.things.glmnet(X=Xtr, y=ytr, DATA_STR, TARGET, PREDICTORS_STR, LOG_FILE, FORCE.ALPHA, MOD.SEL.CV,
                                 bootstrap.nb=BOOTSTRAP.NB, permutation.nb=PERMUTATION.NB)
  TEST = generalize.on.test.dataset(Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, TARGET=TARGET, log_file=LOG_FILE,
                                    permutation.nb=PERMUTATION.NB)
  if(is.null(RESULTS)) RESULTS = data.frame(c(CV, TEST)) else RESULTS = rbind(RESULTS, data.frame(c(CV, TEST)))
}

################################################################################################
## M36~CLINIC+NIGLOB
################################################################################################
PREDICTORS_STR = "CLINIC+NIGLOB"

for(TARGET in db$col_targets){
  #TARGET = "TMTB_TIME.M36"
  PREDICTORS = c(db$col_clinic, db$col_niglob)
  PREFIX = paste(OUTPUT, "/", DATA_STR, "_", TARGET, "~", PREDICTORS_STR, sep="")
  if (!file.exists(PREFIX)) dir.create(PREFIX)
  setwd(PREFIX)
  D_LEARN = DBLEARN[!is.na(DBLEARN[, TARGET]), c(TARGET, PREDICTORS)]
  D_TEST = DBTEST[!is.na(DBTEST[, TARGET]), c(TARGET, PREDICTORS)]
  ## FR CV
  y = D_LEARN[,TARGET]
  cv = cross_val(length(y),type="k-fold", k=10, random=TRUE, seed=97)
  dump("cv",file="cv.schema.R")
  
  Xtr = as.matrix(D_LEARN[, PREDICTORS])
  ytr = D_LEARN[, TARGET]
  Xte = as.matrix(D_TEST[, PREDICTORS])
  yte = D_TEST[, TARGET]
  CV = do.a.lot.of.things.glmnet(X=Xtr, y=ytr, DATA_STR, TARGET, PREDICTORS_STR, LOG_FILE, FORCE.ALPHA, MOD.SEL.CV,
                                 bootstrap.nb=BOOTSTRAP.NB, permutation.nb=PERMUTATION.NB)
  TEST = generalize.on.test.dataset(Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, TARGET=TARGET, log_file=LOG_FILE,
                                    permutation.nb=PERMUTATION.NB)
  if(is.null(RESULTS)) RESULTS = data.frame(c(CV, TEST)) else RESULTS = rbind(RESULTS, data.frame(c(CV, TEST)))
}

write.csv(RESULTS, OUTPUT_SUMMARY, row.names=FALSE)

# rsync -azvun --delete /neurospin/mescog/2014_mescog_predict_cog_decline ~/data/
