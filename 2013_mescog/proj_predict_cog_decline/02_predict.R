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

################################################################################################
## M36~M0
################################################################################################

EXPERIMENT = "SIMPLE"

for(TARGET in db$col_targets){
    #TARGET = "TMTB_TIME.M36"
    PREDICTORS = strsplit(TARGET, "[.]")[[1]][1]
    PREFIX = paste(OUTPUT, "/", TARGET, "~", EXPERIMENT, sep="")
    if (!file.exists(PREFIX)) dir.create(PREFIX)
    setwd(PREFIX)
    D_FR = db$DB_FR[!is.na(db$DB_FR[, TARGET]), c(TARGET, PREDICTORS)]
    D_GR = db$DB_GR[!is.na(db$DB_GR[, TARGET]), c(TARGET, PREDICTORS)]
    ## FR CV
    y = D_FR[,TARGET]
    cv = cross_val(length(y),type="k-fold", k=10, random=TRUE, seed=97)
    dump("cv",file="cv.schema.R")
    
    y.pred.cv = c();  y.true.cv = c()
    for(test in cv){
        D_FR_train = D_FR[-test,]
        D_FR_test = D_FR[test,]
        formula = formula(paste(TARGET,"~",PREDICTORS))
        modlm = lm(formula, data = D_FR_train)
        y.pred.cv = c(y.pred.cv, predict(modlm, D_FR_test))
        y.true.cv = c(y.true.cv, D_FR_test[,TARGET])
    }
    loss.cv = round(loss.reg(y.true.cv, y.pred.cv),digit=2)
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
    modlm.fr = lm(formula, data = D_FR)
    y.pred.all = predict(modlm.fr, D_FR)
    loss.all = round(loss.reg(y, y.pred.all),digit=2)
    sink(LOG_FILE, append = TRUE)
    print(loss.all)
    sink()
    
    ## Plot true vs predicted
    ## ======================
    
    pdf("glm_FR_cv_true-vs-pred.pdf")
    p = qplot(y.true.cv, y.pred.cv, geom = c("smooth","point"), method="lm",
        main=paste(TARGET," - true vs. pred - CV - [R2_10CV=",loss.cv[["r2"]],"]",sep=""))
    print(p); dev.off()
  
    # - true vs. pred (no CV)
    
    #svg("all.bestcv.glmnet.true-vs-pred.svg")
    pdf("glm_FR_nocv_true-vs-pred.pdf")
    p = qplot(y, y.pred.all, geom = c("smooth","point"), method="lm",
        main=paste(TARGET," - true vs. pred - no CV - [R2=", loss.all[["r2"]],"]",sep=""))
    print(p); dev.off()

    cat("\nGeneralize on German dataset:\n",file=LOG_FILE,append=TRUE)
    cat("-------------------------------\n",file=LOG_FILE,append=TRUE)
    y.preD_FR = predict(modlm.fr, D_FR)
    y.preD_GR  = predict(modlm.fr, D_GR)

    loss.fr = round(loss.reg(D_FR[,TARGET], y.preD_FR, df2=2), digit=2)
    loss.d  = round(loss.reg(D_GR[,TARGET],  y.preD_GR,  df2=2), digit=2)
    
    pdf("glm_GR_true-vs-pred_GR.pdf")
    y.preD_GR = as.vector(y.preD_GR)
    p = qplot(D_GR[,TARGET], y.preD_GR, geom = c("smooth","point"), method="lm",
        main=paste(TARGET," - true vs. pred - D - [R2=", loss.d[["R2"]] ,"]",sep=""))
    print(p);dev.off()    
    
    sink(LOG_FILE, append = TRUE)
    print(rbind(c(center=1,loss.fr),c(center=2,loss.d)))
    sink()
}

################################################################################################
## M36~clin
################################################################################################
EXPERIMENT = "CLINIC"

for(TARGET in db$col_targets){
  #TARGET = "TMTB_TIME.M36"
  PREDICTORS = db$col_clinic
  PREFIX = paste(OUTPUT, "/", TARGET, "~", EXPERIMENT, sep="")
  if (!file.exists(PREFIX)) dir.create(PREFIX)
  setwd(PREFIX)
  D_FR = db$DB_FR[!is.na(db$DB_FR[, TARGET]), c(TARGET, PREDICTORS)]
  D_GR = db$DB_GR[!is.na(db$DB_GR[, TARGET]), c(TARGET, PREDICTORS)]
  ## FR CV
  y = D_FR[,TARGET]
  cv = cross_val(length(y),type="k-fold", k=10, random=TRUE, seed=97)
  dump("cv",file="cv.schema.R")

  Xfr = as.matrix(D_FR[, db$col_clinic])
  yfr = D_FR[, TARGET]
  Xgr = as.matrix(D_GR[, db$col_clinic])
  ygr = D_FR[, TARGET]
  do.a.lot.of.things.glmnet(X=Xfr, y=yfr, LOG_FILE, FORCE.ALPHA, MOD.SEL.CV)
  generalize.on.german.dataset(Xy.FR$X, Xy.FR$y, Xy.D$X, Xy.D$y, LOG_FILE)
}

## M36_FR~M0_clin+imglob
## =====================
setwd(WD)
PRED.NAME = "clin+imglob"
cols = c("id", scores.m36, scores.m0, clinic.cte, clinic.m0, image.glob)

D_FR = read.table("2011-02_data_splitFR-D/m36FR~m0_flatten-sulci_fillmissing.csv")#173 490
D_GR  = read.table("2011-02_data_splitFR-D/m36D~m0_flatten-sulci_fillmissing.csv")#101 490
all(colnames(D_GR)==colnames(D_FR))


D_FR = D_FR[,cols]
D_GR = D_GR[,cols]

#TARGET="RANKIN_3" # 
#TARGET="TMTBT_3" #
#TARGET="SCORETOT_3" #
RM.FROM.PREDICTORS=c(scores.m36,"id")

for(i in 1:length(scores.m36)){
    TARGET = scores.m36[i]
    PREFIX = paste("2011-02_results_splitFR-D/",EXPERIMENT,"__",TARGET,"~",PRED.NAME,sep="")

    Xy.FR    = get.X.y(D_FR, TARGET, RM.FROM.PREDICTORS)
    Xy.D     = get.X.y(D_GR,  TARGET, RM.FROM.PREDICTORS)
    paths = set.paths(WD, PREFIX, TARGET)
    #X=Xy.FR$X; y=Xy.FR$y; LOG_FILE=LOG_FILE; FORCE.ALPHA; MOD.SEL.CV
    do.a.lot.of.things.glmnet(Xy.FR$X, Xy.FR$y, LOG_FILE, FORCE.ALPHA, MOD.SEL.CV)
    generalize.on.german.dataset(Xy.FR$X, Xy.FR$y, Xy.D$X, Xy.D$y, LOG_FILE)
}

## M36_FR~M0_imglob
## =====================
setwd(WD)
PRED.NAME = "imglob"
cols = c("id", scores.m36, image.glob)

D_FR = read.table("2011-02_data_splitFR-D/m36FR~m0_flatten-sulci_fillmissing.csv")#173 490
D_GR  = read.table("2011-02_data_splitFR-D/m36D~m0_flatten-sulci_fillmissing.csv")#101 490
all(colnames(D_GR)==colnames(D_FR))


D_FR = D_FR[,cols]
D_GR = D_GR[,cols]

#TARGET="RANKIN_3" # 
#TARGET="TMTBT_3" #
#TARGET="SCORETOT_3" #
RM.FROM.PREDICTORS=c(scores.m36,"id")

for(i in 1:length(scores.m36)){
    TARGET = scores.m36[i]
    PREFIX = paste("2011-02_results_splitFR-D/",EXPERIMENT,"__",TARGET,"~",PRED.NAME,sep="")

    Xy.FR    = get.X.y(D_FR, TARGET, RM.FROM.PREDICTORS)
    Xy.D     = get.X.y(D_GR,  TARGET, RM.FROM.PREDICTORS)
    paths = set.paths(WD, PREFIX, TARGET)
    #X=Xy.FR$X; y=Xy.FR$y; LOG_FILE=LOG_FILE; FORCE.ALPHA; MOD.SEL.CV
    do.a.lot.of.things.glmnet(Xy.FR$X, Xy.FR$y, LOG_FILE, FORCE.ALPHA, MOD.SEL.CV)
    generalize.on.german.dataset(Xy.FR$X, Xy.FR$y, Xy.D$X, Xy.D$y, LOG_FILE)
}

## M36_FR~M0_demo+imglob
## =====================
setwd(WD)
PRED.NAME = "demo+imglob"
cols = c("id", scores.m36, demographic, image.glob)

D_FR = read.table("2011-02_data_splitFR-D/m36FR~m0_flatten-sulci_fillmissing.csv")#173 490
D_GR  = read.table("2011-02_data_splitFR-D/m36D~m0_flatten-sulci_fillmissing.csv")#101 490
all(colnames(D_GR)==colnames(D_FR))


D_FR = D_FR[,cols]
D_GR = D_GR[,cols]

#TARGET="RANKIN_3" # 
#TARGET="TMTBT_3" #
#TARGET="SCORETOT_3" #
RM.FROM.PREDICTORS=c(scores.m36,"id")

for(i in 1:length(scores.m36)){
    TARGET = scores.m36[i]
    PREFIX = paste("2011-02_results_splitFR-D/",EXPERIMENT,"__",TARGET,"~",PRED.NAME,sep="")

    Xy.FR    = get.X.y(D_FR, TARGET, RM.FROM.PREDICTORS)
    Xy.D     = get.X.y(D_GR,  TARGET, RM.FROM.PREDICTORS)
    paths = set.paths(WD, PREFIX, TARGET)
    #X=Xy.FR$X; y=Xy.FR$y; LOG_FILE=LOG_FILE; FORCE.ALPHA; MOD.SEL.CV
    do.a.lot.of.things.glmnet(Xy.FR$X, Xy.FR$y, LOG_FILE, FORCE.ALPHA, MOD.SEL.CV)
    generalize.on.german.dataset(Xy.FR$X, Xy.FR$y, Xy.D$X, Xy.D$y, LOG_FILE)
}



## FR-M36~M0__RANKIN_3~clin+imglob+sulci
## =====================================
setwd(WD)
PRED.NAME = "clin+imglob+sulci"

D_FR = read.table("2011-02_data_splitFR-D/m36FR~m0_flatten-sulci_fillmissing.csv")#173 490
D_GR  = read.table("2011-02_data_splitFR-D/m36D~m0_flatten-sulci_fillmissing.csv")#101 490
all(colnames(D_GR)==colnames(D_FR))
 
#TARGET="RANKIN_3" # 99 490
#TARGET="TMTBT_3" #75 490
#TARGET="SCORETOT_3" #84 490
RM.FROM.PREDICTORS=c(scores.m36,"id")

for(i in 1:length(scores.m36)){
    TARGET = scores.m36[i]
    PREFIX = paste("2011-02_results_splitFR-D/",EXPERIMENT,"__",TARGET,"~",PRED.NAME,sep="")

    Xy.FR    = get.X.y(D_FR, TARGET, RM.FROM.PREDICTORS)
    Xy.D     = get.X.y(D_GR,  TARGET, RM.FROM.PREDICTORS)
    paths = set.paths(WD, PREFIX, TARGET)

    do.a.lot.of.things.glmnet(Xy.FR$X, Xy.FR$y, LOG_FILE, FORCE.ALPHA, MOD.SEL.CV)
    generalize.on.german.dataset(Xy.FR$X, Xy.FR$y, Xy.D$X, Xy.D$y, LOG_FILE)
}



