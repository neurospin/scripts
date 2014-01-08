install.packages("glmnet")


WD  = paste(Sys.getenv("HOME"),"data/2014_mescog_predict_cog_decline",sep="/")
SRC = paste(Sys.getenv("HOME"),"git/scripts/2013_mescog/proj_predict_cog_decline",sep="/")

## Load dataset
## ============
D = read.csv(DATASET_PATH)
colnames(D)
dim(D) # 378 1287

source(paste(SRC,"utils.R",sep="/"))


setwd(WD)
source(paste(SRC,"utils.R",sep="/"))
source(paste(SRC,"variables.R",sep="/"))

require(glmnet)
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
EXPERIMENT = "FR-M36~M0"

## M36_FR~M0
## ==============
VAR.NAME = ""
RM.FROM.PREDICTORS=c(scores.m36,"id")

for(i in 1:length(scores.m36)){
    setwd(WD)
    D.FR = read.table("2011-02_data_splitFR-D/m36FR~m0_flatten-sulci_fillmissing.csv")#173 490
    D.D  = read.table("2011-02_data_splitFR-D/m36D~m0_flatten-sulci_fillmissing.csv")#101 490
    cols = c("id", scores.m36, scores.m0)
    all(colnames(D.D)==colnames(D.FR))
    #i=1
    RESP.NAME = scores.m36[i]
    PRED.NAME = scores.m0[i]
    PREFIX = paste("2011-02_results_splitFR-D/",EXPERIMENT,"__",RESP.NAME,"~",PRED.NAME,sep="")
    #PREFIX = paste("2011-02_results_splitFR-D/",DATA.NAME,"_",RESP.NAME,sep="")
    paths = set.paths(WD, PREFIX, RESP.NAME)
    cols = c(RESP.NAME, PRED.NAME)
    D.FR = D.FR[,cols]
    D.D = D.D[,cols]
    D.FR = D.FR[!is.na(D.FR[,RESP.NAME]),]
    D.D = D.D[!is.na(D.D[,RESP.NAME]),]
    ## FR CV
    y = D.FR[,RESP.NAME]
    cv = cross_val(length(y),type="k-fold", k=10, random=TRUE, seed=97)
    dump("cv",file="cv.schema.R")
    
    y.pred.cv = c();  y.true.cv = c()
    for(test in cv){
        D.FR.train = D.FR[-test,]
        D.FR.test = D.FR[test,]
        formula = formula(paste(RESP.NAME,"~",PRED.NAME))
        modlm = lm(formula, data = D.FR.train)
        y.pred.cv = c(y.pred.cv, predict(modlm, D.FR.test))
        y.true.cv = c(y.true.cv, D.FR.test[,RESP.NAME])
    }
    loss.cv = round(loss.reg(y.true.cv, y.pred.cv),digit=2)
    log_file = paths$log_file
    ## CV predictions
    ## ==============

    cat("\nCV predictions\n",file=log_file,append=TRUE)
    cat("--------------\n",file=log_file,append=TRUE)

    sink(log_file, append = TRUE)
    print(loss.cv)
    sink()

    ## Fit and predict all (same train) data, (look for overfit)
    ## =========================================================

    cat("\nFit and predict all (same train) data, (look for overfit)\n",file=log_file,append=TRUE)
    cat("---------------------------------------------------------\n",file=log_file,append=TRUE)
    formula = formula(paste(RESP.NAME,"~",PRED.NAME))
    modlm.fr = lm(formula, data = D.FR)
    y.pred.all = predict(modlm.fr, D.FR)
    loss.all = round(loss.reg(y, y.pred.all),digit=2)
    sink(log_file, append = TRUE)
    print(loss.all)
    sink()
    
    
    ## Plot true vs predicted
    ## ======================
    
    require(ggplot2)
    pdf("glm.true-vs-pred.pdf")
    p = qplot(y.true.cv, y.pred.cv, geom = c("smooth","point"), method="lm",
        main=paste(RESP.NAME," - true vs. pred - CV - [R2_10CV=",loss.cv[["r2"]],"]",sep=""))
    print(p)

    # - true vs. pred (no CV)
    
    #svg("all.bestcv.glmnet.true-vs-pred.svg")
    p = qplot(y, y.pred.all, geom = c("smooth","point"), method="lm",
        main=paste(RESP.NAME," - true vs. pred - no CV - [R2=", loss.all[["r2"]],"]",sep=""))
    print(p)
    dev.off()

    cat("\nGeneralize on German dataset:\n",file=log_file,append=TRUE)
    cat("-------------------------------\n",file=log_file,append=TRUE)
    
    
    y.pred.fr = predict(modlm.fr, D.FR)
    y.pred.d  = predict(modlm.fr, D.D)

    loss.fr = round(loss.reg(D.FR[,RESP.NAME], y.pred.fr, df2=2), digit=2)
    loss.d  = round(loss.reg(D.D[,RESP.NAME],  y.pred.d,  df2=2), digit=2)
    
    pdf("true-vs-pred.D.pdf")
    y.pred.d = as.vector(y.pred.d)
    p = qplot(D.D[,RESP.NAME], y.pred.d, geom = c("smooth","point"), method="lm",
        main=paste(RESP.NAME," - true vs. pred - D - [R2=", loss.d[["R2"]] ,"]",sep=""))
    print(p)
    dev.off()    
    
    sink(log_file, append = TRUE)
    print(rbind(c(center=1,loss.fr),c(center=2,loss.d)))
    sink()
}

## FR-M36~M0__RANKIN_3~clin
## ========================
setwd(WD)
PRED.NAME = "clin"
cols = c("id", scores.m36, scores.m0, clinic.cte, clinic.m0)

D.FR = read.table("2011-02_data_splitFR-D/m36FR~m0_flatten-sulci_fillmissing.csv")#173 490
D.D  = read.table("2011-02_data_splitFR-D/m36D~m0_flatten-sulci_fillmissing.csv")#101 490
all(colnames(D.D)==colnames(D.FR))


D.FR = D.FR[,cols]
D.D = D.D[,cols]

RM.FROM.PREDICTORS=c(scores.m36,"id")

for(i in 1:length(scores.m36)){
    RESP.NAME = scores.m36[i]
    #TRIVIAL.PRED.NAME = scores.m0[i]
    PREFIX = paste("2011-02_results_splitFR-D/",EXPERIMENT,"__",RESP.NAME,"~",PRED.NAME,sep="")
    Xy.FR    = get.X.y(D.FR, RESP.NAME, RM.FROM.PREDICTORS)
    Xy.D     = get.X.y(D.D,  RESP.NAME, RM.FROM.PREDICTORS)
    paths = set.paths(WD, PREFIX, RESP.NAME)
    # X=Xy.FR$X; y=Xy.FR$y
    do.a.lot.of.things.glmnet(X=Xy.FR$X, y=Xy.FR$y, paths$log_file, FORCE.ALPHA, MOD.SEL.CV)
    # X.fr=Xy.FR$X; y.fr=Xy.FR$y; X.d=Xy.D$X; y.d=Xy.D$y; log_file=paths$log_file
    generalize.on.german.dataset(Xy.FR$X, Xy.FR$y, Xy.D$X, Xy.D$y, paths$log_file)
}

## M36_FR~M0_clin+imglob
## =====================
setwd(WD)
PRED.NAME = "clin+imglob"
cols = c("id", scores.m36, scores.m0, clinic.cte, clinic.m0, image.glob)

D.FR = read.table("2011-02_data_splitFR-D/m36FR~m0_flatten-sulci_fillmissing.csv")#173 490
D.D  = read.table("2011-02_data_splitFR-D/m36D~m0_flatten-sulci_fillmissing.csv")#101 490
all(colnames(D.D)==colnames(D.FR))


D.FR = D.FR[,cols]
D.D = D.D[,cols]

#RESP.NAME="RANKIN_3" # 
#RESP.NAME="TMTBT_3" #
#RESP.NAME="SCORETOT_3" #
RM.FROM.PREDICTORS=c(scores.m36,"id")

for(i in 1:length(scores.m36)){
    RESP.NAME = scores.m36[i]
    PREFIX = paste("2011-02_results_splitFR-D/",EXPERIMENT,"__",RESP.NAME,"~",PRED.NAME,sep="")

    Xy.FR    = get.X.y(D.FR, RESP.NAME, RM.FROM.PREDICTORS)
    Xy.D     = get.X.y(D.D,  RESP.NAME, RM.FROM.PREDICTORS)
    paths = set.paths(WD, PREFIX, RESP.NAME)
    #X=Xy.FR$X; y=Xy.FR$y; log_file=paths$log_file; FORCE.ALPHA; MOD.SEL.CV
    do.a.lot.of.things.glmnet(Xy.FR$X, Xy.FR$y, paths$log_file, FORCE.ALPHA, MOD.SEL.CV)
    generalize.on.german.dataset(Xy.FR$X, Xy.FR$y, Xy.D$X, Xy.D$y, paths$log_file)
}

## M36_FR~M0_imglob
## =====================
setwd(WD)
PRED.NAME = "imglob"
cols = c("id", scores.m36, image.glob)

D.FR = read.table("2011-02_data_splitFR-D/m36FR~m0_flatten-sulci_fillmissing.csv")#173 490
D.D  = read.table("2011-02_data_splitFR-D/m36D~m0_flatten-sulci_fillmissing.csv")#101 490
all(colnames(D.D)==colnames(D.FR))


D.FR = D.FR[,cols]
D.D = D.D[,cols]

#RESP.NAME="RANKIN_3" # 
#RESP.NAME="TMTBT_3" #
#RESP.NAME="SCORETOT_3" #
RM.FROM.PREDICTORS=c(scores.m36,"id")

for(i in 1:length(scores.m36)){
    RESP.NAME = scores.m36[i]
    PREFIX = paste("2011-02_results_splitFR-D/",EXPERIMENT,"__",RESP.NAME,"~",PRED.NAME,sep="")

    Xy.FR    = get.X.y(D.FR, RESP.NAME, RM.FROM.PREDICTORS)
    Xy.D     = get.X.y(D.D,  RESP.NAME, RM.FROM.PREDICTORS)
    paths = set.paths(WD, PREFIX, RESP.NAME)
    #X=Xy.FR$X; y=Xy.FR$y; log_file=paths$log_file; FORCE.ALPHA; MOD.SEL.CV
    do.a.lot.of.things.glmnet(Xy.FR$X, Xy.FR$y, paths$log_file, FORCE.ALPHA, MOD.SEL.CV)
    generalize.on.german.dataset(Xy.FR$X, Xy.FR$y, Xy.D$X, Xy.D$y, paths$log_file)
}

## M36_FR~M0_demo+imglob
## =====================
setwd(WD)
PRED.NAME = "demo+imglob"
cols = c("id", scores.m36, demographic, image.glob)

D.FR = read.table("2011-02_data_splitFR-D/m36FR~m0_flatten-sulci_fillmissing.csv")#173 490
D.D  = read.table("2011-02_data_splitFR-D/m36D~m0_flatten-sulci_fillmissing.csv")#101 490
all(colnames(D.D)==colnames(D.FR))


D.FR = D.FR[,cols]
D.D = D.D[,cols]

#RESP.NAME="RANKIN_3" # 
#RESP.NAME="TMTBT_3" #
#RESP.NAME="SCORETOT_3" #
RM.FROM.PREDICTORS=c(scores.m36,"id")

for(i in 1:length(scores.m36)){
    RESP.NAME = scores.m36[i]
    PREFIX = paste("2011-02_results_splitFR-D/",EXPERIMENT,"__",RESP.NAME,"~",PRED.NAME,sep="")

    Xy.FR    = get.X.y(D.FR, RESP.NAME, RM.FROM.PREDICTORS)
    Xy.D     = get.X.y(D.D,  RESP.NAME, RM.FROM.PREDICTORS)
    paths = set.paths(WD, PREFIX, RESP.NAME)
    #X=Xy.FR$X; y=Xy.FR$y; log_file=paths$log_file; FORCE.ALPHA; MOD.SEL.CV
    do.a.lot.of.things.glmnet(Xy.FR$X, Xy.FR$y, paths$log_file, FORCE.ALPHA, MOD.SEL.CV)
    generalize.on.german.dataset(Xy.FR$X, Xy.FR$y, Xy.D$X, Xy.D$y, paths$log_file)
}



## FR-M36~M0__RANKIN_3~clin+imglob+sulci
## =====================================
setwd(WD)
PRED.NAME = "clin+imglob+sulci"

D.FR = read.table("2011-02_data_splitFR-D/m36FR~m0_flatten-sulci_fillmissing.csv")#173 490
D.D  = read.table("2011-02_data_splitFR-D/m36D~m0_flatten-sulci_fillmissing.csv")#101 490
all(colnames(D.D)==colnames(D.FR))
 
#RESP.NAME="RANKIN_3" # 99 490
#RESP.NAME="TMTBT_3" #75 490
#RESP.NAME="SCORETOT_3" #84 490
RM.FROM.PREDICTORS=c(scores.m36,"id")

for(i in 1:length(scores.m36)){
    RESP.NAME = scores.m36[i]
    PREFIX = paste("2011-02_results_splitFR-D/",EXPERIMENT,"__",RESP.NAME,"~",PRED.NAME,sep="")

    Xy.FR    = get.X.y(D.FR, RESP.NAME, RM.FROM.PREDICTORS)
    Xy.D     = get.X.y(D.D,  RESP.NAME, RM.FROM.PREDICTORS)
    paths = set.paths(WD, PREFIX, RESP.NAME)

    do.a.lot.of.things.glmnet(Xy.FR$X, Xy.FR$y, paths$log_file, FORCE.ALPHA, MOD.SEL.CV)
    generalize.on.german.dataset(Xy.FR$X, Xy.FR$y, Xy.D$X, Xy.D$y, paths$log_file)
}



