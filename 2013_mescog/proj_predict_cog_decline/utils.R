################################################################################################
## READ INPUT
################################################################################################
read_db=function(infile){
  DB = read.csv(infile, header=TRUE, as.is=TRUE)
  col_info = c("ID", "SITE")
  col_targets = colnames(DB)[grep("36", colnames(DB))]
  col_predictors = colnames(DB)[!(colnames(DB) %in% c(col_info, col_targets))]
  col_niglob = col_predictors[grep("LLV|LLcount|WMHV|MBcount|BPF", col_predictors)]
  col_clinic = col_predictors[!(col_predictors %in% col_niglob)]
  
  if(!all(sort(colnames(DB)) == sort(c(col_info, col_targets, col_clinic, col_niglob)))){
    print("ERROR COLNAMES DO NOT MATCH")
    sys.on.exit()
  }
  return(list(DB_FR=DB[DB$SITE == "FR", ], DB_GR = DB[DB$SITE == "GR", ], col_info=col_info,
              col_targets=col_targets, col_clinic=col_clinic, col_niglob=col_niglob))
}
################################################################################################
## ML
################################################################################################

##' Sampling for cross-validation procedures
##' 
##' This function gives a list of traning and testing IDs to be used
##' for cross-validation.
##' 
##' @param x a vector of \code{row.names} or a number indicating the
##'        sample size
##' @param type leave-one-out or K-fold (default, loo)
##' @param K number of folds for K-fold (default K=10)
##' @param random should the IDs be randomized
##' @param  a seed to fix random generation
##' @return a \code{list} test indexes
##' @author ed, chl
##' @seealso \code{cv.glm {boot}}
##' @examples
##' cross_val(50,type="k-fold")
##' cross_val(26,type="k-fold",k=5)
##' cross_val(26,type="k-fold",k=5,random=T)
##' cross_val(100,"k",random=T,seed=101)
##' n <- 100
##' x1 <- rnorm(n,mean=5);x2 <- rnorm(n,mean=5);X=cbind(x1,x2)
##' rownames(X)=1:n
##' y <- x1+2*x2 + rnorm(n)
##' d=data.frame(X,y)
##' rownames(d)=1:n
##' 
##' y.hat.test <- c()
##' loo.cv <- cross_val(n)
##' for (test_idx in loo.cv) {
##'     d.train=d[-test_idx,]
##'     d.test =d[test_idx,]
##'     lm.fit=lm(y~x1+x2,data=d.train)
##'     y.hat.test=append(y.hat.test,predict(lm.fit, newdata =d.test))
##' }
##' loss.test.loo=y.hat.test-d$y
##' cat("CV LOO MSE",sqrt(1/n*sum((loss.test.loo)^2)),"\n")
##' 
##' y.hat.test <- c()
##' kfold.cv <- cross_val(n,type="k-fold")
##' for (test_idx in kfold.cv) {
##'     d.train=d[-test_idx,]
##'     d.test =d[test_idx,]
##'     lm.fit=lm(y~x1+x2,data=d.train)
##'     y.hat.test=append(y.hat.test,predict(lm.fit, newdata =d.test))
##' }
##' loss.test.kfold=y.hat.test-d$y
##' cat("CV K-FOLD MSE",sqrt(1/n*sum((loss.test.kfold)^2)),"\n")
##' 
##' lm.fit.all=lm(y~x1+x2,data=d)
##' y.hat.all=predict(lm.fit.all,newdata =d)
##' loss.all=y.hat.all-d$y
##' cat("ALL MSE",sqrt(1/n*sum((loss.all)^2)),"\n")
cross_val <- function(x, type=c("loo","k-fold"), k=10, random=FALSE,seed) {
  type <- match.arg(type)
  if (is.numeric(x) & (length(x) == 1)) { len <- x }
  else if (length(x) > 1) { len <- length(x)}
  else stop("Cannot determine sample size")
  idx <- seq(1,len)
  folds_test_idx=list()
  if (random && (type != "loo")) {
    if(!missing(seed)){
      set.seed(seed)
      attr(folds_test_idx,'seed')=seed
    }
    idx <- sample(len,rep=FALSE)
  }
  if (type == "k-fold"){
    test_ranges_seq=round(seq(0,len,length.out=k+1))
    # list of (randomized) index of test samples
    for(i in 2:length(test_ranges_seq))
      folds_test_idx[[i-1]]=idx[ (test_ranges_seq[i-1]+1):(test_ranges_seq[i]) ]
    attr(folds_test_idx,'type')=type
  }
  if (type == "loo") {
    for(i in 1:len)
      folds_test_idx[[i]]=i
    attr(folds_test_idx,'type')=type
  }
  return(folds_test_idx)
}

## ========================================================================== ##
## glmnet with lambda selection with internal CV + refil.lm
## ========================================================================== ##
#' glmnet Select lambda by CV
#' combine glmnet with lambda selected by cv.glmnet
#' some minor options: refit.lm ? min.within.1sd?
#' do the prediction :: THIS MUST BE CHANGED
#' respect the fit / predict api
glmnet.cvlambda<-function(X, y, newX, refit.lm=FALSE, min.within.1sd=FALSE, ...){
    require(glmnet)
    #cat("DB",dim(X),length(y), dim(newX) ,"\n")
    # model selection
    cv.glmnet = cv.glmnet(X, y, ...)
    # elasticnet fit
    if(min.within.1sd){lambda = cv.glmnet$lambda.1se}else{lambda = cv.glmnet$lambda.min}
    #cat("L",lambda)
    mod.glmnet = glmnet(X, y, lambda=lambda, ...)
    coefs=as.double(predict(mod.glmnet,type="coef")) # !! first item is the intercept
    # glm refit ?
    if(refit.lm){
        support=coefs[-1]!=0
        # Xte = as.data.frame(cbind(y=y,X[,support]))
        # X.test.d  = as.data.frame(cbind(y=y.test, X.test[,support]))                
        # mod.lm=lm(y ~., data=Xte)
        # preds.ols <- predict(mod.lm, X.test.d)
        # summary(mod.lm)
        X.1    = cbind(intercept=1,X[,support])
        newX.1 = cbind(intercept=1,newX[,support])
        betas.ols = solve((t(X.1)%*%X.1)) %*% t(X.1) %*% y
        preds = newX.1 %*% betas.ols
    }else{
        preds <- predict(mod.glmnet,newX)
    }
    return(list(preds=preds, coefs=coefs, mod.glmnet=mod.glmnet))        
}

## ========================================================================== ##
## Regression utils
## ========================================================================== ##
loss.reg<-function(y.true, y.pred, df2=NULL){
        ## R2: http://en.wikipedia.org/wiki/Coefficient_of_determination
        df1=length(y.true)
        SS.tot       = sum((y.true - mean(y.true))^2)
        #SS.tot.unbiased     = sum((y.true - mean(y.train))^2)
        SS.err       = sum((y.true - y.pred)^2)
        mse = SS.err/df1
        r2  = 1 - SS.err/SS.tot
        #r2.unbiased  = 1 - SS.err/SS.tot.unbiased
        correlation = cor(y.true, y.pred)
        if(is.null(df2))return(c(mse=mse, R2=r2, cor=correlation))
        fstat=((SS.tot - SS.err)/SS.err)* ((df1-df2)/(df2 - 1))
        return(c(mse=mse, R2=r2, cor=correlation, fstat=fstat))
}

#' y <- rnorm(50)
#' X <- matrix(rnorm(500), 50, 10)
#' X[,1:3] <- X[,1:3] + y
#' colnames(X) <- LETTERS[1:10]
#' cv = cross_val(length(y),type="k-fold", k=5, random=TRUE, seed=97)
#' a=cv.regression(X, y, glmnet.cvlambda, cv, min.within.1sd=FALSE, refit.lm=FALSE, alpha=.95)

cv.regression<-function(X, y, func, cv=NULL, ...){
    ntest=max(sapply(cv,length))
    coef.mat <- array(NA, c(ncol(X)+1, length(cv)), 
        dimnames=list(c("(Intercept)",colnames(X)),
                      paste("cv", 1:length(cv),sep="")))
    y.pred  = c()
    y.true  = c()
    y.train = c()
    for (fold.idx in 1:length(cv)) {#fold.idx=1
        #cat("\nCV",fold.idx)
        #folds[[fold.idx]]=list()
        omit=cv[[fold.idx]]
        res=do.call(func,c(list(X=X[-omit,,drop=FALSE], y=y[-omit], 
                                newX=X[omit,,drop=FALSE]),list(...)))
        #cat('DB 1\n')
        y.train = c(y.train, y[-omit])
        y.pred  = c(y.pred, res$preds)
        y.true  = c(y.true, y[omit])
        coefs=res$coefs
        coef.mat[, fold.idx] = coefs
    }
    loss=loss.reg(y.true=y.true, y.pred=y.pred, df2=mean(colSums(coef.mat!=0)))
    invisible(
    list(loss   = loss, 
         y      = list(y.true=y.true, y.pred=y.pred),
         mod    = list(coef.mat=coef.mat),
         params = list(cv=cv, ellipsis=list(...))))
}

cv.regression.data.frame<-function(df, target, predictors, intercept, cv=NULL){
#   df = db$DB_FR[!is.null(db$DB_FR[, target]), ]
#   target = "TMTB_TIME.M36"
#   predictors = db$col_clinic
#   intercept = FALSE
#   cv = cross_val(df[, target],type="k-fold", k=10, random=TRUE, seed=97)
# 
  #ntest=max(sapply(cv,length))
  if(intercept){
    formula = formula(paste(target,"~", paste(predictors, collapse='+')))
    coef.mat <- array(NA, c(length(predictors)+1, length(cv)), 
                    dimnames=list(c("(Intercept)", predictors),
                                  paste("cv", 1:length(cv),sep="")))
  }else{
    coef.mat <- array(NA, c(length(predictors), length(cv)), 
                      dimnames=list(predictors,
                                    paste("cv", 1:length(cv),sep="")))
    formula = formula(paste(target,"~", paste(predictors, collapse='+'), "-1"))
  }
  y.pred  = c()
  y.true  = c()
  y.train = c()

  for(fold.idx in 1:length(cv)){
    test =cv[[fold.idx]]
    #print(test)
    dftr = df[-test,]
    dfte = df[test,]
    modlm = lm(formula, data = dftr)
    #print(modlm)
    y.pred = c(y.pred, predict(modlm, dfte))
    y.true = c(y.true, dfte[, target])
    coef.mat[, fold.idx] = modlm$coefficients
  }
  loss.cv = round(loss.reg(y.true, y.pred, df2=2),digit=2)
  loss=loss.reg(y.true=y.true, y.pred=y.pred, df2=mean(colSums(coef.mat!=0, na.rm=TRUE)))
  invisible(
    list(loss   = loss, 
         y      = list(y.true=y.true, y.pred=y.pred),
         mod    = list(coef.mat=coef.mat),
         params = list(cv=cv)))
}
## ========================================================================== ##
## Paths/datasets utils
## ========================================================================== ##
# get.X.y<-function(D, RESP.NAME, RM.FROM.PREDICTORS){
#     D=D[!is.na(D[,RESP.NAME]),]
#     y=D[,RESP.NAME]
#     # X : remove reponse variables and id
#     X = as.matrix(D[,(!colnames(D) %in% RM.FROM.PREDICTORS)])
#     return(list(X=X, y=y))
# }

# set.paths<-function(WD, PREFIX, RESP.NAME){
#     output_dir=paste(WD,PREFIX,sep="/")
#     log_file=paste(RESP.NAME,".txt",sep="")
#     dir.create(output_dir)
#     setwd(output_dir)
#     return(list(output_dir=output_dir, log_file=log_file))
# }

do.a.lot.of.things.glmnet<-function(X, y, DATA_STR, TARGET, PREDICTORS_STR, FORCE.ALPHA, MOD.SEL.CV,
                                    bootstrap.nb=100, permutation.nb=100){
    log_file = paste("log_enet_",TARGET,"_", PREDICTORS_STR,".txt",sep="" )
    RESULT = list(data=DATA_STR, method="enet", target=TARGET, predictors=PREDICTORS_STR, dim=paste(dim(X), collapse="x"))
    #X=Xfr; y=yfr; log_file=paths$log_file;
    #cv = cross_val(length(y),type="k-fold", k=10, random=TRUE, seed=107)
    cv = cross_val(length(y),type="k-fold", k=10, random=TRUE, seed=97)
    dump("cv",file="cv.schema.R")

    cat(DATA_STR, TARGET, PREDICTORS_STR, "\n",file=log_file)
    cat(paste(rep("=", nchar(TARGET)), collapse=""),"\n",file=log_file,append=TRUE)
    cat("\ndim(X)=",dim(X),"\n",sep=" ",file=log_file,append=TRUE)
    
    ## Sensitivity study: choose alpha
    ## ===============================

    source("cv.schema.R")
    cat("\nSensitivity study: choose alpha\n",file=log_file,append=TRUE)
    cat("-------------------------------\n",file=log_file,append=TRUE)
    if(is.null(FORCE.ALPHA)){print("ERROR FORCE.ALPHA is null"); return(1)}
#     if(is.null(FORCE.ALPHA)){
#     # select the L2 mixing parameter
#     #cv.gs.glmnet=cv_glmnet_gridsearch(X, y, cv, nlambda=100, alphas=seq(0,1,.1))
#     #res=cv_glmnet_gridsearch.param.select(cv.gs.glmnet, plot.it=T, fig.output="cv.gridsearch.glmnet")
#     cv.gs.glmnet=cv.enet.gridsearch(X, y, cv, nlambda=100, alphas=seq(0,1,.1))
#     res=cv.enet.gridsearch.param.select(cv.gs.glmnet, plot.it=T, fig.output="cv.gridsearch.glmnet")
#     sink(log_file, append = TRUE)
#     print(as.data.frame(res))
#     sink()
#     }
    if(!is.null(FORCE.ALPHA))cat("force alpha = ", FORCE.ALPHA,"\n")
    if(!is.null(FORCE.ALPHA)){cat("alpha = ",FORCE.ALPHA,"\n",file="parameters.R")}else cat("alpha = ",res$min.alpha,"\n",file="parameters.R")
    
    ## Alternative for model selection (internal CV)
    ## =============================================

    cat("\nAlternative for model selection (internal CV)\n",file=log_file,append=TRUE)
    cat("---------------------------------------------\n",file=log_file,append=TRUE)
    source("cv.schema.R")
    source("parameters.R")
    # -
    cv.lambda.min     = cv.regression(X,y,glmnet.cvlambda,cv,min.within.1sd=FALSE,refit.lm=FALSE,alpha=alpha)
    cv.lambda.min.glm = cv.regression(X,y,glmnet.cvlambda,cv,min.within.1sd=FALSE,refit.lm=TRUE,alpha=alpha)
    cv.lambda.1sd     = cv.regression(X,y,glmnet.cvlambda,cv,min.within.1sd=TRUE, refit.lm=FALSE,alpha=alpha)
    cv.lambda.1sd.glm = cv.regression(X,y,glmnet.cvlambda,cv,min.within.1sd=TRUE, refit.lm=TRUE,alpha=alpha)

    options(width=200)
    res.summary=rbind(
    data.frame(method="cv.lambda.min",     as.list(cv.lambda.min$loss)),
    data.frame(method="cv.lambda.min.glm", as.list(cv.lambda.min.glm$loss)),
    data.frame(method="cv.lambda.1sd",     as.list(cv.lambda.1sd$loss)),
    data.frame(method="cv.lambda.1sd.glm", as.list(cv.lambda.1sd.glm$loss)))

    sink(log_file, append = TRUE)
    print(res.summary)
    sink()

    res=list(
    "cv.lambda.min"=cv.lambda.min,
    "cv.lambda.min.glm"=cv.lambda.min.glm,
    "cv.lambda.1sd"=cv.lambda.1sd,
    "cv.lambda.1sd.glm"=cv.lambda.1sd.glm)

    choosen.idx = switch(MOD.SEL.CV,
    manual.cv.lambda.min     =1,
    manual.cv.lambda.min.glm =2,
    manual.cv.lambda.1sd     =3,
    manual.cv.lambda.1sd.glm =4,
    auto.max.r2            =which.max(res.summary[,"R2"]),
    auto.max.cor           =which.max(res.summary[,"cor"]),
    auto.min.mse           =which.min(res.summary[,"mse"])
    )

    cv_enet = res[[choosen.idx]]
    save(cv_enet,file="cv_bestcv_glmnet.Rdata")

    cat("refit.lm       = ",cv_enet$params$ellipsis$refit.lm,"\n",file="parameters.R",append = TRUE)
    cat("min.within.1sd = ",cv_enet$params$ellipsis$min.within.1sd,"\n",file="parameters.R",append = TRUE)

    ## CV predictions
    ## ==============

    cat("\nCV predictions\n",file=log_file,append=TRUE)
    cat("--------------\n",file=log_file,append=TRUE)
    load("cv_bestcv_glmnet.Rdata")

    loss.cv = round(cv_enet$loss, digit=2)
    sink(log_file, append = TRUE)
    print(loss.cv)
    sink()
    loss.cv = loss.cv[c("R2", "cor", "fstat")]
    names(loss.cv) = c("r2_cv", "cor_cv", "fstat_cv")
    RESULT = c(RESULT, loss.cv)
    

    ## Fit and predict all (same train) data, (look for overfit)
    ## =========================================================

    cat("\nFit and predict all (same train) data, (look for overfit)\n",file=log_file,append=TRUE)
    cat("---------------------------------------------------------\n",file=log_file,append=TRUE)
    source("parameters.R")
    all_enet=glmnet.cvlambda(X, y, newX=X, refit.lm=refit.lm, min.within.1sd=min.within.1sd, alpha=alpha)
    save(all_enet,file="all_bestcv_glmnet.Rdata")

    load("all_bestcv_glmnet.Rdata")

    loss.all = round(loss.reg(y, all_enet$preds, 
        df2=sum(all_enet$coefs!=0)), digit=2)
    sink(log_file, append = TRUE)
    print(loss.all)
    sink()
    loss.all = loss.all[c("R2", "cor")]
    names(loss.all) = c("r2_all", "cor_all")
    RESULT = c(RESULT, loss.all)


    ## Plot true vs predicted
    ## ======================

    source("cv.schema.R")
    source("parameters.R")

    # - true vs. pred (CV)
    load(file="cv_bestcv_glmnet.Rdata")
    y.true = cv_enet$y$y.true
    y.pred = cv_enet$y$y.pred
    loss.cv  = round(cv_enet$loss, digit=2)
    
    require(ggplot2)
    pdf("cv_bestcv_glmnet_true-vs-pred.pdf")
    p = qplot(y.true, y.pred, geom = c("smooth","point"), method="lm",
        main=paste(TARGET," - true vs. pred - CV - [R2_10CV=",loss.cv[["R2"]],"]",sep=""))
    print(p)
    dev.off()

    # - true vs. pred (no CV)
    load("all_bestcv_glmnet.Rdata")
    y.true = y
    y.pred = as.vector(all_enet$preds)
    
    #svg("all_bestcv_glmnet_true-vs-pred.svg")
    pdf("all_bestcv_glmnet_true-vs-pred.pdf")
    p = qplot(y.true, y.pred, geom = c("smooth","point"), method="lm",
        main=paste(TARGET," - true vs. pred - no CV - [R2=", loss.all[["r2_all"]],"]",sep=""))
    print(p)
    dev.off()

    ## Coeficients
    ## ===========

    cat("\nCoeficients (+ sd bootstarping):\n",file=log_file,append=TRUE)
    cat("--------------------------------\n",file=log_file,append=TRUE)
    # - (a) do the permutations
    source("cv.schema.R")
    source("parameters.R")
    #load(file="all_bestcv_glmnet.Rdata")

    coefs.names=c("intercept",colnames(X))

    #if(file.exists("bootstrap_bestcv_glmnet.Rdata.tmp"))load("bootstrap_bestcv_glmnet.Rdata.tmp")
    #load("perms.cv_bestcv_glmnet.Rdata.tmp")
    if(!exists("bootstrap.list"))bootstrap.list=list()
    bootstrap.idx=(length(bootstrap.list)+1):bootstrap.nb

    #bootstrap=1:100
    for(boot.i in bootstrap.idx){
        cat("BOOTSTRAP",boot.i,"*****************\n")
        resamp.idx=sample.int(length(y), length(y), replace=T)

        #res=cv_glmnet_lambda(X,y.rnd,cv,min.within.1sd=FALSE,alpha=alpha)
        res = cv.lambda.min.glm = glmnet.cvlambda(
            X=X[resamp.idx,], y=y[resamp.idx], newX=X[resamp.idx,],
            min.within.1sd=min.within.1sd, refit.lm=refit.lm,
            alpha=alpha)
        bootstrap.list[[boot.i]]=res
        #save(bootstrap.list,file="bootstrap_bestcv_glmnet.Rdata.tmp")
    }
    save(bootstrap.list,file="bootstrap_bestcv_glmnet.Rdata")

    # - (b) build the coef table
    load(file="bootstrap_bestcv_glmnet.Rdata")
    load(file="all_bestcv_glmnet.Rdata")
    coefs.names=c("intercept",colnames(X))
    coefs.mat=array(NA, dim=c(length(all_enet$coefs),length(bootstrap.list)),
            dimnames=list(coefs.names,
                          paste("boot", 1:length(bootstrap.list),sep="")))
    for(boot.i in 1:length(bootstrap.list))coefs.mat[,boot.i]=bootstrap.list[[boot.i]]$coefs

    coefs.df=cbind(
        coef=all_enet$coefs,
        boot.mean=apply(coefs.mat,1,mean),
        boot.sd  =apply(coefs.mat,1,sd),
        boot.count=apply(coefs.mat,1,function(x)sum(x!=0)))

    coefs.df=as.data.frame(coefs.df)

    options(width=200)
    sink(log_file, append = TRUE)
    cat("\n#coefs!=0 (",sum(coefs.df$coef!=0),")\n",sep="")
    cat("~~~~~~~~~~~~~~\n")
    coefs.not.null = coefs.df[coefs.df$coef!=0,]
    coefs.not.null = coefs.not.null[order(abs(coefs.not.null$coef), decreasing = TRUE),]
    print(coefs.not.null)

    cat("\ncoefs!=0  in more than 50% of boostraped sample\n")
    cat("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
    coefs.more.50 = coefs.df[coefs.df$boot.count>(bootstrap.nb/2),]
    coefs.more.50 = coefs.more.50[order(coefs.more.50$boot.count, decreasing = TRUE),]
    print(coefs.more.50)
    sink()


    ## Permutations: calibrate E_10CV(R2)
    ## ==================================

    cat("\nPermutations: calibrate E_10CV:\n",file=log_file,append=TRUE)
    cat("-------------------------------\n",file=log_file,append=TRUE)
    source("cv.schema.R")
    source("parameters.R")
    # -

    if(file.exists("perms_bestcv_glmnet.Rdata"))load("perms_bestcv_glmnet.Rdata")
    #load("perms.cv_bestcv_glmnet.Rdata.tmp")
#     if(!exists("perms.lists")){
#         perms.cv.list=list(); perms.all.list=list()
#     }else{
#         perms.cv.list=perms.lists$perms.cv.list
#         perms.all.list=perms.lists$perms.all.list
#     }
#     permutations=(length(perms.cv.list)+1):100
    permutations = 1:permutation.nb
    perms.cv.list=list(); perms.all.list=list()
    for(perm.i in permutations){
        cat("\nPERM",perm.i,"*****************\n")
        y.rnd=sample(y)
        #loss CV
        loss.cv = cv.regression(X, y.rnd, glmnet.cvlambda, cv,
            min.within.1sd=min.within.1sd, refit.lm=refit.lm,
            alpha=alpha)$loss
        # loss all
        all_bestcv_glmnet_rnd=glmnet.cvlambda(X, y.rnd, newX=X, refit.lm=refit.lm, min.within.1sd=min.within.1sd, alpha=alpha)
        loss.all = loss.reg(y.rnd, all_bestcv_glmnet_rnd$preds, df2=sum(all_bestcv_glmnet_rnd$coefs!=0))

        #cat("\n");print(data.frame(lapply(res,function(x)mean(x,na.rm=TRUE))))
        perms.cv.list[[perm.i]]=loss.cv
        perms.all.list[[perm.i]]=loss.all
        #save(perms.list,file="perms.cv_bestcv_glmnet.Rdata.tmp")
    }
    perms.lists = list(perms.cv.list=perms.cv.list, perms.all.list=perms.all.list)
    save(perms.lists, file="perms_bestcv_glmnet.Rdata")

    cat("# Compute p-values\n", file=log_file, append=TRUE)
    cat("## ---------------\n", file=log_file, append=TRUE)

    load(file="perms_bestcv_glmnet.Rdata")
    perms.cv.list  = perms.lists$perms.cv.list
    perms.all.list = perms.lists$perms.all.list
    #load(file="perms.cv_bestcv_glmnet.Rdata.tmp")
    perms.loss.cv = perms.loss.all = NULL
    for(i in 1:length(perms.cv.list)){#i=1
        loss.cv = data.frame(as.list(perms.cv.list[[i]]))
        loss.all = data.frame(as.list(perms.all.list[[i]]))
        if(is.null(perms.loss.cv)){
            perms.loss.cv  = loss.cv
            perms.loss.all = loss.all
        } else{
             perms.loss.cv = rbind(perms.loss.cv, loss.cv)
             perms.loss.all = rbind(perms.loss.all, loss.all)
        }
    }

    load(file="cv_bestcv_glmnet.Rdata")
    true.loss.cv = cv_enet$loss
    load(file="all_bestcv_glmnet.Rdata")
    true.loss.all = loss.reg(y, all_enet$preds, df2=sum(all_enet$coefs!=0))

    perm.pval = rbind(
        data.frame(c(nperms=nrow(perms.loss.all), type="all", as.list(colSums(true.loss.all<perms.loss.all,na.rm=T)))),
        data.frame(c(nperms=nrow(perms.loss.cv),  type="cv",  as.list(colSums(true.loss.cv<perms.loss.cv,na.rm=T)))))

    sink(log_file, append = TRUE)
    cat("perms>true ?\n")
    print(perm.pval)
    sink()
    return(RESULT)
}
#Xtr=Xytr$X; ytr=Xytr$y; Xte=Xyte$X; yte=Xyte$y; log_file=paths$log_file
generalize.on.test.dataset<-function(Xtr, ytr, Xte, yte, TARGET, PREDICTORS_STR, permutation.nb){
    log_file = paste("log_enet_",TARGET,"_", PREDICTORS_STR,".txt",sep="" )
    cat("\nGeneralize on Test dataset:\n",file=log_file,append=TRUE)
    cat("-------------------------------\n",file=log_file,append=TRUE)
    
    
    load("all_bestcv_glmnet.Rdata")
    mod.fr = all_enet$mod.glmnet
    y.pred.fr = predict(mod.fr, Xtr)
    y.pred.d  = predict(mod.fr, Xte)

    loss.fr = round(loss.reg(ytr, y.pred.fr, df2=sum(all_enet$coefs!=0)), digit=2)
    loss.d  = round(loss.reg(yte,  y.pred.d,  df2=sum(all_enet$coefs!=0)), digit=2)
    
    pdf("all_bestcv_glmnet_true-vs-pred_test.pdf")
    y.pred.d = as.vector(y.pred.d)
    p = qplot(yte, y.pred.d, geom = c("smooth","point"), method="lm",
        main=paste(TARGET," - true vs. pred - TEST - [R2=", loss.d[["R2"]] ,"]",sep=""))
    print(p)
    dev.off()    
    
    sink(log_file, append = TRUE)
    print(rbind(c(center=1,loss.fr),c(center=2,loss.d)))
    sink()

#     if(file.exists("permsD.all_bestcv_glmnet.Rdata"))load("permsD.all_bestcv_glmnet.Rdata")
#     if(!exists("perms.list"))perms.list=list()
    perms.list=list()
    permutations=1:permutation.nb

    for(perm.i in permutations){
        cat("PERM",perm.i,"*****************\n")
        y.rnd=sample(yte)
        loss=loss.reg(y.true=y.rnd,   y.pred=y.pred.d, df2=sum(all_enet$coefs!=0))
        perms.list[[perm.i]]=loss
        #save(perms.list,file="permsD.all_bestcv_glmnet.Rdata")
    }
    save(perms.list,file="perms_all_bestcv_glmnet_test.Rdata")

    cat("# Compute p-values\n",file=log_file,append=TRUE)
    cat("## ~~~~~~~~~~~~~~~\n",file=log_file,append=TRUE)

    #load(file="perms_all_bestcv_glmnet_test.Rdata")
    #load(file="perms.cv_bestcv_glmnet.Rdata.tmp")
    perms=NULL
    for(i in 1:length(perms.list)){#i=1
        p=perms.list[[i]]
        if(is.null(perms))perms=as.data.frame(as.list(p)) else perms = rbind(perms, p)
    }

    sink(log_file, append = TRUE)
    cat("perms>true ?\n")
    print(c(nperms=nrow(perms),colSums(perms>loss.d,na.rm=T)))
    sink()
    loss.d = loss.d[c("R2", "cor", "fstat")]
    names(loss.d) = c("r2_test", "cor_test", "fstat_test")
    return(as.list(loss.d))
}


do.a.lot.of.things.glm<-function(DBtr, DBte, TARGET, INTERCEPT, PREDICTORS, DATA_STR, PREDICTORS_STR,
                                    bootstrap.nb=100, permutation.nb=100){
  #TARGET="TMTB_TIME.M36"; 
  #DBtr=db$DB_FR[!is.na(db$DB_FR[, TARGET]), ]
  #DBte=db$DB_GR[!is.na(db$DB_GR[, TARGET]), ]
  # INTERCEPT=FALSE; PREDICTORS=c(db$col_niglob, "TMTB_TIME");
  #DATA_STR="TOTO"; PREDICTORS_STR="TOTO"; log_file="toto.txt"
  #bootstrap.nb=100; permutation.nb=100
  #DB[, TARGET]
  log_file = paste("log_enet_",TARGET,"_", PREDICTORS_STR,".txt",sep="" )
  RESULT = list(data=DATA_STR, method="glm", target=TARGET, predictors=PREDICTORS_STR, dim=paste(nrow(DBtr), length(PREDICTORS), sep="x"))
  cv = cross_val(length(DBtr[, TARGET]),type="k-fold", k=10, random=TRUE, seed=97)
  dump("cv",file="cv.schema.R")
  
  cat(DATA_STR, TARGET, PREDICTORS_STR, "\n",file=log_file)
  cat(paste(rep("=", nchar(TARGET)), collapse=""),"\n",file=log_file,append=TRUE)
  cat("\ndim(X)=",nrow(DBtr), length(PREDICTORS),"\n",sep=" ",file=log_file,append=TRUE)
  
  #df=DBtr; target=TARGET; intercept=INTERCEPT; predictors=PREDICTORS
  cv_glm = cv.regression.data.frame(df=DBtr, target=TARGET, predictors=PREDICTORS, intercept=INTERCEPT, cv=cv)
  cv_glm$loss = round(cv_glm$loss, digit=2)
  save(cv_glm, file="cv_glm.Rdata")
  
  ## CV predictions
  ## ==============  
  cat("\nCV predictions\n",file=log_file,append=TRUE)
  cat("--------------\n",file=log_file,append=TRUE)
  sink(log_file, append = TRUE)
  print(cv_glm$loss)
  sink()
  loss.cv = cv_glm$loss[c("R2", "cor", "fstat")]
  names(loss.cv) = c("r2_cv", "cor_cv", "fstat_cv")
  RESULT = c(RESULT, loss.cv)
  
  ## Fit and predict all (same train) data, (look for overfit)
  ## =========================================================
  cat("\nFit and predict all (same train) data, (look for overfit)\n",file=log_file,append=TRUE)
  cat("---------------------------------------------------------\n",file=log_file,append=TRUE)
  if(INTERCEPT) formula = formula(paste(TARGET,"~", paste(PREDICTORS, collapse='+'))) else formula = formula(paste(TARGET,"~", paste(PREDICTORS, collapse='+'), "-1"))
  modlm = lm(formula, data = DBtr)
  all_glm = list(y=list(y.true=DBtr[, TARGET], y.pred=predict(modlm, DBtr)), coefs=modlm$coefficients, mod=modlm)
  all_glm$loss=round(loss.reg(all_glm$y$y.true, all_glm$y$y.pred, 
                              df2=sum(all_glm$coefs!=0)), digit=2)
  save(all_glm,file="all_glm.Rdata")
  sink(log_file, append = TRUE)
  print(all_glm$loss)
  sink()
  loss.all = all_glm$loss[c("R2", "cor")]
  names(loss.all) = c("r2_all", "cor_all")
  RESULT = c(RESULT, loss.all)
  
  ## Plot true vs predicted
  ## ======================
  require(ggplot2)
  pdf("cv_glm_true-vs-pred.pdf")
  p = qplot(cv_glm$y$y.true, cv_glm$y$y.pred, geom = c("smooth","point"), method="lm",
            main=paste(TARGET," - true vs. pred - CV - [R2_10CV=",round(cv_glm$loss['R2'],2),"]",sep=""))
  print(p)
  dev.off()
  
  # - true vs. pred (no CV)
  #svg("all_bestcv_glmnet_true-vs-pred.svg")
  pdf("all_bestcv_glmnet_true-vs-pred.pdf")
  p = qplot(all_glm$y$y.true, all_glm$y$y.pred, geom = c("smooth","point"), method="lm",
            main=paste(TARGET," - true vs. pred - no CV - [R2=", round(all_glm$loss['R2'],2),"]",sep=""))
  print(p)
  dev.off()
  
  ## Coeficients
  ## ===========
  
  cat("\nCoeficients (+ sd bootstarping):\n",file=log_file,append=TRUE)
  cat("--------------------------------\n",file=log_file,append=TRUE)
  # - (a) do the permutations
  bootstrap.list=list()
  bootstrap.idx=(length(bootstrap.list)+1):bootstrap.nb
  for(boot.i in bootstrap.idx){
    cat("BOOTSTRAP",boot.i,"*****************\n")
    resamp.idx=sample.int(nrow(DBtr), nrow(DBtr), replace=T)
    DBtr.boot = DBtr[resamp.idx,]
    modlm.boot = lm(formula, data=DBtr.boot)
    boot_glm = list(y=list(y.true=DBtr.boot[, TARGET], y.pred=predict(modlm, DBtr.boot)),
                    coefs=modlm.boot$coefficients, mod=modlm.boot)
    boot_glm$loss=round(loss.reg(boot_glm$y$y.true, boot_glm$y$y.pred, 
                                df2=sum(all_glm$coefs!=0)), digit=2)
    bootstrap.list[[boot.i]] = boot_glm
    #save(bootstrap.list,file="bootstrap_bestcv_glmnet.Rdata.tmp")
  }
  save(bootstrap.list,file="bootstrap_glm.Rdata")
  
  # - (b) build the coef table
  coefs.mat=array(NA, dim=c(length(all_glm$coefs),length(bootstrap.list)),
                  dimnames=list(names(all_glm$coefs),
                                paste("boot", 1:length(bootstrap.list),sep="")))
  for(boot.i in 1:length(bootstrap.list))coefs.mat[,boot.i]=bootstrap.list[[boot.i]]$coefs
  
  coefs.df=cbind(
    coef=all_glm$coefs,
    boot.mean=apply(coefs.mat,1,mean),
    boot.sd  =apply(coefs.mat,1,sd),
    boot.count=apply(coefs.mat,1,function(x)sum(x!=0)))
  
  coefs.df=as.data.frame(coefs.df)
  
  options(width=200)
  sink(log_file, append = TRUE)
  cat("\n#coefs!=0 (",sum(coefs.df$coef!=0),")\n",sep="")
  cat("~~~~~~~~~~~~~~\n")
  coefs.not.null = coefs.df[coefs.df$coef!=0,]
  coefs.not.null = coefs.not.null[order(abs(coefs.not.null$coef), decreasing = TRUE),]
  print(coefs.df)
  sink()

  ## Permutations: calibrate E_10CV(R2)
  ## ==================================
  
  cat("\nPermutations: calibrate E_10CV:\n",file=log_file,append=TRUE)
  cat("-------------------------------\n",file=log_file,append=TRUE)
  permutations = 1:permutation.nb
  perms.cv.list=list(); perms.all.list=list()
  for(perm.i in permutations){
    cat("\nPERM",perm.i,"*****************\n")
    DBtr.perm = DBtr
    DBtr.perm[, TARGET] = sample(DBtr.perm[, TARGET])
    # loss CV
    loss.cv.perm = cv.regression.data.frame(df=DBtr.perm, target=TARGET, predictors=PREDICTORS, 
                                           intercept=INTERCEPT, cv=cv)$loss
    # loss all
    modlm.perm = lm(formula, data = DBtr.perm)
    loss.all.perm = round(loss.reg(DBtr.perm[, TARGET], predict(modlm, DBtr.perm), 
                                df2=sum(all_glm$coefs!=0)), digit=2)
    
    perms.cv.list[[perm.i]]=loss.cv.perm
    perms.all.list[[perm.i]]=loss.all.perm
    #save(perms.list,file="perms.cv_bestcv_glmnet.Rdata.tmp")
  }
  perms.lists = list(perms.cv.list=perms.cv.list, perms.all.list=perms.all.list)
  save(perms.lists, file="perms_glm.Rdata")
  
  cat("# Compute p-values\n", file=log_file, append=TRUE)
  cat("## ---------------\n", file=log_file, append=TRUE)
  perms.loss.cv = perms.loss.all = NULL
  for(i in 1:length(perms.cv.list)){#i=1
    loss.cv = data.frame(as.list(perms.cv.list[[i]]))
    loss.all = data.frame(as.list(perms.all.list[[i]]))
    if(is.null(perms.loss.cv)){
      perms.loss.cv  = loss.cv
      perms.loss.all = loss.all
    } else{
      perms.loss.cv = rbind(perms.loss.cv, loss.cv)
      perms.loss.all = rbind(perms.loss.all, loss.all)
    }
  }
  
  perm.pval = rbind(
    data.frame(c(nperms=nrow(perms.loss.all), type="all", as.list(colSums(all_glm$loss<perms.loss.all,na.rm=T)))),
    data.frame(c(nperms=nrow(perms.loss.cv),  type="cv",  as.list(colSums(cv_glm$loss<perms.loss.cv,na.rm=T)))))
  
  sink(log_file, append = TRUE)
  cat("perms>true ?\n")
  print(perm.pval)
  sink()
  
  cat("\nGeneralize on Test dataset:\n",file=log_file,append=TRUE)
  cat("-------------------------------\n",file=log_file,append=TRUE)

  y.pred.tr = predict(all_glm$mod, DBtr)
  y.pred.te  = predict(all_glm$mod, DBte)
  
  loss.tr = round(loss.reg(DBtr[, TARGET], y.pred.tr, df2=sum(all_glm$coefs!=0)), digit=2)
  loss.te = round(loss.reg(DBte[, TARGET], y.pred.te,  df2=sum(all_glm$coefs!=0)), digit=2)
  
  pdf("all_glm_true-vs-pred_test.pdf")
  p = qplot(DBte[, TARGET], y.pred.te, geom = c("smooth","point"), method="lm",
            main=paste(TARGET," - true vs. pred - TEST - [R2=", loss.te[["R2"]] ,"]",sep=""))
  print(p)
  dev.off()    
  
  sink(log_file, append = TRUE)
  print(rbind(c(center=1,loss.tr),c(center=2,loss.te)))
  sink()

  perms.list=list()
  permutations=1:permutation.nb
  for(perm.i in permutations){
    cat("PERM",perm.i,"*****************\n")
    loss.te.perm = loss.reg(y.true=sample(DBte[, TARGET]),   y.pred=y.pred.te, df2=sum(all_glm$coefs!=0))
    perms.list[[perm.i]]=loss.te.perm
  }
  save(perms.list,file="perms_all_glm_test.Rdata")
  
  cat("# Compute p-values\n",file=log_file,append=TRUE)
  cat("## ~~~~~~~~~~~~~~~\n",file=log_file,append=TRUE)
  perms=NULL
  for(i in 1:length(perms.list)){#i=1
    p=perms.list[[i]]
    if(is.null(perms))perms=as.data.frame(as.list(p)) else perms = rbind(perms, p)
  }
  
  sink(log_file, append = TRUE)
  cat("perms>true ?\n")
  print(c(nperms=nrow(perms),colSums(perms>loss.te,na.rm=T)))
  sink()
  loss.te = loss.te[c("R2", "cor", "fstat")]
  names(loss.te) = c("r2_test", "cor_test", "fstat_test")
  RESULT = c(RESULT, as.list(loss.te))
  return(RESULT)
}

## ================
## MODEL COMPARISON
## ================
compare.models<-function(mod1_path, mod2_path, dbtr, dbte, target){
  
  dbtr = dbtr[!is.na(dbtr[, target]), ]
  dbte = dbte[!is.na(dbte[, target]), ]
  
  mod_path = mod1_path
  
  if(exists("all_glm")) rm(all_glm)
  if(exists("all_enet")) rm(all_enet)  
  load(mod_path)
  if(exists("all_glm")){
    rm(all_glm)
  }
  if(exists("all_enet")){
    modall_enet1$mod.glmnet
  }
  
  all_enet1 = all_enet
  if(exists("all_glm")) rm(all_glm)
  if(exists("all_enet")) rm(all_enet)
  load(mod2_path)
  all_enet2 = all_enet
  w1 = as.matrix(all_enet1$mod.glmnet$beta)
  w1.col = rownames(w1)
  w1.col.nonnull = rownames(w1)[w1!=0]
  w2 = as.matrix(all_enet2$mod.glmnet$beta)
  w2.col = rownames(w2)
  w2.col.nonnull = rownames(w2)[w2!=0]
  
  Xtr1 = as.matrix(dbtr[, rownames(w1)])
  Xte1 = as.matrix(dbte[, rownames(w1)])
  Xtr2 = as.matrix(dbtr[, rownames(w2)])
  Xte2 = as.matrix(dbte[, rownames(w2)])
  ytr1 = dbtr[, target]
  yte1 = dbte[, target]
  ytr2 = dbtr[, target]
  yte2 = dbte[, target]
  
  all((predict(all_enet1$mod.glmnet, Xtr1) - all_enet1$preds) == 0)
  all((predict(all_enet2$mod.glmnet, Xtr2) - all_enet2$preds) == 0)
  
  ypred1 = predict(all_enet1$mod.glmnet, Xte1)
  ypred2 = predict(all_enet2$mod.glmnet, Xte2)
  
  loss1 = round(loss.reg(yte1, ypred1, df2=sum(all_enet1$coefs!=0)), digit=2)
  loss2 = round(loss.reg(yte2, ypred2, df2=sum(all_enet2$coefs!=0)), digit=2)
  
  inter =intersect(w1.col.nonnull, w2.col.nonnull)
  in1.not.in2 = setdiff(w1.col.nonnull, w2.col.nonnull)
  in2.not.in1 = setdiff(w2.col.nonnull, w1.col.nonnull)
  super2.nonnull = c(w1.col.nonnull, in2.not.in1) # 2 such 1 is nested in 2
  
  df1 = length(w1.col.nonnull) + 1
  df2 = length(super2.nonnull) + 1
  # QC all(Xyte1$y == Xyte2$y)
  
  rsste1 = sum((yte1 - ypred1)^2)
  rsste2 = sum((yte2 - ypred2)^2)
  
  n = length(yte1)
  fstat = ((rsste1 - rsste2) / (df2 - df1)) / (rsste2 / (n - df1))
  
  pval=1-pf(fstat, df2 - df1, n - df2)
  return(data.frame(var=target, fstat=fstat, pval=pval, df1, df2, p1=length(w1.col.nonnull), p2=length(w2.col.nonnull), n.in1.not.in2=length(in1.not.in2), n.in2.not.in1=length(in2.not.in1), in2.not.in1=paste(in2.not.in1, collapse=","), R2.test.1=loss1["R2"], R2.test.2=loss2["R2"]))
}

# compare.models<-function(f1, f2, cols1, cols2, D.FR, D.D, TARGET){
# 
#     Xytr1    = get.X.y(D.FR[,cols1], TARGET, RM.FROM.PREDICTORS)
#     Xyte1     = get.X.y(D.D[,cols1],  TARGET, RM.FROM.PREDICTORS)
#     Xytr2    = get.X.y(D.FR[,cols2], TARGET, RM.FROM.PREDICTORS)
#     Xyte2     = get.X.y(D.D[,cols2],  TARGET, RM.FROM.PREDICTORS)
#     # mod1
#     load(f1)
#     all_enet1 = all_enet
#     rm(all_enet)
#     mod1.fr = all_enet1$mod.glmnet
#     y.pred1.fr = predict(mod1.fr, Xytr1$X)
#     y.pred1.d  = predict(mod1.fr, Xyte1$X)
#     loss1.fr = round(loss.reg(Xytr1$y, y.pred1.fr, df2=sum(all_enet1$coefs!=0)), digit=2)
#     loss1.d  = round(loss.reg(Xyte1$y,  y.pred1.d,  df2=sum(all_enet1$coefs!=0)), digit=2)
#     # mod2
#     load(f2)
#     all_enet2 = all_enet
#     rm(all_enet)
#     mod2.fr = all_enet2$mod.glmnet
#     y.pred2.fr = predict(mod2.fr, Xytr2$X)
#     y.pred2.d  = predict(mod2.fr, Xyte2$X)
#     loss2.fr = round(loss.reg(Xytr2$y, y.pred2.fr, df2=sum(all_enet2$coefs!=0)), digit=2)
#     loss2.d  = round(loss.reg(Xyte2$y,  y.pred2.d,  df2=sum(all_enet2$coefs!=0)), digit=2)
# 
#     # predictors
#     w1 = as.matrix(all_enet1$mod.glmnet$beta)
#     w1.col = rownames(w1)
#     w1.col.nonnull = rownames(w1)[w1!=0]
#     w2 = as.matrix(all_enet2$mod.glmnet$beta)
#     w2.col = rownames(w2)
#     w2.col.nonnull = rownames(w2)[w2!=0]
#     # QC intersect(w1.col, w2.col) == w1.col
#     inter =intersect(w1.col.nonnull, w2.col.nonnull)
#     in1.not.in2 = setdiff(w1.col.nonnull, w2.col.nonnull)
#     in2.not.in1 = setdiff(w2.col.nonnull, w1.col.nonnull)
#     super2.nonnull = c(w1.col.nonnull, in2.not.in1) # 2 such 1 is nested in 2
#     
#     df1 = length(w1.col.nonnull) + 1
#     df2 = length(super2.nonnull) + 1
#     # QC all(Xyte1$y == Xyte2$y)
# 
#     rss1.d = sum((y.pred1.d - Xyte1$y)^2)
#     rss2.d = sum((y.pred2.d - Xyte2$y)^2)
# 
#     n = length(y.pred1.d)
#     fstat = ((rss1.d - rss2.d) / (df2 - df1)) / (rss2.d / (n - df1))
# 
#     return(data.frame(var=TARGET, fstat=fstat, p.pval=1-pf(fstat, df2 - df1, n - df2), df1, df2, n.pred1=length(w1.col.nonnull), n.pred2=length(w2.col.nonnull), n.in1.not.in2=length(in1.not.in2), n.in2.not.in1=length(in2.not.in1), in2.not.in1=paste(in2.not.in1, collapse=","), R2.1.d=loss1.d["R2"], R2.2.d=loss2.d["R2"]))
# }

