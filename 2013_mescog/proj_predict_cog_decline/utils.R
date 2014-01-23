################################################################################################
## READ INPUT
################################################################################################
read_db=function(infile, to_remove){
  DB = read.csv(infile, header=TRUE, as.is=TRUE)
  col_info = c("ID", "SITE")
  col_targets =   c("TMTB_TIME.M36","MDRS_TOTAL.M36","MRS.M36","BARTHEL.M36","MMSE.M36")
  to_remove = unique(c(to_remove, col_info, col_targets, colnames(DB)[grep("36", colnames(DB))]))
  col_predictors = colnames(DB)[!(colnames(DB) %in% to_remove)]
  col_niglob = col_predictors[grep("LLV|LLcount|WMHV|MBcount|BPF", col_predictors)]
  col_clinic = col_predictors[!(col_predictors %in% col_niglob)]

  if(!all(sort(colnames(DB)) == sort(unique(c(col_info, col_targets, col_clinic, col_niglob, to_remove))))){
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
##' loss_test.loo=y.hat.test-d$y
##' cat("CV LOO MSE",sqrt(1/n*sum((loss_test.loo)^2)),"\n")
##' 
##' y.hat.test <- c()
##' kfold.cv <- cross_val(n,type="k-fold")
##' for (test_idx in kfold.cv) {
##'     d.train=d[-test_idx,]
##'     d.test =d[test_idx,]
##'     lm.fit=lm(y~x1+x2,data=d.train)
##'     y.hat.test=append(y.hat.test,predict(lm.fit, newdata =d.test))
##' }
##' loss_test.kfold=y.hat.test-d$y
##' cat("CV K-FOLD MSE",sqrt(1/n*sum((loss_test.kfold)^2)),"\n")
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
    cv_glmnet = cv.glmnet(X, y, ...)
    # elasticnet fit
    if(min.within.1sd){lambda = cv_glmnet$lambda.1se}else{lambda = cv_glmnet$lambda.min}
    #cat("L",lambda)
    mod.glmnet = glmnet(X, y, lambda=lambda, ...)
    coef = predict(mod.glmnet,type="coef")
    coef.names = rownames(coef)
    coef=as.double(coef) # !! first item is the intercept
    names(coef) = coef.names
    # glm refit ?
    if(refit.lm){
        support=coef[-1]!=0
        # Xte = as.data.frame(cbind(y=y,X[,support]))
        # X.test.d  = as.data.frame(cbind(y=y.test, X.test[,support]))                
        # mod.lm=lm(y ~., data=Xte)
        # preds.ols <- predict(mod.lm, X.test.d)
        # summary(mod.lm)
        X.1    = cbind(intercept=1,X[,support])
        newX.1 = cbind(intercept=1,newX[,support])
        betas.ols = solve((t(X.1)%*%X.1)) %*% t(X.1) %*% y
        y_pred = newX.1 %*% betas.ols
    }else{
        y_pred <- as.vector(predict(mod.glmnet,newX))
    }
    return(list(y_true=y, y_pred=y_pred, coef=coef, mod=mod.glmnet))        
}

## ========================================================================== ##
## Regression utils
## ========================================================================== ##
loss.reg<-function(y_true, y_pred, df2=NULL, suffix=NULL){
    ## r2: http://en.wikipedia.org/wiki/Coefficient_of_determination
    df1=length(y_true)
    SS.tot       = sum((y_true - mean(y_true))^2)
    #SS.tot.unbiased     = sum((y_true - mean(y.train))^2)
    SS.err       = sum((y_true - y_pred)^2)
    mse = SS.err/df1
    r2  = 1 - SS.err/SS.tot
    #r2.unbiased  = 1 - SS.err/SS.tot.unbiased
    correlation = cor(y_true, y_pred)
    loss = c(mse=mse, r2=r2, cor=correlation)
    if(!is.null(df2))
      loss = c(loss, fstat=((SS.tot - SS.err)/SS.err)* ((df1-df2)/(df2 - 1)))
    #loss = c(mse=mse, r2=r2, cor=correlation, fstat=fstat)
    if(!is.null(suffix))names(loss)=paste(names(loss), suffix, sep="_")
    return(loss)
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
    y_pred  = c()
    y_true  = c()
    y.train = c()
    for (fold.idx in 1:length(cv)) {#fold.idx=1
        #cat("\nCV",fold.idx)
        #folds[[fold.idx]]=list()
        omit=cv[[fold.idx]]
        res=do.call(func,c(list(X=X[-omit,,drop=FALSE], y=y[-omit], 
                                newX=X[omit,,drop=FALSE]),list(...)))
        #cat('DB 1\n')
        y.train = c(y.train, y[-omit])
        y_pred  = c(y_pred, res$y_pred)
        y_true  = c(y_true, y[omit])
        #coef=
        coef.mat[, fold.idx] = res$coef
    }
    loss=loss.reg(y_true=y_true, y_pred=y_pred, df2=mean(colSums(coef.mat!=0)))
    invisible(
    list(loss   = loss, 
         y_true = y_true,
         y_pred = y_pred,
         coef   = coef.mat,
         #mod    = list(coef.mat=coef.mat),
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
  y_pred  = c()
  y_true  = c()
  y.train = c()

  for(fold.idx in 1:length(cv)){
    test =cv[[fold.idx]]
    #print(test)
    dftr = df[-test,]
    dfte = df[test,]
    modlm = lm(formula, data = dftr)
    #print(modlm)
    y_pred = c(y_pred, predict(modlm, dfte))
    y_true = c(y_true, dfte[, target])
    coef.mat[, fold.idx] = modlm$coefficients
  }
  loss.cv = round(loss.reg(y_true, y_pred, df2=2),digit=2)
  loss=loss.reg(y_true=y_true, y_pred=y_pred, df2=mean(colSums(coef.mat!=0, na.rm=TRUE)))
  invisible(
    list(loss   = loss,
         y_true = y_true,
         y_pred = y_pred,
         coef   = coef.mat,
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

do.a.lot.of.things.glmnet<-function(DBtr, DBte, TARGET, PREDICTORS, DATA_STR, PREDICTORS_STR, BASELINE,
                                    FORCE.ALPHA, MOD.SEL.CV,
                                    bootstrap.nb=100, permutation.nb=100){
    Xtr = as.matrix(DBtr[, PREDICTORS])
    ytr = DBtr[, TARGET]
    Xte = as.matrix(DBte[, PREDICTORS])
    yte = DBte[, TARGET]
    log_file = paste("log_enet_",TARGET,"_", PREDICTORS_STR,".txt",sep="" )
    #log_file = "/tmp/toto.txt"
    RESULT = list(data=DATA_STR, method="enet", target=TARGET, predictors=PREDICTORS_STR, dim=paste(dim(Xtr), collapse="x"))
    #X=Xfr; ytr=yfr; log_file=paths$log_file;
    #cv = cross_val(length(ytr),type="k-fold", k=10, random=TRUE, seed=107)
    cv = cross_val(length(ytr),type="k-fold", k=10, random=TRUE, seed=97)
    dump("cv",file="cv.schema.R")

    cat(DATA_STR, TARGET, PREDICTORS_STR, "\n",file=log_file)
    cat(paste(rep("=", nchar(TARGET)), collapse=""),"\n",file=log_file,append=TRUE)
    cat("\ndim(Xtr)=",dim(Xtr),"\n",sep=" ",file=log_file,append=TRUE)
    
    ## Sensitivity study: choose alpha
    ## ===============================
    cat("\nSensitivity study: choose alpha\n",file=log_file,append=TRUE)
    cat("-------------------------------\n",file=log_file,append=TRUE)
    if(is.null(FORCE.ALPHA)){print("ERROR FORCE.ALPHA is null"); return(1)}
#     if(is.null(FORCE.ALPHA)){
#     # select the L2 mixing parameter
#     #cv.gs.glmnet=cv_glmnet_gridsearch(Xtr, ytr, cv, nlambda=100, alphas=seq(0,1,.1))
#     #res=cv_glmnet_gridsearch.param.select(cv.gs.glmnet, plot.it=T, fig.output="cv.gridsearch.glmnet")
#     cv.gs.glmnet=cv.enet.gridsearch(Xtr, ytr, cv, nlambda=100, alphas=seq(0,1,.1))
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
    source("parameters.R")
    # -
    cv.lambda.min     = cv.regression(Xtr,ytr,glmnet.cvlambda,cv,min.within.1sd=FALSE,refit.lm=FALSE,alpha=alpha)
    cv.lambda.min.glm = cv.regression(Xtr,ytr,glmnet.cvlambda,cv,min.within.1sd=FALSE,refit.lm=TRUE,alpha=alpha)
    cv.lambda.1sd     = cv.regression(Xtr,ytr,glmnet.cvlambda,cv,min.within.1sd=TRUE, refit.lm=FALSE,alpha=alpha)
    cv.lambda.1sd.glm = cv.regression(Xtr,ytr,glmnet.cvlambda,cv,min.within.1sd=TRUE, refit.lm=TRUE,alpha=alpha)

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
    auto.max.r2            =which.max(res.summary[,"r2"]),
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
    #load("cv_bestcv_glmnet.Rdata")
    loss.cv = round(cv_enet$loss, digit=2)
    sink(log_file, append = TRUE)
    print(loss.cv)
    sink()
    loss.cv = loss.cv[c("r2", "cor", "fstat")]
    names(loss.cv) = c("r2_cv", "cor_cv", "fstat_cv")
    RESULT = c(RESULT, loss.cv)

    ## Fit and predict all (same train) data, (look for overfit)
    ## =========================================================

    cat("\nFit and predict all (same train) data, (look for overfit)\n",file=log_file,append=TRUE)
    cat("---------------------------------------------------------\n",file=log_file,append=TRUE)
    source("parameters.R")
    all_enet=glmnet.cvlambda(Xtr, ytr, newX=Xtr, refit.lm=refit.lm, min.within.1sd=min.within.1sd, alpha=alpha)
    loss.all = round(loss.reg(ytr, all_enet$y_pred, 
        df2=sum(all_enet$coef!=0)), digit=2)
    sink(log_file, append = TRUE)
    print(loss.all)
    sink()
    loss.all = loss.all[c("r2", "cor", "fstat")]
    names(loss.all) = c("r2_all", "cor_all", "fstat_all")
    RESULT = c(RESULT, loss.all)

    ## Plot true vs predicted
    ## ======================
    #source("cv.schema.R")
    source("parameters.R")

    # - true vs. pred (CV)
    pdf("cv_bestcv_glmnet_true-vs-pred.pdf")
    d=rbind(data.frame(baseline=DBtr[unlist(cv_enet$params$cv), BASELINE], M36=cv_enet$y_true, lab="true"),
            data.frame(baseline=DBtr[unlist(cv_enet$params$cv), BASELINE], M36=cv_enet$y_pred, lab="pred"))
    p = qplot(baseline, M36, color=lab, data=d, main=paste(TARGET," - true vs. pred - CV - [r2_10CV=",loss.cv[["r2_cv"]],"]",sep="")) + geom_abline(color="grey")
    #all(DBtr[unlist(cv_enet$params$cv), TARGET] == cv_enet$y_true)
    print(p)
    dev.off()

    # - true vs. pred (no CV)
    pdf("all_bestcv_glmnet_true-vs-pred.pdf")
    d=rbind(data.frame(baseline=DBtr[, BASELINE], M36=all_enet$y_true, lab="true"),
            data.frame(baseline=DBtr[, BASELINE], M36=all_enet$y_pred, lab="pred"))
    p = qplot(baseline, M36, color=lab, data=d, main=paste(TARGET," - true vs. pred - no CV - [r2=",loss.all[["r2_all"]],"]",sep="")) + geom_abline(color="grey")
    print(p)
    dev.off()
    
    ## Coeficients
    ## ===========
    cat("\nCoeficients (+ sd bootstarping):\n",file=log_file,append=TRUE)
    cat("--------------------------------\n",file=log_file,append=TRUE)
    # - (a) do the permutations
    source("parameters.R")
    #load(file="all_bestcv_glmnet.Rdata")
    coef.names = names(all_enet$coef)

    #if(file.exists("bootstrap_bestcv_glmnet.Rdata.tmp"))load("bootstrap_bestcv_glmnet.Rdata.tmp")
    #load("perms.cv_bestcv_glmnet.Rdata.tmp")
    #if(!exists("bootstrap.list"))
    bootstrap.list=list()
    bootstrap.idx=(length(bootstrap.list)+1):bootstrap.nb

    #bootstrap=1:100
    for(boot.i in bootstrap.idx){
        cat("BOOTSTRAP",boot.i,"*****************\n")
        resamp.idx=sample.int(length(ytr), length(ytr), replace=T)

        #res=cv_glmnet_lambda(Xtr,ytr.rnd,cv,min.within.1sd=FALSE,alpha=alpha)
        res = cv.lambda.min.glm = glmnet.cvlambda(
            X=Xtr[resamp.idx,], y=ytr[resamp.idx], newX=Xtr[resamp.idx,],
            min.within.1sd=min.within.1sd, refit.lm=refit.lm,
            alpha=alpha)
        bootstrap.list[[boot.i]]=res
        #save(bootstrap.list,file="bootstrap_bestcv_glmnet.Rdata.tmp")
    }
    save(bootstrap.list,file="bootstrap_bestcv_glmnet.Rdata")

    # - (b) build the coef table
    #load(file="bootstrap_bestcv_glmnet.Rdata")
    #load(file="all_bestcv_glmnet.Rdata")
    #coef.names=c("intercept",colnames(Xtr))
    coef.mat=array(NA, dim=c(length(all_enet$coef),length(bootstrap.list)),
            dimnames=list(coef.names,
                          paste("boot", 1:length(bootstrap.list),sep="")))
    for(boot.i in 1:length(bootstrap.list))coef.mat[,boot.i]=bootstrap.list[[boot.i]]$coef

    coef.df=cbind(
        coef=all_enet$coef,
        boot.mean=apply(coef.mat,1,mean),
        boot.sd  =apply(coef.mat,1,sd),
        boot.count=apply(coef.mat,1,function(x)sum(x!=0)))

    coef.df=as.data.frame(coef.df)

    options(width=200)
    sink(log_file, append = TRUE)
    cat("\n#coef!=0 (",sum(coef.df$coef!=0),")\n",sep="")
    cat("~~~~~~~~~~~~~~\n")
    coef.not.null = coef.df[coef.df$coef!=0,]
    coef.not.null = coef.not.null[order(abs(coef.not.null$coef), decreasing = TRUE),]
    print(coef.not.null)

    cat("\ncoef!=0  in more than 50% of boostraped sample\n")
    cat("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
    coef.more.50 = coef.df[coef.df$boot.count>(bootstrap.nb/2),]
    coef.more.50 = coef.more.50[order(coef.more.50$boot.count, decreasing = TRUE),]
    print(coef.more.50)
    sink()


    ## Permutations: calibrate E_10CV(r2)
    ## ==================================

    cat("\nPermutations: calibrate E_10CV:\n",file=log_file,append=TRUE)
    cat("-------------------------------\n",file=log_file,append=TRUE)
    #source("cv.schema.R")
    source("parameters.R")
    # -
    permutations = 1:permutation.nb
    perms.cv.list=list(); perms.all.list=list()
    for(perm.i in permutations){
        cat("\nPERM",perm.i,"*****************\n")
        ytr.rnd=sample(ytr)
        #loss CV
        loss.cv = cv.regression(Xtr, ytr.rnd, glmnet.cvlambda, cv,
            min.within.1sd=min.within.1sd, refit.lm=refit.lm,
            alpha=alpha)$loss
        # loss all
        all_bestcv_glmnet_rnd=glmnet.cvlambda(Xtr, ytr.rnd, newX=Xtr, refit.lm=refit.lm, min.within.1sd=min.within.1sd, alpha=alpha)
        loss.all = loss.reg(ytr.rnd, all_bestcv_glmnet_rnd$y_pred, df2=sum(all_bestcv_glmnet_rnd$coef!=0))

        #cat("\n");print(data.frame(lapply(res,function(x)mean(x,na.rm=TRUE))))
        perms.cv.list[[perm.i]]=loss.cv
        perms.all.list[[perm.i]]=loss.all
        #save(perms.list,file="perms.cv_bestcv_glmnet.Rdata.tmp")
    }
    perms.lists = list(perms.cv.list=perms.cv.list, perms.all.list=perms.all.list)
    save(perms.lists, file="perms_bestcv_glmnet.Rdata")

    cat("# Compute p-values\n", file=log_file, append=TRUE)
    cat("## ---------------\n", file=log_file, append=TRUE)

    #load(file="perms_bestcv_glmnet.Rdata")
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

    #load(file="cv_bestcv_glmnet.Rdata")
    true.loss.cv = cv_enet$loss
    #load(file="all_bestcv_glmnet.Rdata")
    true.loss.all = loss.reg(ytr, all_enet$y_pred, df2=sum(all_enet$coef!=0))

    perm.pval = rbind(
        data.frame(c(nperms=nrow(perms.loss.all), type="all", as.list(colSums(true.loss.all<perms.loss.all,na.rm=T)))),
        data.frame(c(nperms=nrow(perms.loss.cv),  type="cv",  as.list(colSums(true.loss.cv<perms.loss.cv,na.rm=T)))))

    sink(log_file, append = TRUE)
    cat("perms>true ?\n")
    print(perm.pval)
    sink()

    cat("\nGeneralize on Test dataset:\n",file=log_file,append=TRUE)
    cat("-------------------------------\n",file=log_file,append=TRUE)
#    load("all_bestcv_glmnet.Rdata")
    y_pred_tr = as.vector(predict(all_enet$mod, Xtr))
    y_pred_te  = as.vector(predict(all_enet$mod, Xte))
    losstr = round(loss.reg(ytr, y_pred_tr, df2=sum(all_enet$coef!=0)), digit=2)
    loss_te  = round(loss.reg(yte,  y_pred_te,  df2=sum(all_enet$coef!=0)), digit=2)
    sink(log_file, append = TRUE)
    print(rbind(c(center=1,losstr),c(center=2,loss_te)))
    sink()
    loss_te = loss_te[c("r2", "cor", "fstat")]
    names(loss_te) = c("r2_test", "cor_test", "fstat_test")
    RESULT = c(RESULT, loss_te)
    all_enet$y_pred_te = y_pred_te
    all_enet$y_true_te = yte
    all_enet$loss_te = loss_te
    
    # - true vs. pred (no CV)
    pdf("all_bestcv_glmnet_true-vs-pred_test.pdf")
    d=rbind(data.frame(baseline=DBte[, BASELINE], M36=yte, lab="true"),
            data.frame(baseline=DBte[, BASELINE], M36=y_pred_te, lab="pred"))
    p = qplot(baseline, M36, color=lab, data=d, main=paste(TARGET," - true vs. pred - TEST - [r2=",loss_te[["r2_test"]],"]",sep="")) + geom_abline(color="grey")
    print(p)
    dev.off()
    
    # - Permutation
    perms.list=list()
    permutations=1:permutation.nb

    for(perm.i in permutations){
        cat("PERM",perm.i,"*****************\n")
        y.rnd=sample(yte)
        loss=loss.reg(y_true=y.rnd,   y_pred=y_pred_te, df2=sum(all_enet$coef!=0))
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
    print(c(nperms=nrow(perms),colSums(perms>loss_te,na.rm=T)))
    sink()
    #loss_te = loss_te[c("r2", "cor", "fstat")]
    #names(loss_te) = c("r2_test", "cor_test", "fstat_test")
    model = all_enet
    save(model, file="all_bestcv_glmnet.Rdata")
    return(RESULT)
}


do.a.lot.of.things.glm<-function(DBtr, DBte, TARGET, INTERCEPT, PREDICTORS, DATA_STR, PREDICTORS_STR, BASELINE,
                                    bootstrap.nb=100, permutation.nb=100){
  log_file = paste("log_enet_",TARGET,"_", PREDICTORS_STR,".txt",sep="" )
  RESULT = list(data=DATA_STR, method="glm", target=TARGET, predictors=PREDICTORS_STR, dim=paste(nrow(DBtr), length(PREDICTORS), sep="x"))
  cv = cross_val(length(DBtr[, TARGET]),type="k-fold", k=10, random=TRUE, seed=97)
  dump("cv",file="cv.schema.R")
  
  cat(DATA_STR, TARGET, PREDICTORS_STR, "\n",file=log_file)
  cat(paste(rep("=", nchar(TARGET)), collapse=""),"\n",file=log_file,append=TRUE)
  cat("\ndim(Xtr)=",nrow(DBtr), length(PREDICTORS),"\n",sep=" ",file=log_file,append=TRUE)
  
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
  loss.cv = cv_glm$loss[c("r2", "cor", "fstat")]
  names(loss.cv) = c("r2_cv", "cor_cv", "fstat_cv")
  RESULT = c(RESULT, loss.cv)
  
  ## Fit and predict all (same train) data, (look for overfit)
  ## =========================================================
  cat("\nFit and predict all (same train) data, (look for overfit)\n",file=log_file,append=TRUE)
  cat("---------------------------------------------------------\n",file=log_file,append=TRUE)
  if(INTERCEPT) formula = formula(paste(TARGET,"~", paste(PREDICTORS, collapse='+'))) else formula = formula(paste(TARGET,"~", paste(PREDICTORS, collapse='+'), "-1"))
  modlm = lm(formula, data = DBtr)
  all_glm = list(y_true=DBtr[, TARGET], y_pred=predict(modlm, DBtr), coef=modlm$coefficients, mod=modlm)
  all_glm$loss=round(loss.reg(all_glm$y_true, all_glm$y_pred, 
                              df2=sum(all_glm$coef!=0)), digit=2)
  sink(log_file, append = TRUE)
  print(all_glm$loss)
  sink()
  loss.all = all_glm$loss[c("r2", "cor", "fstat")]
  names(loss.all) = c("r2_all", "cor_all", "fstat_all")
  RESULT = c(RESULT, loss.all)
  
  ## Plot true vs predicted
  ## ======================
  # - true vs. pred (CV)
  pdf("cv_glm_true-vs-pred.pdf")
  d=rbind(data.frame(baseline=DBtr[unlist(cv_glm$params$cv), BASELINE], M36=cv_glm$y_true, lab="true"),
          data.frame(baseline=DBtr[unlist(cv_glm$params$cv), BASELINE], M36=cv_glm$y_pred, lab="pred"))
  p = qplot(baseline, M36, color=lab, data=d, main=paste(TARGET," - true vs. pred - CV - [r2_10CV=",loss.cv[["r2_cv"]],"]",sep="")) + geom_abline(color="grey")
  #all(DBtr[unlist(cv_enet$params$cv), TARGET] == cv_enet$y_true)
  print(p)
  dev.off()
  
  # - true vs. pred (no CV)
  pdf("all_glm_true-vs-pred.pdf")
  d=rbind(data.frame(baseline=DBtr[, BASELINE], M36=all_glm$y_true, lab="true"),
          data.frame(baseline=DBtr[, BASELINE], M36=all_glm$y_pred, lab="pred"))
  p = qplot(baseline, M36, color=lab, data=d, main=paste(TARGET," - true vs. pred - no CV - [r2=",loss.all[["r2_all"]],"]",sep="")) + geom_abline(color="grey")
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
    boot_glm = list(y_true=DBtr.boot[, TARGET], y_pred=predict(modlm, DBtr.boot),
                    coef=modlm.boot$coefficients, mod=modlm.boot)
    boot_glm$loss=round(loss.reg(boot_glm$y_true, boot_glm$y_pred, 
                                df2=sum(all_glm$coef!=0)), digit=2)
    bootstrap.list[[boot.i]] = boot_glm
    #save(bootstrap.list,file="bootstrap_bestcv_glmnet.Rdata.tmp")
  }
  save(bootstrap.list,file="bootstrap_glm.Rdata")
  
  # - (b) build the coef table
  coef.mat=array(NA, dim=c(length(all_glm$coef),length(bootstrap.list)),
                  dimnames=list(names(all_glm$coef),
                                paste("boot", 1:length(bootstrap.list),sep="")))
  for(boot.i in 1:length(bootstrap.list))coef.mat[,boot.i]=bootstrap.list[[boot.i]]$coef
  
  coef.df=cbind(
    coef=all_glm$coef,
    boot.mean=apply(coef.mat,1,mean),
    boot.sd  =apply(coef.mat,1,sd),
    boot.count=apply(coef.mat,1,function(x)sum(x!=0)))
  
  coef.df=as.data.frame(coef.df)
  
  options(width=200)
  sink(log_file, append = TRUE)
  cat("\n#coef!=0 (",sum(coef.df$coef!=0),")\n",sep="")
  cat("~~~~~~~~~~~~~~\n")
  coef.not.null = coef.df[coef.df$coef!=0,]
  coef.not.null = coef.not.null[order(abs(coef.not.null$coef), decreasing = TRUE),]
  print(coef.df)
  sink()

  ## Permutations: calibrate E_10CV(r2)
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
                                df2=sum(all_glm$coef!=0)), digit=2)
    
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

  y_pred_tr = predict(all_glm$mod, DBtr)
  y_pred_te  = predict(all_glm$mod, DBte)
  loss_tr = round(loss.reg(DBtr[, TARGET], y_pred_tr, df2=sum(all_glm$coef!=0)), digit=2)
  loss_te = round(loss.reg(DBte[, TARGET], y_pred_te,  df2=sum(all_glm$coef!=0)), digit=2)  
  sink(log_file, append = TRUE)
  print(rbind(c(center=1,loss_tr),c(center=2,loss_te)))
  sink()
  loss_te = loss_te[c("r2", "cor", "fstat")]
  names(loss_te) = c("r2_test", "cor_test", "fstat_test")
  RESULT = c(RESULT, loss_te)
  all_glm$y_pred_te = y_pred_te
  all_glm$y_true_te = DBte[, TARGET]
  all_glm$loss_te = loss_te
  
  pdf("all_glm_true-vs-pred_test.pdf")
  d=rbind(data.frame(baseline=DBte[, BASELINE], M36=DBte[, TARGET], lab="true"),
          data.frame(baseline=DBte[, BASELINE], M36=y_pred_te, lab="pred"))
  p = qplot(baseline, M36, color=lab, data=d, main=paste(TARGET," - true vs. pred - TEST - [r2=",loss_te[["r2_test"]],"]",sep="")) + geom_abline(color="grey")
  print(p)
  dev.off()
  
  perms.list=list()
  permutations=1:permutation.nb
  for(perm.i in permutations){
    cat("PERM",perm.i,"*****************\n")
    loss_te.perm = loss.reg(y_true=sample(DBte[, TARGET]),   y_pred=y_pred_te, df2=sum(all_glm$coef!=0))
    perms.list[[perm.i]]=loss_te.perm
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
  print(c(nperms=nrow(perms),colSums(perms>loss_te,na.rm=T)))
  sink()
  model = all_glm
  save(model, file="all_glm.Rdata")
  return(RESULT)
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
#     mod1.fr = all_enet1$mod
#     y_pred1.fr = predict(mod1.fr, Xytr1$X)
#     y_pred1.d  = predict(mod1.fr, Xyte1$X)
#     loss1.fr = round(loss.reg(Xytr1$y, y_pred1.fr, df2=sum(all_enet1$coef!=0)), digit=2)
#     loss1.d  = round(loss.reg(Xyte1$y,  y_pred1.d,  df2=sum(all_enet1$coef!=0)), digit=2)
#     # mod2
#     load(f2)
#     all_enet2 = all_enet
#     rm(all_enet)
#     mod2.fr = all_enet2$mod
#     y_pred2.fr = predict(mod2.fr, Xytr2$X)
#     y_pred2.d  = predict(mod2.fr, Xyte2$X)
#     loss2.fr = round(loss.reg(Xytr2$y, y_pred2.fr, df2=sum(all_enet2$coef!=0)), digit=2)
#     loss2.d  = round(loss.reg(Xyte2$y,  y_pred2.d,  df2=sum(all_enet2$coef!=0)), digit=2)
# 
#     # predictors
#     w1 = as.matrix(all_enet1$mod$beta)
#     w1.col = rownames(w1)
#     w1.col.nonnull = rownames(w1)[w1!=0]
#     w2 = as.matrix(all_enet2$mod$beta)
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
#     rss1.d = sum((y_pred1.d - Xyte1$y)^2)
#     rss2.d = sum((y_pred2.d - Xyte2$y)^2)
# 
#     n = length(y_pred1.d)
#     fstat = ((rss1.d - rss2.d) / (df2 - df1)) / (rss2.d / (n - df1))
# 
#     return(data.frame(var=TARGET, fstat=fstat, p.pval=1-pf(fstat, df2 - df1, n - df2), df1, df2, n.pred1=length(w1.col.nonnull), n.pred2=length(w2.col.nonnull), n.in1.not.in2=length(in1.not.in2), n.in2.not.in1=length(in2.not.in1), in2.not.in1=paste(in2.not.in1, collapse=","), r2.1.d=loss1.d["r2"], r2.2.d=loss2.d["r2"]))
# }

