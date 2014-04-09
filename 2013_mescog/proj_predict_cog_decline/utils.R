################################################################################################
## READ INPUT
################################################################################################
variables<-function(){
  col_info      = c("ID", "SITE")
  col_targets   = c("TMTB_TIME.M36","MDRS_TOTAL.M36","MRS.M36","MMSE.M36")
  col_baselines = c("TMTB_TIME","MDRS_TOTAL","MRS","MMSE")
#  col_niglob    = c("LLV", "LLcount", "WMHV", "MBcount","BPF")
  col_niglob    = c("LLV","BPF")#, "LLcount" "WMHV", "MBcount") 
  col_clinic    = c("AGE_AT_INCLUSION", "SEX", "EDUCATION", "SYS_BP", "DIA_BP", "SMOKING", "LDL", "HOMOCYSTEIN", "HBA1C", "CRP17", "ALCOHOL")
  return(list(col_info=col_info, col_targets=col_targets, col_baselines=col_baselines, col_niglob=col_niglob, col_clinic=col_clinic))
}

read_db=function(infile, to_remove = c()){#, to_remove = c("DELTA_BP", "TRIGLY", "MIGRAINE_WITH_AURA", "AVC", 
                         #                         "TRBEQUILIBRE", "TRBMARCHE", "DEMENTIA",
                         #                         "HYPERTENSION", "HYPERCHOL", "HDL", "FAST_GLUC", "NIHSS",
                         #                         "LLVn", "WMHVn", "BRAINVOL", "LLcount", "BARTHEL")){
  DB = read.csv(infile, header=TRUE, as.is=TRUE)
#   to_remove = unique(c(to_remove, col_info, col_targets, colnames(DB)[grep("36", colnames(DB))]))
#   col_predictors = colnames(DB)[!(colnames(DB) %in% to_remove)]
#   col_niglob = col_predictors[grep("LLV|LLcount|WMHV|MBcount|BPF", col_predictors)]
#   col_clinic = col_predictors[!(col_predictors %in% col_niglob)]
# 
#   if(!all(sort(colnames(DB)) == sort(unique(c(col_info, col_targets, col_clinic, col_niglob, to_remove))))){
#     print("ERROR COLNAMES DO NOT MATCH")
#     sys.on.exit()
  return(c(list(DB=DB), variables()))
}

################################################################################################
## MISSING DATA
################################################################################################

imput_missing_lm<-function(D, id, to_impute, predictors){
  #if(is.null(col_toimpute)) col_toimpute = colnames(D) 
  #predictors = colnames(D)[!(colnames(D) %in% skip)]
  Dimp = D
  imputation = NULL
  for(v in to_impute){
    #v = "TMTB_TIME"
    for(i in which(is.na(D[,v]))){
      #predictors_i = names(D[i, predictors])[!is.na(D[i, predictors]) & !(names(D[i, predictors]) %in% c("ID", "SITE"))]
      predictors_i = names(D[i, predictors])[!is.na(D[i, predictors])]
      f_str = paste(v,paste(predictors_i, collapse="+"),sep="~")
      f = formula(f_str)
      mod = lm(f, D)
      Dimp[i, v] = predict(mod, D[i,,drop=F])
      imputation = rbind(imputation, data.frame(var=v, ID=id[i], model=f_str))
    }
    #D[is.na(D[,v]), ] = mean(D[,v], na.rm=T)
  }
  attr(Dimp, "models") = imputation
  return(Dimp)
}

imput_missing_mean<-function(D, id, to_impute){
  imputation = NULL
  for(v in to_impute){
    missing = is.na(D[, v])
    if(sum(missing) >0){
      mu = mean(D[,v], na.rm=T)
      #cat("impute",sum(missing), v, "by", mu, "\n")
      D[missing, v] = mu
      imputation = rbind(imputation, data.frame(var=v, ID=id[missing], model=mu))
    }
  }
  attr(D, "models") = imputation
  return(D)
}

################################################################################################
## SPLIT DATASET
################################################################################################

twofold_site_stratified_with_same_target_distribution_rm_na<-function(D, TARGET){
  D = D[!is.na(D[, TARGET]),]
  Dfr = D[D$SITE == "FR",]
  Dge = D[D$SITE == "GE",]
  
  split_idx_by_pairs<-function(idx){
    set1 = c()
    set2 = c()
    for(i in seq(1, length(idx), 2)){
      if(i+1 <= length(idx)){
        tmp = sample(idx[i:(i+1)])
        set1 = c(set1, tmp[1])
        set2 = c(set2, tmp[2])
      }else{
        set1 = c(set1, idx[i])
      }
    }
    return(list(set1, set2))
  }
  #Dfr
  sfr = split_idx_by_pairs(order(Dfr[, TARGET]))
  #Dge
  sge = split_idx_by_pairs(order(Dge[, TARGET]))
  D1 = rbind(Dfr[sfr[[1]],], Dge[sge[[1]], ])
  D2 = rbind(Dfr[sfr[[2]],], Dge[sge[[2]], ])
  splits = list(list(tr=D1, te=D2), list(tr=D1, te=D2))
  attr(splits, "D1_summary") = summary(D1[, TARGET])
  attr(splits, "D2_summary") = summary(D2[, TARGET])
  return(splits)
}

twofold_bysite_rm_na<-function(D, TARGET){
  D = D[!is.na(D[, TARGET]),]
  Dfr = D[D$SITE == "FR",]
  Dge = D[D$SITE == "GE",]
  splits = list(list(tr=Dfr, te=Dge), list(tr=Dge, te=Dfr))
  return(splits)
}

kfold_site_stratified_rm_na<-function(D, TARGET, k=10){
  #D = db$DB; TARGET="TMTB_TIME.M36"
  D = D[!is.na(D[, TARGET]),]
  #D$SITE = as.character(D$SITE)
  Dfr = D[D$SITE == "FR",]
  Dge = D[D$SITE == "GE",]
  cvfr = cross_val(nrow(Dfr), type="k-fold", k=k)
  cvge = cross_val(nrow(Dge), type="k-fold", k=k)
  splits = list()
  for(f in 1:k){
    tr = rbind(Dfr[-cvfr[[f]],],  Dge[-cvge[[f]],])
    te = rbind(Dfr[cvfr[[f]],],  Dge[cvge[[f]],])
    #cat("-----------\nTR:", dim(tr),"\n"); print(summary(tr[, TARGET])); cat("TE", dim(te),"\n"); print(summary(te[, TARGET]))
    splits[[f]] = list(tr=tr, te=te)
  }
  return(splits)
  }

################################################################################################
## Loss func
################################################################################################

loss_reg<-function(y_true, y_pred, df2=NULL, suffix=NULL){
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

################################################################################################
## utils stat
################################################################################################

## http://wiki.stdout.org/rcookbook/Graphs/Plotting%20means%20and%20error%20bars%20%28ggplot2%29/

## Summarizes data.
## Gives count, mean, standard deviation, standard error of the mean, and confidence interval (default 95%).
##   data: a data frame.
##   measurevar: the name of a column that contains the variable to be summariezed
##   groupvars: a vector containing names of columns that contain grouping variables
##   na.rm: a boolean that indicates whether to ignore NA's
##   conf.interval: the percent range of the confidence interval (default is 95%)
summarySE <- function(data=NULL, measurevar, groupvars=NULL, na.rm=FALSE,
                      conf.interval=.95, .drop=TRUE) {
  require(plyr)
  
  # New version of length which can handle NA's: if na.rm==T, don't count them
  length2 <- function (x, na.rm=FALSE) {
    if (na.rm) sum(!is.na(x))
    else       length(x)
  }
  
  # This is does the summary; it's not easy to understand...
  datac <- ddply(data, groupvars, .drop=.drop,
                 .fun= function(xx, col, na.rm) {
                   c( N    = length2(xx[,col], na.rm=na.rm),
                      mean = mean   (xx[,col], na.rm=na.rm),
                      sd   = sd     (xx[,col], na.rm=na.rm)
                   )
                 },
                 measurevar,
                 na.rm
  )
  
  # Rename the "mean" column    
  datac <- rename(datac, c("mean"=measurevar))
  
  datac$se <- datac$sd / sqrt(datac$N)  # Calculate standard error of the mean
  
  # Confidence interval multiplier for standard error
  # Calculate t-statistic for confidence interval: 
  # e.g., if conf.interval is .95, use .975 (above/below), and use df=N-1
  ciMult <- qt(conf.interval/2 + .5, datac$N-1)
  datac$ci <- datac$se * ciMult
  
  return(datac)
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

#Recursive partitioning tree with adjusted intercept
# Use a Recursive partitioning, within each cell estimate an intercept to minimize the error of the model:
# TARGET = BASELINE + intercept
rpart_inter.learn<-function(data, TARGET, BASELINE, tree){
  intercepts = c()
  counts = c()
  group = rep(NA, nrow(data))
  left = rep(TRUE, nrow(data))
  for(i in 1:length(tree)){
    node = tree[i]
    #node="MMSE<20.5"; node="LLV>=1592"
    exp = parse(text=paste("data$",node , sep=""))
    print(exp)
    subset = eval(exp)
    subset = subset & left
    group[subset] = i
    intercept = mean(data[subset, TARGET] - data[subset, BASELINE]) 
    cat(sum(subset), intercept, "\n")
    counts = c(counts, sum(subset))
    intercepts = c(intercepts, intercept)
    #sum(left & (!subset))
    left = left & (!subset)
  }
  intercept = mean(data[left, TARGET] - data[left, BASELINE])
  intercepts = c(intercepts, intercept)
  group[left] = i+1
  counts = c(counts, sum(left))
  return(list(tree=tree, intercepts=intercepts, counts=counts, group=group, TARGET=TARGET, BASELINE=BASELINE))
}

rpart_inter.predict<-function(mod, data, limits=c(-Inf, +Inf)){
  left = rep(TRUE, nrow(data))
  counts = c()
  group = rep(NA, nrow(data))
  left = rep(TRUE, nrow(data))
  predictions = rep(NA, nrow(data))
  for(i in 1:length(mod$tree)){
    node = mod$tree[i]
    exp = parse(text=paste("data$", node , sep=""))
    subset = eval(exp)
    subset = subset & left
    group[subset] = i
    predictions[subset] = data[subset, mod$BASELINE] + mod$intercepts[i]
    counts = c(counts, sum(subset))
    #sum(left & (!subset))
    left = left & (!subset)
  }
  predictions[left] = data[left, mod$BASELINE] + mod$intercepts[i+1]
  group[left] = i+1
  counts = c(counts, sum(left))
  attr(predictions, "counts") = counts
  attr(predictions, "group") = group
  predictions[predictions < limits[1]] = limits[1]
  predictions[predictions > limits[2]] = limits[2]
  return(predictions)
}



rpart.groups<-function(mod, data, i=1, group=NULL, subset=NULL, name="", ret_idx=FALSE){
  if(is.null(group)) group = rep(0, nrow(data))
  if(is.null(subset)) subset= rep(TRUE, nrow(data))
  if(class(mod) == "rpart")
    mod = cbind(var=as.character(mod$frame[,"var"]), rules = labels(mod, collapse=F))
  if(mod[i, "var"] == "<leaf>"){
    cat(i, sum(subset), "\n")
    group[subset] = i
    if(is.null(attr(group, "name"))) attr(group, "name") = list()
    attr(group, "name")[[as.character(i)]] = name
    #print(group)
    return(list(idx=i, group=group))
    }else{
      lname = paste(mod[i, "var"], mod[i, "ltemp"], sep="")
      rname = paste(mod[i, "var"], mod[i, "rtemp"], sep="") 
      lexp = paste("lsubset = data$", lname, sep="")
      rexp = paste("rsubset = data$", rname, sep="") 
      eval(parse(text=lexp)); lsubset = lsubset & subset
      eval(parse(text=rexp)); rsubset = rsubset & subset
      lret = rpart.groups(mod, data, i+1, group, lsubset, paste(name, lname, sep="/"), ret_idx=TRUE)
      #print(lret)
      rret = rpart.groups(mod, data, lret$idx+1, group, rsubset, paste(name, rname, sep="/"), ret_idx=TRUE)
      #print(rret)
      lgroup = lret$group
      rgroup = rret$group
      group = lgroup+rgroup
      attr(group, "name") = c(attr(lgroup, "name"), attr(rgroup, "name"))
      #   cat(">>>>\n")
      #   print(lgroup)
      #   print(rgroup)
      #   print(lgroup+rgroup)
      #   cat("====\n")
      if(ret_idx)
        return(list(idx=rret$idx, group=group))
      else
        return(group)
  }
}
#group = rpart.groups(rpart_mod, data=d)

#group = rpart.groups(mod, data)
#group

# Learn lm on subgroups of samples
subgrouplm.learn<-function(group, data, TARGET, BASELINE){
  models = list()
  counts = c()
  formula = formula(paste(TARGET , "~", BASELINE))
  for(g in unique(group)){
    model = lm(formula, data=data, subset=group==g)
    counts = c(counts, sum(group==g))
    models[[as.character(g)]] = model
  }
  return(list(models=models, counts=counts, group=group, TARGET=TARGET, BASELINE=BASELINE))
}

subgrouplm.predict<-function(subgrouplm.mod, group, data, limits=c(-Inf, +Inf)){
  #left = rep(TRUE, nrow(data))
  counts = c()
  #group = rep(NA, nrow(data))
  #left = rep(TRUE, nrow(data))
  predictions = rep(NA, nrow(data))
  #attach(data)
  for(gstr in names(subgrouplm.mod$models)){
    g = as.integer(gstr)
    predictions[group==g] = predict(subgrouplm.mod$models[[gstr]], newdata=data[group==g,])
    counts = c(counts, sum(group==g))
  }
  #detach(data)
  #predictions[left] = predict(mod$models[[i+1]], newdata=data[left,])
  #group[left] = "left"
  #counts = c(counts, sum(left))
  attr(predictions, "counts") = counts
  attr(predictions, "group") = group
  predictions[predictions < limits[1]] = limits[1]
  predictions[predictions > limits[2]] = limits[2]
  return(predictions)
}


#Partition samples set according rules and fit multiple lm on sub-samples
partmlm.learn.old<-function(data, TARGET, BASELINE, partitions){
  models = list()
  counts = c()
  group = rep(NA, nrow(data))
  left = rep(TRUE, nrow(data))
  formula = formula(paste(TARGET , "~", BASELINE))
  attach(data)
  for(i in 1:length(partitions)){
    node = partitions[i]
    #node="MMSE<20.5"; node="LLV>=1592"
    exp = parse(text=node)
    #print(exp)
    subset = eval(exp)
    subset = subset & left
    group[subset] = node
    model = lm(formula, data=data, subset=subset)
    #cat(sum(subset), intercept, "\n")
    counts = c(counts, sum(subset))
    models[[i]] = model
    #sum(left & (!subset))
    left = left & (!subset)
  }
  detach(data)
  model = lm(formula, data=data, subset=left)
  models[[i+1]] = model
  group[left] = "left"
  counts = c(counts, sum(left))
  return(list(partitions=partitions, models=models, counts=counts, group=group, TARGET=TARGET, BASELINE=BASELINE))
}

partmlm.predict.old<-function(mod, data, limits=c(-Inf, +Inf)){
  left = rep(TRUE, nrow(data))
  counts = c()
  group = rep(NA, nrow(data))
  left = rep(TRUE, nrow(data))
  predictions = rep(NA, nrow(data))
  attach(data)
  for(i in 1:length(mod$partitions)){
    node = mod$partitions[i]
    exp = parse(text=node)
    subset = eval(exp)
    subset = subset & left
    group[subset] = node
    predictions[subset] = predict(mod$models[[i]], newdata=data[subset,])
    counts = c(counts, sum(subset))
    #sum(left & (!subset))
    left = left & (!subset)
  }
  detach(data)
  predictions[left] = predict(mod$models[[i+1]], newdata=data[left,])
  group[left] = "left"
  counts = c(counts, sum(left))
  attr(predictions, "counts") = counts
  attr(predictions, "group") = group
  predictions[predictions < limits[1]] = limits[1]
  predictions[predictions > limits[2]] = limits[2]
  return(predictions)
}

################################################################################################
## RESULTS_TAB
################################################################################################

RESULTS_TAB_summarize_diff<-function(R, KEYS){
  KEYS = c(KEYS, "TARGET")
  R = rbind(R[(R$PREDICTORS == "BASELINE") & (R$MODEL =="GLM"),], R[(R$MODEL =="ENET"),])
  #R = R[,c("ALPHA","SEED","TARGET","PREDICTORS", "r2_te", "r2_tr", "r2_te_se")]
  # 4 TARGETs x 4 PREDICTORS x 2 score x 2 SEED x  x 11 ALPHA
  nrow(R) == 4 * 4 * length(SEEDS) * length(ALPHAS)
  
  #ddply(R, .(MODEL, ALPHA, SEED, PREDICTORS), summarise,mean=mean(r2_te),sd=sd(r2_te))
  # Diff pairs of PREDICTORS
  
  b    = R[R$PREDICTORS == "BASELINE",               ]
  bni  = R[R$PREDICTORS == "BASELINE+NIGLOB",        ]
  bc   = R[R$PREDICTORS == "BASELINE+CLINIC",        ]
  bcni = R[R$PREDICTORS == "BASELINE+CLINIC+NIGLOB", ]
  m1 = merge(b, bni, by=KEYS, suffixes=c("_b", "_bni"))
  m2 = merge(bc, bcni, by=KEYS, suffixes=c("_bc", "_bcni"))
  m = merge(m1, m2, by=KEYS)
  m$diff_te = (m$r2_te_bni - m$r2_te_b) + (m$r2_te_bcni - m$r2_te_bc)
  diff_by_keys = m[, c(KEYS, "diff_te")]
  nrow(m) == 4 * length(SEEDS)
  ## Average over target
  KEYS_NO_TARGET = KEYS[!(KEYS %in% "TARGET")]
  diff_by_keys_average_targets = ddply(diff_by_keys, as.quoted(KEYS_NO_TARGET), summarise, diff_te_mu=mean(diff_te), diff_te_sd=sd(diff_te))
  return(list(max=diff_by_keys_average_targets[which.max(diff_by_keys_average_targets$diff_te_mu), ], diff_by_keys=diff_by_keys))
}