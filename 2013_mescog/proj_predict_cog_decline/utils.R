################################################################################################
## READ INPUT
################################################################################################
read_db=function(infile, to_remove = c()){#, to_remove = c("DELTA_BP", "TRIGLY", "MIGRAINE_WITH_AURA", "AVC", 
                         #                         "TRBEQUILIBRE", "TRBMARCHE", "DEMENTIA",
                         #                         "HYPERTENSION", "HYPERCHOL", "HDL", "FAST_GLUC", "NIHSS",
                         #                         "LLVn", "WMHVn", "BRAINVOL", "LLcount", "BARTHEL")){
  DB = read.csv(infile, header=TRUE, as.is=TRUE)
  col_info = c("ID", "SITE")
  col_targets =   c("TMTB_TIME.M36","MDRS_TOTAL.M36","MRS.M36","MMSE.M36")
  col_baselines =   c("TMTB_TIME","MDRS_TOTAL","MRS","MMSE")
  to_remove = unique(c(to_remove, col_info, col_targets, colnames(DB)[grep("36", colnames(DB))]))
  col_predictors = colnames(DB)[!(colnames(DB) %in% to_remove)]
  col_niglob = col_predictors[grep("LLV|LLcount|WMHV|MBcount|BPF", col_predictors)]
  col_clinic = col_predictors[!(col_predictors %in% col_niglob)]

  if(!all(sort(colnames(DB)) == sort(unique(c(col_info, col_targets, col_clinic, col_niglob, to_remove))))){
    print("ERROR COLNAMES DO NOT MATCH")
    sys.on.exit()
  }
  return(list(DB=DB, col_info=col_info,
              col_targets=col_targets, col_clinic=col_clinic, col_niglob=col_niglob, col_baselines=col_baselines))
}

################################################################################################
## MISSING DATA
################################################################################################

imput_missing<-function(D, skip=c()){
  # imput all missing values by colmean except M36
  #targets = colnames(D)[grep("M36",colnames(D))]
  predictors = colnames(D)[!(colnames(D) %in% skip)]
  Dimp = D
  imputation = NULL
  for(v in predictors){
    #v = "TMTB_TIME"
    for(i in which(is.na(D[,v]))){
      predictors_i = names(D[i, predictors])[!is.na(D[i, predictors]) & !(names(D[i, predictors]) %in% c("ID", "SITE"))]
      f_str = paste(v,paste(predictors_i, collapse="+"),sep="~")
      f = formula(f_str)
      mod = lm(f, D)
      Dimp[i, v] = predict(mod, D[i,,drop=F])
      imputation = rbind(imputation, data.frame(var=v, ID=D$ID[i], model=f_str))
    }
    #D[is.na(D[,v]), ] = mean(D[,v], na.rm=T)
  }
  return(list(Dimputed = Dimp, models=imputation))
}

################################################################################################
## SPLIT DATASET
################################################################################################

split_db_site_stratified_with_same_target_distribution_rm_na<-function(D, TARGET){
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

kfold_site_stratified_rm_na<-function(D, TARGET){
  #D = db$DB; TARGET="TMTB_TIME.M36"
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