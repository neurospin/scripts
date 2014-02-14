library(rpart)
library(ggplot2)

SRC = paste(Sys.getenv("HOME"),"git/scripts/2013_mescog/proj_predict_cog_decline",sep="/")
source(paste(SRC,"utils.R",sep="/"))

BASE_DIR = "/neurospin/mescog/proj_predict_cog_decline"

# INPUT ---
#INPUT_DATA = paste(BASE_DIR, "data", "dataset_clinic_niglob_20140128_imputed_lm.csv", sep="/")
#INPUT_DATA = paste(BASE_DIR, "data", "dataset_clinic_niglob_20140204_nomissing_BPF-LLV_imputed_lm.csv", sep="/")
#INPUT_DATA = paste(BASE_DIR, "data", "dataset_clinic_niglob_20140205_nomissing_BPF-LLV_imputed.csv", sep="/")
INPUT_DATA = paste(BASE_DIR, "data", "dataset_clinic_niglob_20140205_nomissing_BPF-LLV_imputed.csv", sep="/")
INPUT = "/neurospin/mescog/proj_predict_cog_decline/20140205_nomissing_BPF-LLV_imputed"
OUTPUT=INPUT

VALIDATION = "CV"

db = read_db(INPUT_DATA)

## -------------------------------------------------------------------------------------------
ERR = NULL
SUMMARY = NULL

pdf(paste(OUTPUT, "error_refitall_rpart_tree_M36_by_M0.pdf", sep="/"))

for(TARGET in db$col_targets){
  # TARGET = "TMTB_TIME.M36"
  # TARGET =  "MMSE.M36"
  cat("== ", TARGET, " =============================================================================== \n" )
  d = db$DB[!is.na(db$DB[, TARGET]),]
  BASELINE = strsplit(TARGET, "[.]")[[1]][1]
  # NO NI
  PREDICTORS_STR = "BASELINE"
  cat("-- ", PREDICTORS_STR, "----------------------------------------------------------------------- \n" )
  PREDICTORS = BASELINE
  formula = formula(paste(TARGET,"~", paste(PREDICTORS, collapse='+')))
  #formula = formula(paste(TARGET,"~", paste(PREDICTORS, collapse='+'), "-1"))
  mod = rpart(formula, data=d)
  M0 = d[, BASELINE]
  M36_true=d[, TARGET]
  M36_pred= predict(mod, d)
  M36_err = M36_pred - M36_true
  M36_err_abs = abs(M36_err)
  print(loss_reg(y_true=M36_true, y_pred=M36_pred))
  plot(mod, uniform=TRUE, main=paste(TARGET, "~", PREDICTORS_STR, "R2=", round(loss_reg(y_true=M36_true, y_pred=M36_pred)["r2"][[1]],2)))
  text(mod, use.n=TRUE, all=TRUE, cex=.8)
  ERR=rbind(ERR, data.frame(TARGET=TARGET,PREDICTORS=PREDICTORS_STR, ID=d$ID, dim=paste(dim(d), collapse="x"), M0=M0, M36_true=M36_true, M36_pred=M36_pred, M36_err, M36_err_abs))
  print(mod)
  
  # With NI
  PREDICTORS_STR = "BASELINE+NIGLOB"
  cat("-- ", PREDICTORS_STR, "----------------------------------------------------------------------- \n" )
  PREDICTORS = unique(c(BASELINE,db$col_niglob))
  formula = formula(paste(TARGET,"~", paste(PREDICTORS, collapse='+')))
  #formula = formula(paste(TARGET,"~", paste(PREDICTORS, collapse='+'), "-1"))
  mod_ni = rpart(formula, data=d)
  M0 = d[, BASELINE]
  M36_true=d[, TARGET]
  M36_pred= predict(mod_ni, d)
  print(loss_reg(y_true=M36_true, y_pred=M36_pred))
  plot(mod_ni, uniform=TRUE, main=paste(TARGET, "~", PREDICTORS_STR, "R2=", round(loss_reg(y_true=M36_true, y_pred=M36_pred)["r2"][[1]],2)))
  text(mod_ni, use.n=TRUE, all=TRUE, cex=.8)
  
  M36_err = M36_pred - M36_true
  M36_err_abs = abs(M36_err)
  r = data.frame(TARGET=TARGET,PREDICTORS=PREDICTORS_STR, ID=d$ID, dim=paste(dim(d), collapse="x"), M0=M0, M36_true=M36_true, M36_pred=M36_pred, M36_err, M36_err_abs)
  ERR=rbind(ERR, r)
  print(mod_ni)
  #cat("-- COMPARISON (ANOVA) ----------------------------------------------------------------------- \n" )
  ## PLOT
  r$GROUP = as.factor(mod_ni$where)
  p = ggplot(r, aes(x = M0, y = M36_true)) + geom_point(alpha=.7, aes(colour=GROUP), position = "jitter") + 
    geom_abline(linetype="dotted") + ggtitle(paste(TARGET, "~", PREDICTORS_STR))
  #x11()
  print(p)
  ##
}

write.csv(ERR, paste(OUTPUT, "error_refitall_rpart_M36_by_M0.csv", sep="/"), row.names=FALSE)

dev.off()
#write.csv(ERR, paste(OUTPUT, "error_refitallglm_M36_by_M0.csv", sep="/"), row.names=FALSE)

#########################################################################################################
## MMSE

if(FALSE){
TARGET =  "MMSE.M36"
cat("== ", TARGET, " =============================================================================== \n" )
d = db$DB[!is.na(db$DB[, TARGET]),]
BASELINE = strsplit(TARGET, "[.]")[[1]][1]
# NO NI
#PREDICTORS_STR = "BASELINE+NIGLOB"
#cat("-- ", PREDICTORS_STR, "----------------------------------------------------------------------- \n" )
#PREDICTORS = PREDICTORS = unique(c(BASELINE,db$col_niglob))
# formula = formula(paste(TARGET,"~", paste(PREDICTORS, collapse='+')))
# #formula = formula(paste(TARGET,"~", paste(PREDICTORS, collapse='+'), "-1"))
# mod = rpart(formula, data=d)
# M0 = d[, BASELINE]
# M36_true=d[, TARGET]
# M36_pred= predict(mod, d)
# M36_err = M36_pred - M36_true
# M36_err_abs = abs(M36_err)
# print(loss_reg(y_true=M36_true, y_pred=M36_pred))
# plot(mod, uniform=TRUE, main=paste(TARGET, "~", PREDICTORS_STR, "R2=", round(loss_reg(y_true=M36_true, y_pred=M36_pred)["r2"][[1]],2)))
# text(mod, use.n=TRUE, all=TRUE, cex=.8)
# print(mod)
# ERR=rbind(ERR, data.frame(TARGET=TARGET,PREDICTORS=PREDICTORS_STR, ID=d$ID, dim=paste(dim(d), collapse="x"), M0=M0, M36_true=M36_true, M36_pred=M36_pred, M36_err, M36_err_abs))
# print(summary(mod))



interpect_by_cuttree.learn<-function(data, TARGET, BASELINE, tree){
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

interpect_by_cuttree.predict<-function(mod, data){
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
  return(predictions)
}

tree = c("MMSE<20.5", "LLV>=1592", "MMSE>=25.5", "BPF<0.773")
mod = interpect_by_cuttree.learn(data=d, TARGET="MMSE.M36", BASELINE="MMSE", tree=tree)
d$GROUP = as.factor(mod$group)
p = ggplot(d, aes(x = MMSE, y = MMSE.M36)) + geom_point(alpha=.7, aes(colour=GROUP), position = "jitter") + 
  geom_abline(linetype="dotted") + ggtitle(paste("MMSE.M36", "~", "MMSE"))

y_pred = interpect_by_cuttree.predict(mod, data=d)
d$MMSE.M36_pred = y_pred
loss_reg(d$MMSE.M36, y_pred)

d$GROUP = as.factor(attr(y_pred, "group"))
p = ggplot(d, aes(x = MMSE, y = MMSE.M36_pred)) + geom_point(alpha=.7, aes(colour=GROUP), position = "jitter") + 
  geom_abline(linetype="dotted") + ggtitle(paste("MMSE.M36", "~", "MMSE"))

}