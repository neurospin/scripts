require(ggplot2)
require(glmnet)
library(rpart)
library(ipred) # install.packages("ipred")
library(XLConnect) # install.packages("XLConnect")
library(RColorBrewer)
# read
#wb = loadWorkbook(filename)
#data1 = readWorksheet(wb,sheet1,...)
#data2 = readWorksheet(wb,sheet2,...)
# write
#wb = loadWorkbook(filename)
#createSheet(wb, sheet1)
#writeWorksheet(wb, data1, sheet1, ...)
#saveWorkbook(wb)

#library(RColorBrewer)
#library(plyr)

SRC = paste(Sys.getenv("HOME"),"git/scripts/2013_mescog/proj_predict_cog_decline",sep="/")
BASE_DIR = "/neurospin/mescog/proj_predict_cog_decline"

# INPUT ---
INPUT_DATA = paste(BASE_DIR, "data", "dataset_clinic_niglob_20140728_nomissing_BPF-LLV_imputed.csv", sep="/")

# OUTPUT ---
#OUTPUT = paste(BASE_DIR, sub(".csv", "", sub("dataset_clinic_niglob_", "", basename(INPUT_DATA))), "enet", "M36", sep="/")
OUTPUT = paste(BASE_DIR, sub(".csv", "", sub("dataset_clinic_niglob_", "", basename(INPUT_DATA))), "2015-09", sep="/")

if (!file.exists(OUTPUT)) dir.create(OUTPUT)
VALIDATION = "CV"
#VALIDATION = "All"
#VALIDATION = "FR-GE"
RM_TEST_OUTLIERS = FALSE

source(paste(SRC,"utils.R",sep="/"))

# rsync -azvun --delete /neurospin/mescog/proj_predict_cog_decline ~/data/
# rsync -azvun --delete  ~/data/proj_predict_cog_decline /neurospin/mescog/
################################################################################################
## Simple rules
################################################################################################

simple_prediction<-function(target, d, mod="learn", model=NULL){
  # target = "MMSE.CHANGE"
  # target = "MRS.CHANGE"
  # target = "TMTB_TIME.CHANGE"
  # target = "MDRS_TOTAL.CHANGE"
  # model = simple_prediction(target, d=db$DB, mod="learn")
  # simple_prediction(target, d=db$DB, mod="predict", model=model)
  if(target == "MMSE.CHANGE"){
    m1 = d$LLV>=1964
    m2 = (d$LLV< 1964 & d$MMSE>=27.5) | (d$LLV<1964 & d$MMSE< 27.5 & d$BPF< 0.7645)
    m3 = (d$LLV< 1964 & d$MMSE< 27.5 & d$BPF>=0.7645)
  }
  if(target == "MRS.CHANGE"){
    m1 = (d$MRS>=1.5) & (d$LLV< 1251)
    m2 = (d$MRS< 1.5) & (d$BPF>=0.8572)
    m3 = !(m1 | m2)
  }
  if(target == "TMTB_TIME.CHANGE"){
    m1 = d$TMTB_TIME>=173
    m2 = (d$TMTB_TIME<173) & (d$LLV<394.9)
    m3 = (d$TMTB_TIME<173) & (d$LLV>=394.9)
  }
  if(target == "MDRS_TOTAL.CHANGE"){
    m1 = (d$LLV>=1632) | (d$LLV<1632 & d$BPF<0.749)
    m2 = !m1
    m3 = !(m1 | m2)
  }
  if(sum(m1 | m2 | m3) != dim(d)[1]){
    print("Error")
  }
  if(mod == "learn"){
    return(c(mean(d[, target][m1], na.rm=TRUE), mean(d[, target][m2], na.rm=TRUE), mean(d[, target][m3], na.rm=TRUE)))
  }
  pred = rep(NA, dim(d)[1])
  pred[m1] = model[1]
  pred[m2] = model[2]
  pred[m3] = model[3]
  return(pred)
}


################################################################################################
## READ INPUT
################################################################################################
db = read_db(INPUT_DATA)
#db$DB

db$DB$TMTB_TIME.CHANGE = (db$DB$TMTB_TIME.M36 - db$DB$TMTB_TIME)
db$DB$MDRS_TOTAL.CHANGE = (db$DB$MDRS_TOTAL.M36 - db$DB$MDRS_TOTAL)
db$DB$MRS.CHANGE = (db$DB$MRS.M36 - db$DB$MRS)
db$DB$MMSE.CHANGE = (db$DB$MMSE.M36 - db$DB$MMSE)

################################################################################################
## SIMPLE ENET PREDICTION NO PLOT/PERM etc.
################################################################################################
# M36 ~ M0
if(FALSE){
PNZEROS = 0.5# c(.1, .25 , .5, .75)
ALPHAS = 0.25# seq(0, 1, .25)
#SEEDS = seq(1, 100)
if(VALIDATION == "CV") SEEDS = 11
if(VALIDATION == "FR-GE") SEEDS = 50
TARGETS = db$col_targets
}
if(TRUE){
# CHANGES ~ M0
PNZEROS = 0.5# c(.1, .25 , .5, .75)
ALPHAS = 0.25# seq(0, 1, .25)
SEEDS = 46#seq(1, 100)
#SEED ALPHA PNZERO diff_te_mu diff_te_sd
#46   46  0.25    0.5  0.1919573  0.1745975
#if(VALIDATION == "CV") SEEDS = 11
#if(VALIDATION == "FR-GE") SEEDS = 50
TARGETS = c("TMTB_TIME.CHANGE", "MDRS_TOTAL.CHANGE", "MRS.CHANGE", "MMSE.CHANGE")
}

NPERM = 1#1000
NFOLD = 5
NFOLD = 10

length(PNZEROS) * length(ALPHAS) * length(SEEDS) * NPERM * 4 * 4
FORGET = TRUE

SETTINGS = list("BASELINE"       = c(),
                "BASELINE+NIGLOB"       = db$col_niglob,
                "BASELINE+NIGLOBFULL"       = c(db$col_niglob, db$col_niglob_full),
                "BASELINE+CLINIC"       = db$col_clinic,
                "BASELINE+CLINIC+NIGLOB"= c(db$col_clinic, db$col_niglob),
                "BASELINE+CLINIC+NIGLOBFULL"= c(db$col_clinic, db$col_niglob, db$col_niglob_full))
SETTINGS2 = list("BASELINE"       = c(),
                "BASELINE+LLV"       = "LLV",
                "BASELINE+BPF"       = "BPF",
                "BASELINE+MBcount"       = "MBcount",
                "BASELINE+LLV+BPF"       = c("LLV", "BPF"),
                "BASELINE+LLV+BPF+MBcount"       = c("LLV", "BPF", "MBcount"))
RESULTS_TAB_ALL = NULL
RESULTS_TAB_MEAN = NULL
COMPARISONS = NULL

RESULTS = list()

baselines = unlist(lapply(strsplit(TARGETS, "[.]"), function(x)x[1]))
PREDICTORS_FULL = c(baselines, SETTINGS[["BASELINE+CLINIC+NIGLOBFULL"]])

#seed=11
for(TARGET in TARGETS){
  RESULTS[[TARGET]] = list()
  #TARGET = "TMTB_TIME.M36"
  #TARGET = "MDRS_TOTAL.M36"; TARGET =  "MRS.M36"; TARGET =          "MMSE.M36" 
  # TARGET = "MMSE.CHANGE"
  cat("** TARGET:", TARGET, "**\n" )
for(PREDICTORS_STR in names(SETTINGS)){
  RESULTS[[TARGET]][[PREDICTORS_STR]] = list()
  #PREDICTORS_STR = "BASELINE+CLINIC+NIGLOB"
  #PREDICTORS_STR = "BASELINE+NIGLOB"
  #print(PREDICTORS)
  #PREXIF = paste(OUTPUT, "/",TARGET, "~", PREDICTORS_STR , sep="")
  #cat(PREXIF,"\n")
  cat("** PREDICTORS_STR:", PREDICTORS_STR, "**\n" )
  BASELINE = strsplit(TARGET, "[.]")[[1]][1]
  PREDICTORS = unique(c(BASELINE, SETTINGS[[PREDICTORS_STR]]))
for(ALPHA in ALPHAS){
  # ALPHA = ALPHAS
  #cat(" ** ALPHA:",ALPHA,"**\n")
  RESULTS[[TARGET]][[PREDICTORS_STR]][[as.character(ALPHA)]] = list()
for(PNZERO in PNZEROS){
    #cat(" ** PNZERO:",PNZERO,"**\n")
    RESULTS[[TARGET]][[PREDICTORS_STR]][[as.character(ALPHA)]][[as.character(PNZERO)]] = list()
for(SEED in SEEDS){
  #cat("  ** SEED:", SEED, "**\n" )
  RESULTS[[TARGET]][[PREDICTORS_STR]][[as.character(ALPHA)]][[as.character(PNZERO)]][[SEED]] = list()
  set.seed(SEED)
  #pdf(paste(OUTPUT, "/",TARGET, "_datasets_qqplot.pdf", sep=""))
  #qqplot(D_tr[, TARGET], D_te[, TARGET], main=TARGET)
  #dev.off()
  #cat("=== ", TARGET, " ===\n")
  #print(D_tr_summary)
  #print(D_te_summary)
for(PERM in 1:NPERM){
  DB = db$DB
  if(PERM != 1){
    DB[!is.na(DB[, TARGET]), TARGET] = sample(DB[!is.na(DB[, TARGET]), TARGET])
  }
  #cat("    ** PERM:", PERM, "**\n" )
  RESULTS[[TARGET]][[PREDICTORS_STR]][[as.character(ALPHA)]][[as.character(PNZERO)]][[SEED]][[PERM]] = list()
  #SPLITS = split_db_site_stratified_with_same_target_distribution_rm_na(DB, TARGET)
  if(VALIDATION == "CV"){
    SPLITS = kfold_site_stratified_rm_na(DB, TARGET, k=NFOLD)
  }
  if(VALIDATION == "FR-GE"){
    SPLITS = twofold_bysite_rm_na(DB, TARGET)
  }
  if(VALIDATION == "All"){
    D = DB[!is.na(DB[, TARGET]),]
    SPLITS = list(list(tr=D, te=D))
  }
  
for(FOLD in 1:length(SPLITS)){
  #cat("   ** fold:", FOLD, "**\n" )
  RESULTS[[TARGET]][[PREDICTORS_STR]][[as.character(ALPHA)]][[as.character(PNZERO)]][[SEED]][[PERM]][[FOLD]] = list()
  D_tr = SPLITS[[FOLD]]$tr
  D_te = SPLITS[[FOLD]]$te
  if(RM_TEST_OUTLIERS){
    idx_test_keep =
      D_te[, BASELINE] <= max(D_tr[, BASELINE], na.rm=TRUE) &
      D_te[, TARGET]   <= max(D_tr[, TARGET], na.rm=TRUE)   &
      D_te[, BASELINE] >= min(D_tr[, BASELINE], na.rm=TRUE) &
      D_te[, TARGET]   >= min(D_tr[, TARGET], na.rm=TRUE)
    D_te = D_te[idx_test_keep, ]
  }
  X_tr = as.matrix(D_tr[, PREDICTORS])
  y_true_tr = D_tr[, TARGET]
  X_te = as.matrix(D_te[, PREDICTORS])
  y_true_te = D_te[, TARGET]
  #y = c(y_true_tr, y_true_te)

  ## Learn
  ## -----

  ## Enet ##
  if(dim(X_tr)[2]>1){
  cv_glmnet = cv.glmnet(X_tr, y_true_tr, alpha=ALPHA)
  lambda = cv_glmnet$lambda.min # == cv_glmnet$lambda[which.min(cv_glmnet$cvm)]
  enet_nzero = cv_glmnet$nzero[which.min(cv_glmnet$cvm)][[1]]
  enet_nzero_min = max(round(dim(X_tr)[2]*PNZERO), 2)
  if(enet_nzero < enet_nzero_min)
    lambda = cv_glmnet$lambda[which(cv_glmnet$nzero > enet_nzero_min)[1]]
  # if cannot find such lambda take the last one (least penalization)
  if(is.na(lambda)) lambda = cv_glmnet$lambda[length(cv_glmnet$lambda)]
  mod_enet = glmnet(X_tr, y_true_tr, lambda=lambda, alpha=ALPHA)
  enet_nzero = sum(mod_enet$beta!=0)
  coef_enet = as.double(mod_enet$beta); names(coef_enet) = rownames(mod_enet$beta); coef_enet = coef_enet[coef_enet!=0]
  }

  ## GLM ##
  formula = formula(paste(TARGET,"~", paste(PREDICTORS, collapse='+')))
  mod_glm = lm(formula, data=D_tr)
  coef_glm = mod_glm$coefficients
  
  ## rpart  ##
  rmse_small = errorest(formula, data=D_tr, model=rpart, predict=function(object, newdata, cp=.05) predict(prune(object, cp=cp), newdata))
  rmse_full = errorest(formula, data=D_tr, model=rpart)
  cat("err small-full=", (rmse_small$error^2 - rmse_full$error^2) / (rmse_full$error^2), "\n")
  mod_rpart = prune(rpart(formula, data=D_tr), cp=.05)
  
  ## simple ##
  mod_simple = simple_prediction(target=TARGET, d=D_tr, mod="learn", model=NULL)
  
  ## Prediction
  ## ----------
  
  ## ENET ##
  if(dim(X_tr)[2]>1){
  y_pred_te_enet = predict(mod_enet, X_te)
  y_pred_tr_enet = predict(mod_enet, X_tr)
  loss_te_enet = loss_reg(y_true_te, y_pred_te_enet, NULL, suffix="te")
  #loss_te_enet = loss_reg(y_true_te[idx_test_keep], y_pred_te_enet[idx_test_keep], NULL, suffix="te")
  loss_tr_enet = loss_reg(y_true_tr, y_pred_tr_enet, NULL, suffix="tr")
  }
  ## GLM ##
  y_pred_te_glm = predict(mod_glm, D_te)
  y_pred_tr_glm = predict(mod_glm, D_tr)
  loss_te_glm = loss_reg(y_true_te, y_pred_te_glm, NULL, suffix="te")
  #loss_te_glm = loss_reg(y_true_te[idx_test_keep], y_pred_te_glm[idx_test_keep], NULL, suffix="te")
  loss_tr_glm = loss_reg(y_true_tr, y_pred_tr_glm, NULL, suffix="tr")

  ## rpart ##
  y_pred_te_rpart = predict(mod_rpart, D_te)
  y_pred_tr_rpart = predict(mod_rpart, D_tr)
  loss_te_rpart = loss_reg(y_true_te, y_pred_te_rpart, NULL, suffix="te")
  loss_tr_rpart = loss_reg(y_true_tr, y_pred_tr_rpart, NULL, suffix="tr")

  ## simple ##
  y_pred_te_simple = simple_prediction(target=TARGET, d=D_te, mod="predict", model=mod_simple)
  y_pred_tr_simple = simple_prediction(target=TARGET, d=D_tr, mod="predict", model=mod_simple)
  loss_te_simple = loss_reg(y_true_te, y_pred_te_simple, NULL, suffix="te")
  loss_tr_simple = loss_reg(y_true_tr, y_pred_tr_simple, NULL, suffix="tr")
  
  ## Store results in global strurture ##
  ## GLM ##
  result_glm = list(FOLD=FOLD, PREDICTORS_STR=PREDICTORS_STR, TARGET=TARGET,
  mod=mod_glm, coef_glm=coef_glm,
  y_pred_te=y_pred_te_glm,
  y_pred_tr=y_pred_tr_glm,
  loss_te=loss_te_glm,
  loss_tr=loss_tr_glm,       
  X_tr=X_tr, X_te=X_te, y_true_tr=y_true_tr, y_true_te=y_true_te,
  D_tr=D_tr, D_te=D_te)
  RESULTS[[TARGET]][[PREDICTORS_STR]][[as.character(ALPHA)]][[as.character(PNZERO)]][[SEED]][[PERM]][[FOLD]][["GLM"]] = result_glm

  ## Enet ##
  if(dim(X_tr)[2]>1){
    result_enet = list(# ENET
      mod = mod_enet, coef_enet=coef_enet, 
      y_pred_te = y_pred_te_enet,
      y_pred_tr = y_pred_tr_enet,
      loss_te = loss_te_enet,
      loss_tr = loss_tr_enet,
      X_tr=X_tr, X_te=X_te, y_true_tr=y_true_tr, y_true_te=y_true_te, 
      D_tr=D_tr, D_te=D_te)           
    RESULTS[[TARGET]][[PREDICTORS_STR]][[as.character(ALPHA)]][[as.character(PNZERO)]][[SEED]][[PERM]][[FOLD]][["ENET"]] = result_enet
  }

  ## rpart ##
  result_rpart = list(FOLD=FOLD, PREDICTORS_STR=PREDICTORS_STR, TARGET=TARGET,
                    mod=mod_rpart, coef_rpart=NA,
                    y_pred_te=y_pred_te_rpart,
                    y_pred_tr=y_pred_tr_rpart,
                    loss_te=loss_te_rpart,
                    loss_tr=loss_tr_rpart,       
                    X_tr=X_tr, X_te=X_te, y_true_tr=y_true_tr, y_true_te=y_true_te,
                    D_tr=D_tr, D_te=D_te)
  RESULTS[[TARGET]][[PREDICTORS_STR]][[as.character(ALPHA)]][[as.character(PNZERO)]][[SEED]][[PERM]][[FOLD]][["RPART"]] = result_rpart

  ## simple ##
  result_simple = list(FOLD=FOLD, PREDICTORS_STR=PREDICTORS_STR, TARGET=TARGET,
                      mod=mod_simple, coef_simple=NA,
                      y_pred_te=y_pred_te_simple,
                      y_pred_tr=y_pred_tr_simple,
                      loss_te=loss_te_simple,
                      loss_tr=loss_tr_simple,       
                      X_tr=X_tr, X_te=X_te, y_true_tr=y_true_tr, y_true_te=y_true_te,
                      D_tr=D_tr, D_te=D_te)
  RESULTS[[TARGET]][[PREDICTORS_STR]][[as.character(ALPHA)]][[as.character(PNZERO)]][[SEED]][[PERM]][[FOLD]][["SIMPLE"]] = result_simple
  
  ## Store results in global RESULTS_TAB_ALL ##

  get_coef<-function(coefs=NA){
#     get_coef(coef_glm)
#     get_coef(coef_enet)
#     get_coef()
    coefs_all = rep(NA, length(PREDICTORS_FULL))
    names(coefs_all)  = PREDICTORS_FULL
    if(!is.null(coefs)){
      coefs_names = names(coefs)[names(coefs) %in% PREDICTORS_FULL]
      coefs_all[coefs_names] = coefs[coefs_names]
    }
    return(coefs_all)
  }
  
  ## glm ##    
  library(gdata)
  res_glm = data.frame(TARGET=TARGET, PREDICTORS=PREDICTORS_STR, ALPHA=ALPHA, PNZERO=PNZERO, SEED=SEED, PERM=PERM, FOLD=FOLD, MODEL="GLM",
       dim=paste(dim(X_tr), collapse="x"),
       as.list(c(loss_te_glm, mse_te_macro=NA,    r2_te_macro=NA,   cor_te_macro=NA,
                 mse_te_se=NA, r2_te_se=NA, cor_te_se=NA, mse_te_ci=NA, r2_te_ci=NA, cor_te_ci=NA,
                 loss_tr_glm,
                 get_coef(coef_glm))))
  RESULTS_TAB_ALL = rbind(RESULTS_TAB_ALL, res_glm)

  ##  enet ##
  if(dim(X_tr)[2]>1){
  res_enet = data.frame(TARGET=TARGET, PREDICTORS=PREDICTORS_STR, ALPHA=ALPHA, PNZERO=PNZERO, SEED=SEED, PERM=PERM, FOLD=FOLD, MODEL="ENET", 
            dim=paste(dim(X_tr), collapse="x"),
            as.list(c(loss_te_enet, mse_te_macro=NA,    r2_te_macro=NA,   cor_te_macro=NA,
                      mse_te_se=NA, r2_te_se=NA, cor_te_se=NA, mse_te_ci=NA, r2_te_ci=NA, cor_te_ci=NA,
                      loss_tr_enet,
                    get_coef(coef_enet))))
  RESULTS_TAB_ALL = rbind(RESULTS_TAB_ALL, res_enet)

  ## rpart ##
  res_rpart = data.frame(TARGET=TARGET, PREDICTORS=PREDICTORS_STR, ALPHA=ALPHA, PNZERO=PNZERO, SEED=SEED, PERM=PERM, FOLD=FOLD, MODEL="RPART",
                   dim=paste(dim(X_tr), collapse="x"),
                   as.list(c(loss_te_rpart, mse_te_macro=NA,    r2_te_macro=NA,   cor_te_macro=NA,
                              mse_te_se=NA, r2_te_se=NA, cor_te_se=NA, mse_te_ci=NA, r2_te_ci=NA, cor_te_ci=NA,
                             loss_tr_rpart,
                             get_coef())))
  RESULTS_TAB_ALL = rbind(RESULTS_TAB_ALL, res_rpart)
  
  }
} # FOLD
  ## Average scores over folds
  ## -------------------------

  perm_curr = RESULTS[[TARGET]][[PREDICTORS_STR]][[as.character(ALPHA)]][[as.character(PNZERO)]][[SEED]][[PERM]]
  y_true_tr_glm = c()
  y_pred_tr_glm = c()
  y_true_te_glm = c()
  y_pred_te_glm = c()
  loss_te_glm = NULL
  coef_glm = NULL
  y_true_te_enet = c()
  y_pred_te_enet = c()
  y_true_tr_enet = c()
  y_pred_tr_enet = c()
  loss_te_enet = NULL
  coef_enet = NULL
  y_true_te_rpart = c()
  y_pred_te_rpart = c()
  y_true_tr_rpart = c()
  y_pred_tr_rpart = c()
  loss_te_rpart = NULL
  y_true_te_simple = c()
  y_pred_te_simple = c()
  y_true_tr_simple = c()
  y_pred_tr_simple = c()
  loss_te_simple = NULL

  for(FOLD in 1:length(perm_curr)){
    cv_curr = RESULTS[[TARGET]][[PREDICTORS_STR]][[as.character(ALPHA)]][[as.character(PNZERO)]][[SEED]][[PERM]][[FOLD]]
    y_true_tr_glm = c(y_true_tr_glm, cv_curr[["GLM"]]$y_true_tr)
    y_pred_tr_glm = c(y_pred_tr_glm, cv_curr[["GLM"]]$y_pred_tr)
    y_true_te_glm = c(y_true_te_glm, cv_curr[["GLM"]]$y_true_te)
    y_pred_te_glm = c(y_pred_te_glm, cv_curr[["GLM"]]$y_pred_te)
    loss_te_glm = rbind(loss_te_glm, cv_curr[["GLM"]]$loss_te)
    coef_glm = rbind(coef_enet, get_coef(cv_curr[["GLM"]]$coef))
    try({
      y_true_tr_enet= c(y_true_tr_enet, cv_curr[["ENET"]]$y_true_tr);
      y_pred_tr_enet= c(y_pred_tr_enet, cv_curr[["ENET"]]$y_pred_tr);
      y_true_te_enet= c(y_true_te_enet, cv_curr[["ENET"]]$y_true_te);
      y_pred_te_enet= c(y_pred_te_enet, cv_curr[["ENET"]]$y_pred_te)
      loss_te_enet = rbind(loss_te_enet, cv_curr[["ENET"]]$loss_te)
      coef_enet = rbind(coef_enet, get_coef(cv_curr[["ENET"]]$coef))
      })
    y_true_tr_rpart = c(y_true_tr_rpart, cv_curr[["RPART"]]$y_true_tr)
    y_pred_tr_rpart = c(y_pred_tr_rpart, cv_curr[["RPART"]]$y_pred_tr)
    y_true_te_rpart = c(y_true_te_rpart, cv_curr[["RPART"]]$y_true_te)
    y_pred_te_rpart = c(y_pred_te_rpart, cv_curr[["RPART"]]$y_pred_te)
    loss_te_rpart = rbind(loss_te_rpart, cv_curr[["RPART"]]$loss_te)

    y_true_tr_simple = c(y_true_tr_simple, cv_curr[["SIMPLE"]]$y_true_tr)
    y_pred_tr_simple = c(y_pred_tr_simple, cv_curr[["SIMPLE"]]$y_pred_tr)
    y_true_te_simple = c(y_true_te_simple, cv_curr[["SIMPLE"]]$y_true_te)
    y_pred_te_simple = c(y_pred_te_simple, cv_curr[["SIMPLE"]]$y_pred_te)
    loss_te_simple = rbind(loss_te_simple, cv_curr[["SIMPLE"]]$loss_te)
  }

  if(!all((y_true_te_glm == y_true_te_enet) & (y_true_te_enet == y_true_te_rpart) & (y_true_te_rpart == y_true_te_simple))){
    print("Error: y_true_te* are not all equals ")
  }
  y_true_te = y_true_te_glm

  ## GLM ##
  loss_te_glm_micro = loss_reg(y_true_te_glm, y_pred_te_glm, suffix="te") # micro: average agregated
  # scores
  loss_te_glm_macro = apply(loss_te_glm, 2, mean)
  names(loss_te_glm_macro) <- paste(names(loss_te_glm_macro), "macro", sep="_")
  loss_te_se_glm = apply(loss_te_glm, 2, sd) / sqrt(nrow(loss_te_glm))
  names(loss_te_se_glm) <- paste(names(loss_te_se_glm), "se", sep="_")
  # one-sided confidence interval
  loss_te_ci_glm = qt(0.975, df=nrow(loss_te_glm)-1)*loss_te_se_glm / sqrt(nrow(loss_te_glm))
  names(loss_te_ci_glm) <- sub("_se", "_ci", names(loss_te_ci_glm))
  RESULTS_TAB_MEAN = rbind(RESULTS_TAB_MEAN, 
    data.frame(TARGET=TARGET, PREDICTORS=PREDICTORS_STR, ALPHA=ALPHA, PNZERO=PNZERO, SEED=SEED, PERM=PERM, FOLD="Average", MODEL="GLM", 
             dim=paste(dim(X_tr), collapse="x"),
             as.list(c(loss_te_glm_micro,
                       loss_te_glm_macro,
                       loss_te_se_glm, loss_te_ci_glm,
                       bic=bic(sum((y_true_te - y_pred_te_glm)^2), n = length(y_true_te), p=length(PREDICTORS)),
                       loss_reg(y_true_tr_glm, y_pred_tr_glm, suffix="tr"),
                       colMeans(coef_glm, na.rm=TRUE)))))

  ## Enet ##
  if(!is.null(loss_te_enet)){
    loss_te_enet_micro = loss_reg(y_true_te_enet, y_pred_te_enet, suffix="te")
    # scores
    loss_te_enet_macro = apply(loss_te_enet, 2, mean)
    names(loss_te_enet_macro) <- paste(names(loss_te_enet_macro), "macro", sep="_")
    loss_te_se_enet = apply(loss_te_enet, 2, sd) / sqrt(nrow(loss_te_enet))
    names(loss_te_se_enet) <- paste(names(loss_te_se_enet), "se", sep="_")
    # one-sided confidence interval
    loss_te_ci_enet = qt(0.975, df=nrow(loss_te_enet)-1)*loss_te_se_enet / sqrt(nrow(loss_te_enet))
    names(loss_te_ci_enet) <- sub("_se", "_ci", names(loss_te_ci_enet))
    RESULTS_TAB_MEAN = rbind(RESULTS_TAB_MEAN, 
      data.frame(TARGET=TARGET, PREDICTORS=PREDICTORS_STR, ALPHA=ALPHA, PNZERO=PNZERO, SEED=SEED, PERM=PERM, FOLD="Average", MODEL="ENET", 
                   dim=paste(dim(X_tr), collapse="x"),
                   as.list(c(loss_te_enet_micro,
                             loss_te_enet_macro,
                             loss_te_se_enet, loss_te_ci_enet,
                             bic=bic(sum((y_true_te - y_pred_te_enet)^2), n = length(y_true_te), p=length(PREDICTORS)),
                             loss_reg(y_true_tr_enet, y_pred_tr_enet, suffix="tr"),
                             colMeans(coef_enet, na.rm=TRUE)))))
  }

  ## rpart ##
  loss_te_rpart_micro = loss_reg(y_true_te_rpart, y_pred_te_rpart, suffix="te")
  # scores
  loss_te_rpart_macro = apply(loss_te_rpart, 2, mean)
  names(loss_te_rpart_macro) <- paste(names(loss_te_rpart_macro), "macro", sep="_")
  loss_te_se_rpart = apply(loss_te_rpart, 2, sd) / sqrt(nrow(loss_te_rpart))
  names(loss_te_se_rpart) <- paste(names(loss_te_se_rpart), "se", sep="_")
  # one-sided confidence interval
  loss_te_ci_rpart = qt(0.975, df=nrow(loss_te_rpart)-1)*loss_te_se_rpart / sqrt(nrow(loss_te_rpart))
  names(loss_te_ci_rpart) <- sub("_se", "_ci", names(loss_te_ci_rpart))
  
  RESULTS_TAB_MEAN = rbind(RESULTS_TAB_MEAN, 
    data.frame(TARGET=TARGET, PREDICTORS=PREDICTORS_STR, ALPHA=ALPHA, PNZERO=PNZERO, SEED=SEED, PERM=PERM, FOLD="Average", MODEL="RPART", 
               dim=paste(dim(X_tr), collapse="x"),
               as.list(c(loss_te_rpart_micro,
                         loss_te_rpart_macro,
                        loss_te_se_rpart, loss_te_ci_rpart,
                        bic=bic(sum((y_true_te - y_pred_te_rpart)^2), n = length(y_true_te), p=length(PREDICTORS)),
                        loss_reg(y_true_tr_rpart, y_pred_tr_rpart, suffix="tr"),
                        get_coef()))))

  ## simple ##
  loss_te_simple_micro = loss_reg(y_true_te_simple, y_pred_te_simple, suffix="te")
  # scores
  loss_te_simple_macro = apply(loss_te_simple, 2, mean)
  names(loss_te_simple_macro) <- paste(names(loss_te_simple_macro), "macro", sep="_")
  loss_te_se_simple = apply(loss_te_simple, 2, sd) / sqrt(nrow(loss_te_simple))
  names(loss_te_se_simple) <- paste(names(loss_te_se_simple), "se", sep="_")
  # one-sided confidence interval
  loss_te_ci_simple = qt(0.975, df=nrow(loss_te_simple)-1)*loss_te_se_simple / sqrt(nrow(loss_te_simple))
  names(loss_te_ci_simple) <- sub("_se", "_ci", names(loss_te_ci_simple))
  rss = sum((y_true_te - y_pred_te_simple)^2)
  cat("bic simple", bic(rss, n=length(y_true_te), p=3), rss, length(y_true_te), "\n")
  RESULTS_TAB_MEAN = rbind(RESULTS_TAB_MEAN, 
                           data.frame(TARGET=TARGET, PREDICTORS=PREDICTORS_STR, ALPHA=ALPHA, PNZERO=PNZERO, SEED=SEED, PERM=PERM, FOLD="Average", MODEL="SIMPLE", 
                                      dim=paste(dim(X_tr), collapse="x"),
                                      as.list(c(loss_te_simple_micro,
                                                loss_te_simple_macro,
                                                loss_te_se_simple, loss_te_ci_simple,
                                                bic=bic(sum((y_true_te - y_pred_te_simple)^2), n=length(y_true_te), p=3),
                                                loss_reg(y_true_tr_simple, y_pred_tr_simple, suffix="tr"),
                                                get_coef()))))


  RESULTS[[TARGET]][[PREDICTORS_STR]][[as.character(ALPHA)]][[as.character(PNZERO)]][[SEED]][[PERM]][["Average"]][["GLM"]]    = list(y_pred_te=y_pred_te_glm,    y_true_te=y_true_te)
  RESULTS[[TARGET]][[PREDICTORS_STR]][[as.character(ALPHA)]][[as.character(PNZERO)]][[SEED]][[PERM]][["Average"]][["ENET"]]   = list(y_pred_te=y_pred_te_enet,   y_true_te=y_true_te)
  RESULTS[[TARGET]][[PREDICTORS_STR]][[as.character(ALPHA)]][[as.character(PNZERO)]][[SEED]][[PERM]][["Average"]][["RPART"]]  = list(y_pred_te=y_pred_te_rpart,  y_true_te=y_true_te)
  RESULTS[[TARGET]][[PREDICTORS_STR]][[as.character(ALPHA)]][[as.character(PNZERO)]][[SEED]][[PERM]][["Average"]][["SIMPLE"]] = list(y_pred_te=y_pred_te_simple, y_true_te=y_true_te)

  ## Comparisons
  ## -----------

  # Nonparametric Wilcoxon signed ranks tests on the paired differences of absolute errors of predictions.
  if(dim(X_tr)[2]>1){
    y_pred_te_other = y_pred_te_enet
    loss_te_other_micro = loss_te_enet_micro
  }else{
    y_pred_te_other = y_pred_te_glm
    loss_te_other_micro = loss_te_glm_micro
  }
  ## rpart vs Enet ##
  wt = wilcox.test(abs(y_pred_te_rpart - y_true_te) - abs(y_pred_te_other - y_true_te))
  tt = t.test(abs(y_pred_te_rpart - y_true_te) - abs(y_pred_te_other - y_true_te))
  #cat("rpart vs enet", dim(X_tr)[1], wt$statistic, wt$p.value, "\n")
  delta_ratio_loss = (loss_te_rpart_micro - loss_te_other_micro) / loss_te_other_micro
  # - + + => rpart>other
  names(delta_ratio_loss) = paste(names(delta_ratio_loss), "m1_m2divm2" , sep="_")
  COMPARISONS = rbind(COMPARISONS,
  data.frame(TARGET=TARGET, PREDICTORS=PREDICTORS_STR, ALPHA=ALPHA, PNZERO=PNZERO, SEED=SEED, PERM=PERM, M1="RPART", M2="ENET",
             as.list(c(delta_ratio_loss, pval_ttest=tt$p.value, pval_wilcox=wt$p.value))))

  ## simple vs Enet ##
  wt = wilcox.test(abs(y_pred_te_simple - y_true_te) - abs(y_pred_te_other - y_true_te))
  tt = t.test(abs(y_pred_te_simple - y_true_te) - abs(y_pred_te_other - y_true_te))
  #cat("simple vs enet", dim(X_tr)[1], wt$statistic, wt$p.value, "\n")
  delta_ratio_loss = (loss_te_simple_micro - loss_te_other_micro) / loss_te_other_micro
  # - + + => simple>other
  names(delta_ratio_loss) = paste(names(delta_ratio_loss), "m1_m2divm2" , sep="_")
  COMPARISONS = rbind(COMPARISONS,
                      data.frame(TARGET=TARGET, PREDICTORS=PREDICTORS_STR, ALPHA=ALPHA, PNZERO=PNZERO, SEED=SEED, PERM=PERM, M1="SIMPLE", M2="ENET",
                                 as.list(c(delta_ratio_loss, pval_ttest=tt$p.value, pval_wilcox=wt$p.value))))

  if(FORGET && PERM!=1)
    RESULTS[[TARGET]][[PREDICTORS_STR]][[as.character(ALPHA)]][[as.character(PNZERO)]][[SEED]][[PERM]] = list()
} # PERM
} # SEED
} # ALPHA
} # PNZERO
} # PREDICTORS_STR
} # TARGET

## --------------------------------------------------------------------------------------------------
## Compare predictors models
# 
COMPARISONS_MODELS = NULL

for(TARGET in TARGETS){
y_true_te = RESULTS[[TARGET]][["BASELINE"]][[as.character(ALPHA)]][[as.character(PNZERO)]][[SEED]][[PERM]][["Average"]][["GLM"]]$y_true_te
t6 =  RESULTS[[TARGET]][["BASELINE+CLINIC+NIGLOBFULL"]][[as.character(ALPHA)]][[as.character(PNZERO)]][[SEED]][[PERM]][["Average"]][["ENET"]]$y_true_te
if(!all(y_true_te == t6))print("Error true are not the same")

m1 = RESULTS[[TARGET]][["BASELINE"]][[as.character(ALPHA)]][[as.character(PNZERO)]][[SEED]][[PERM]][["Average"]][["GLM"]]$y_pred_te
m1e = abs(m1 - y_true_te)
m2 = RESULTS[[TARGET]][["BASELINE+NIGLOB"]][[as.character(ALPHA)]][[as.character(PNZERO)]][[SEED]][[PERM]][["Average"]][["ENET"]]$y_pred_te
m2e = abs(m2 - y_true_te)
m3 = RESULTS[[TARGET]][["BASELINE+NIGLOBFULL"]][[as.character(ALPHA)]][[as.character(PNZERO)]][[SEED]][[PERM]][["Average"]][["ENET"]]$y_pred_te
m3e = abs(m3 - y_true_te)
m4 = RESULTS[[TARGET]][["BASELINE+CLINIC"]][[as.character(ALPHA)]][[as.character(PNZERO)]][[SEED]][[PERM]][["Average"]][["ENET"]]$y_pred_te
m4e = abs(m4 - y_true_te)
m5 = RESULTS[[TARGET]][["BASELINE+CLINIC+NIGLOB"]][[as.character(ALPHA)]][[as.character(PNZERO)]][[SEED]][[PERM]][["Average"]][["ENET"]]$y_pred_te
m5e = abs(m5 - y_true_te)
m6 = RESULTS[[TARGET]][["BASELINE+CLINIC+NIGLOBFULL"]][[as.character(ALPHA)]][[as.character(PNZERO)]][[SEED]][[PERM]][["Average"]][["ENET"]]$y_pred_te
m6e = abs(m6 - y_true_te)
spl = RESULTS[[TARGET]][["BASELINE+NIGLOB"]][[as.character(ALPHA)]][[as.character(PNZERO)]][[SEED]][[PERM]][["Average"]][["SIMPLE"]]$y_pred_te
sple = abs(spl - y_true_te)

abs_err = list(m1e, m2e, m3e, m4e, m5e, m6e)
which_min_enet = which.min(unlist(lapply(abs_err, function(x)sum(x^2))))
min_enet_abserr = abs_err[[which_min_enet]]

## rpart vs Enet ##
COMPARISONS_MODELS = rbind(COMPARISONS_MODELS,
rbind(
  data.frame(TARGET=TARGET, M1="BASELINE", M2="BASELINE+NIGLOB",
    pval_ttest = t.test(m1e - m2e)$p.value,
    pval_wilcox = wilcox.test(m1e - m2e)$p.value),

  data.frame(TARGET=TARGET, M1="BASELINE+NIGLOB", M2="BASELINE+NIGLOBFULL",
    pval_ttest = t.test(m2e - m3e)$p.value,
    pval_wilcox = wilcox.test(m2e - m3e)$p.value),

  data.frame(TARGET=TARGET, M1="BASELINE+CLINIC", M2="BASELINE+CLINIC+NIGLOB",
    pval_ttest = t.test(m4e - m5e)$p.value,
    pval_wilcox = wilcox.test(m4e - m5e)$p.value),

  data.frame(TARGET=TARGET, M1="BASELINE+CLINIC+NIGLOB", M2="BASELINE+CLINIC+NIGLOBFULL",
    pval_ttest = t.test(m5e - m6e)$p.value,
    pval_wilcox = wilcox.test(m5e - m6e)$p.value),

  data.frame(TARGET=TARGET, M1=paste("MinEnet", which_min_enet, sep="_"), M2="SIMPLE",
    pval_ttest = t.test(min_enet_abserr - sple)$p.value,
    pval_wilcox = wilcox.test(min_enet_abserr - sple)$p.value)))
}

# #cat("rpart vs enet", dim(X_tr)[1], wt$statistic, wt$p.value, "\n")
# delta_ratio_loss = (loss_te_rpart_micro - loss_te_other_micro) / loss_te_other_micro
# # - + + => rpart>other
# names(delta_ratio_loss) = paste(names(delta_ratio_loss), "m1_m2divm2" , sep="_")
# COMPARISONS = rbind(COMPARISONS,
#                     data.frame(TARGET=TARGET, PREDICTORS=PREDICTORS_STR, ALPHA=ALPHA, PNZERO=PNZERO, SEED=SEED, PERM=PERM, M1="RPART", M2="ENET",
#                                as.list(c(delta_ratio_loss, pval_ttest=tt$p.value, pval_wilcox=wt$p.value))))
# 
# 
# 
# "BASELINE", "BASELINE+NIGLOB", "BASELINE+NIGLOBFULL", "DUMMY1",
# "BASELINE+CLINIC", "BASELINE+CLINIC+NIGLOB", "BASELINE+CLINIC+NIGLOBFULL", "DUMMY2",
# "SIMPLE"

## --------------------------------------------------------------------------------------------------
## small differences essentially on MDRS between old an recomputed, don't know why, merge to simplify
keys = c("TARGET", "PREDICTORS", "MODEL")
columns =  c("mse_te","r2_te","cor_te","mse_tr","r2_tr","cor_tr","mse_te_se","r2_te_se","cor_te_se")
d_old = read.csv("/neurospin/mescog/proj_predict_cog_decline/20140728_nomissing_BPF-LLV_imputed/enet/CHANGE/RESULTS_TAB_CV.csv", as.is=TRUE)
d_old = d_old[(((d_old$PREDICTORS == "BASELINE") & (d_old$MODEL =="GLM")) | d_old$MODEL =="ENET") & d_old$FOLD == "ALL", c(keys, columns)]
d = RESULTS_TAB_MEAN
#d$TARGET = as.character(d$TARGET); d$PREDICTORS = as.character(d$PREDICTORS); d$MODEL = as.character(d$MODEL)

for (i in 1:nrow(d_old)){
  #print(d_old[i, ])
  msk = (d$TARGET == d_old[i, "TARGET"]) & (d$PREDICTORS == d_old[i, "PREDICTORS"]) & (d$MODEL == d_old[i, "MODEL"])
  if(sum(msk)!=1)cat("Error\n")
  print(abs(d[msk, columns] - d_old[i, columns]) / d_old[i, columns])
  d[msk, columns] = d_old[i, columns]
  #look up stuff using data from the row
  #write stuff to the file
}
d = RESULTS_TAB_MEAN
## --------------------------------------------------------------------------------------------------
## summary: the 7 models to be ploted
dlin = d[(((d$PREDICTORS == "BASELINE") & (d$MODEL =="GLM")) | d$MODEL =="ENET"), ]
dlin$PREDICTORS
dlin$MODEL_NB = mapvalues(dlin$PREDICTORS, 
          from = c("BASELINE", "BASELINE+NIGLOB", "BASELINE+NIGLOBFULL", "BASELINE+CLINIC", "BASELINE+CLINIC+NIGLOB", "BASELINE+CLINIC+NIGLOBFULL"), 
          to   = c(1,        2,               3,                   4,               5,                                 6))
dlin$MODEL_NB = as.integer(dlin$MODEL_NB)

dspl = d[(d$MODEL == "SIMPLE") & (d$PREDICTORS =="BASELINE+NIGLOB"), ]
dspl$MODEL_NB = 7

summary = rbind(dlin, dspl)

summary$CHANGE = mapvalues(summary$TARGET, 
                          from = c("TMTB_TIME.CHANGE", "MDRS_TOTAL.CHANGE", "MRS.CHANGE", "MMSE.CHANGE"), 
                          to   = c("TMTB",             "MDRS",              "mRS",        "MMSE"))


dim(summary)

summary = summary[order(summary$MODEL_NB), ]
summary = summary[order(summary$TARGET), ]
summary

colnames(summary) = mapvalues(colnames(summary),
  from = c("TMTB_TIME", "MDRS_TOTAL", "MRS", "MMSE", "AGE_AT_INCLUSION", "SEX",   "EDUCATION", "SYS_BP", "DIA_BP", "SMOKING", "LDL",    "HOMOCYSTEIN", "HBA1C", "CRP17", "ALCOHOL", "LLV", "BPF", "WMHV", "MBcount"),
  to   = c("TMTB",      "MDRS",       "mRS",	"MMSE",	"Age",	            "Gender",	"Education", "Sys bp",	"Dia bp",	"Smoking",	"LDL", "Homocy",     	"HBA1C",	"CRP",	"Alcohol", "Llv",	"BPF",	"WMHv",	"MBn")
)

## --------------------------------------------------------------------------------------------------

wb = loadWorkbook(paste(OUTPUT, "/RESULTS_TAB_",VALIDATION,".xlsx", sep=""), create=TRUE)
createSheet(wb, "CV")
writeWorksheet(wb, RESULTS_TAB_ALL, "CV")
createSheet(wb, "Average")
writeWorksheet(wb, RESULTS_TAB_MEAN, "Average")
createSheet(wb, "Average_with_old")
writeWorksheet(wb, d, "Average_with_old")
createSheet(wb, "Comparisons_enet-vs-rpart")
writeWorksheet(wb, COMPARISONS, "Comparisons_enet-vs-rpart")
createSheet(wb, "Comparisons_models")
writeWorksheet(wb, COMPARISONS_MODELS, "Comparisons_models")
createSheet(wb, "Summary")
writeWorksheet(wb, summary, "Summary")
saveWorkbook(wb)

save(RESULTS, file=paste(OUTPUT, "/RESULTS_",VALIDATION,".Rdata", sep=""))

## --------------------------------------------------------------------------------------------------
## PLOT
d = summary
summary$MODEL_NB = as.factor(summary$MODEL_NB)
palette = brewer.pal(6, "Paired")[c(1, 5, 3, 2, 6, 4)]

#palette = c(palette[1:3], "white", palette[4:6],  "white", "slategray4")
palette = c(palette[1:3],  palette[4:6],  "slategray4")

changep = ggplot(summary, aes(x = MODEL_NB, y = r2_te, fill=MODEL_NB)) +
  geom_bar(stat = "identity", position="dodge", limits=c(.1, 1)) +
  geom_errorbar(aes(ymin=r2_te-r2_te_se, ymax=r2_te+r2_te_se), width=.1) +
  #scale_y_continuous(expand=c(.1, 0))+
  facet_wrap(~target) + scale_fill_manual(values=palette) + ggtitle("CV") + theme(legend.position="none")
#  theme(legend.position="bottom", legend.direction="vertical")
x11(); print(changep)

